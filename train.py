import itertools
import json
import logging
import os
import time
from datetime import datetime

import wandb
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR
from matplotlib import pyplot as plt
from tqdm import tqdm

from dataset import GameDataset
from model import NeuralNetwork

BOARD_SIZE = 9
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
LOG_FOLDER = "logs"
PLOT_FOLDER = "plots"
CHECKPOINT_FOLDER = "checkpoints"
CONFIG_FOLDER = "configs"
TRAIN_DATA_PATH = "data/train_data_big.pkl"
VAL_DATA_PATH = "data/validation_data_big.pkl"


def setup_logger(log_file_path):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logger = logging.getLogger(log_file_path)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def save_config(config, run_name):
    os.makedirs(CONFIG_FOLDER, exist_ok=True)
    config_path = os.path.join(CONFIG_FOLDER, f"{run_name}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def train_loop(dataloader, model, loss_fn, optimizer, logger, epoch, scheduler=None):
    model.train()
    size = len(dataloader.dataset)
    losses, accuracies, accuracies_top3 = [], [], []

    for batch, (X, y, nm_color) in enumerate(dataloader):
        pred = model(X, nm_color)
        y = torch.reshape(y, (-1, 81))
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if isinstance(scheduler, OneCycleLR):
            scheduler.step()

        if batch % 1000 == 0:
            batch_size = len(X)
            # loss, current = loss.item(), batch * batch_size+ len(X)
            current = batch * batch_size + len(X)
            # correct = (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            correct = (pred.argmax(1) == y.argmax(1)).sum().item()
            accuracy = 100 * correct / batch_size

            # Top-3 Accuracy
            _, top3_pred = pred.topk(3, 1, True, True)
            correct_top3 = 0
            target = y.argmax(1).view(-1, 1)
            correct_top3 += top3_pred.eq(target).sum().item()
            accuracy_top3 = 100 * correct_top3 / batch_size

            losses.append(loss.item())
            accuracies.append(accuracy)
            accuracies_top3.append(accuracy_top3)

            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            logger.info(f"Accuracy: {100*correct/batch_size}")
            logger.info(f"Top-3 Accuracy: {accuracy_top3}")
            logger.info(f"max pred: {torch.max(pred[0])}")
            
            wandb.log({
                "train_loss": loss.item(),
                "train_accuracy": accuracy,
                "train_accuracy_top3": accuracy_top3,
                "epoch": epoch,
                "batch": batch,
                "step": (epoch - 1) * len(dataloader) + batch
            })

    return losses, accuracies, accuracies_top3


def validation_loop(dataloader, model, loss_fn, logger, epoch):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, correct, correct_top3 = 0, 0, 0

    with torch.no_grad():
        for X, y, nm_color in dataloader:
            pred = model(X, nm_color)
            y = torch.reshape(y, (-1, 81))
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            
            # Top-3 Accuracy, check if the correct move in top 3 move by the model
            _, top3_pred = pred.topk(3, 1, True, True)
            target = y.argmax(1).view(-1, 1)
            correct_top3 += top3_pred.eq(target).sum().item()

    val_loss /= num_batches
    accuracy = 100 * correct / size
    accuracy_top3 = 100 * correct_top3 / size

    logger.info(
        f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Top-3 Accuracy: {accuracy_top3:>0.1f}%, Avg loss: {val_loss:>8f} \n"
    )
    wandb.log({
        "val_loss": val_loss,
        "val_accuracy": accuracy,
        "val_accuracy_top3": accuracy_top3,
        "epoch": epoch
    })
    return val_loss, accuracy, accuracy_top3


def plot_and_save(logs, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    fig = plt.figure()
    for log_vals, label in logs:
        plt.plot(log_vals, label=label)
    plt.legend()
    plt.savefig(file_name)
    fig.clf()
    plt.close(fig)


def train(
    train_set,
    val_set,
    board_size,
    n_size,
    num_layer,
    learning_rate,
    epoch,
    batch_size,
    dropout,
    run_name,
    weight_decay,
    scheduler_type="ReduceLROnPlateau"
):

    # Setup run-specific paths
    run_log_path = os.path.join(LOG_FOLDER, f"{run_name}.log")
    run_plot_folder = os.path.join(PLOT_FOLDER, run_name)
    run_checkpoint_folder = os.path.join(CHECKPOINT_FOLDER, run_name)

    os.makedirs(run_plot_folder, exist_ok=True)
    os.makedirs(run_checkpoint_folder, exist_ok=True)

    logger = setup_logger(run_log_path)

    training_generator = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    val_generator = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True
    )

    model = NeuralNetwork(board_size, n_size, num_layer, dropout).to(DEVICE)
    model.compile()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    
    scheduler = None
    if scheduler_type == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.50,
            patience=5,
            min_lr=1e-7
        )
    elif scheduler_type == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
    elif scheduler_type == "OneCycleLR":
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=learning_rate * 10,
            steps_per_epoch=len(training_generator), 
            epochs=epoch,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )
    
    losses, losses_avg, accuracies, accuracies_top3, val_losses, val_accuracies, val_accuracies_top3 = [], [], [], [], [], [], []

    # Save initial config
    config = {
        "board_size": board_size,
        "n_size": n_size,
        "num_layer": num_layer,
        "learning_rate": learning_rate,
        "epoch": epoch,
        "batch_size": batch_size,
        "dropout": dropout,
        "scheduler_type": scheduler_type,
        "weight_decay": weight_decay,
        "architecture": "Residual"
    }
    save_config(config, run_name)

    wandb.init(
        project="BadukDeepLearning",
        name=run_name,
        config=config
    )
    # messes up compiling
    # wandb.watch(model, log="all")

    for t in tqdm(range(epoch)):
        logger.info(f"Epoch {t+1}\n-------------------------------")

        # Save checkpoint
        checkpoint_path = os.path.join(run_checkpoint_folder, f"epoch_{t+1}.pth")
        if t % 20 == 0:
            torch.save(model.state_dict(), checkpoint_path)

        losses_ep, accuracies_ep, accuracies_top3_ep = train_loop(
            training_generator, model, loss_fn, optimizer, logger, t+1, scheduler
        )
        val_loss_ep, val_acc_ep, val_acc_top3_ep = validation_loop(val_generator, model, loss_fn, logger, t+1)
        
        if scheduler_type == "ReduceLROnPlateau":
            scheduler.step(val_loss_ep)
        elif scheduler_type == "CosineAnnealingWarmRestarts":
            scheduler.step()
            
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"learning_rate": current_lr, "epoch": t+1})

        loss_avg = sum(losses_ep) / len(losses_ep)

        losses += losses_ep
        accuracies += accuracies_ep
        accuracies_top3 += accuracies_top3_ep
        losses_avg += [loss_avg] * len(losses_ep)
        val_losses += [val_loss_ep] * len(losses_ep)
        val_accuracies += [val_acc_ep] * len(accuracies_ep)
        val_accuracies_top3 += [val_acc_top3_ep] * len(accuracies_ep)

        plot_and_save(
            [
                (losses, "Train Loss"),
                (losses_avg, "Train Avg Loss"),
                (val_losses, "Validation Loss"),
            ],
            os.path.join(run_plot_folder, "loss.png"),
        )
        plot_and_save(
            [(accuracies, "Train Accuracy"), (val_accuracies, "Validation Accuracy")],
            os.path.join(run_plot_folder, "accuracies.png"),
        )
        plot_and_save(
            [(accuracies_top3, "Train Top-3 Accuracy"), (val_accuracies_top3, "Validation Top-3 Accuracy")],
            os.path.join(run_plot_folder, "accuracies_top3.png"),
        )

    wandb.finish()
    return min(val_losses), max(val_accuracies)


def main():
    print(f"Using device {DEVICE}")

    training_set = GameDataset(TRAIN_DATA_PATH, DEVICE, prefetch=True)
    val_set = GameDataset(VAL_DATA_PATH, DEVICE, prefetch=True)

    config_space = {
        "n_sizes": [2048, 4096],
        "num_layers": [3],
        "learning_rates": [0.01],
        "epochs": [900],
        "batch_sizes": [1024],
        "dropouts": [0.1],
        "weight_decays": [1e-1],
        "scheduler_types": ["ReduceLROnPlateau"]
    }

    combinations = itertools.product(*config_space.values())

    for n_size, num_layer, learning_rate, epoch, batch_size, dropout, weight_decay, scheduler_type in combinations:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        start = time.time()
        config = {
            "board_size": BOARD_SIZE,
            "n_size": n_size,
            "num_layer": num_layer,
            "learning_rate": learning_rate,
            "epoch": epoch,
            "batch_size": batch_size,
            "dropout": dropout,
            "scheduler_type": scheduler_type,
            "weight_decay": weight_decay,
            "run_name": run_name,
        }
        print(f"Running Config: {config}")

        min_val_loss, max_val_acc = train(training_set, val_set, **config)

        duration = time.time() - start
        print(
            f"Minimum validation loss: {min_val_loss}, maximum validation accuracy: {max_val_acc}"
        )
        print(f"Ran in {duration} seconds")
        print()


if __name__ == "__main__":
    main()
