import itertools
import json
import logging
import os
import time
from datetime import datetime

import wandb
import torch
from torch import nn
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


def train_loop(dataloader, model, loss_fn, optimizer, logger, epoch):
    model.train()
    size = len(dataloader.dataset)
    losses, accuracies = [], []

    for batch, (X, y, nm_color) in enumerate(dataloader):
        pred = model(X, nm_color)
        y = torch.reshape(y, (-1, 81))
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 1000 == 0:
            batch_size = len(X)
            # loss, current = loss.item(), batch * batch_size+ len(X)
            current = batch * batch_size + len(X)
            # correct = (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            correct = (pred.argmax(1) == y.argmax(1)).sum().item()
            accuracy = 100 * correct / batch_size

            losses.append(loss.item())
            accuracies.append(accuracy)

            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            logger.info(f"Accuracy: {100*correct/batch_size}")
            logger.info(f"max pred: {torch.max(pred[0])}")
            
            wandb.log({
                "train_loss": loss.item(),
                "train_accuracy": accuracy,
                "epoch": epoch,
                "batch": batch,
                "step": (epoch - 1) * len(dataloader) + batch
            })

    return losses, accuracies


def validation_loop(dataloader, model, loss_fn, logger, epoch):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, correct = 0, 0

    with torch.no_grad():
        for X, y, nm_color in dataloader:
            pred = model(X, nm_color)
            y = torch.reshape(y, (-1, 81))
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    val_loss /= num_batches
    accuracy = 100 * correct / size

    logger.info(
        f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n"
    )
    wandb.log({
        "val_loss": val_loss,
        "val_accuracy": accuracy,
        "epoch": epoch
    })
    return val_loss, accuracy


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
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    losses, losses_avg, accuracies, val_losses, val_accuracies = [], [], [], [], []

    # Save initial config
    config = {
        "board_size": board_size,
        "n_size": n_size,
        "num_layer": num_layer,
        "learning_rate": learning_rate,
        "epoch": epoch,
        "batch_size": batch_size,
        "dropout": dropout,
    }
    save_config(config, run_name)

    wandb.init(
        project="BadukDeepLearning",
        name=run_name,
        config=config
    )
    wandb.watch(model, log="all")

    for t in tqdm(range(epoch)):
        logger.info(f"Epoch {t+1}\n-------------------------------")

        # Save checkpoint
        checkpoint_path = os.path.join(run_checkpoint_folder, f"epoch_{t+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        losses_ep, accuracies_ep = train_loop(
            training_generator, model, loss_fn, optimizer, logger, t+1
        )
        val_loss_ep, val_acc_ep = validation_loop(val_generator, model, loss_fn, logger, t+1)

        loss_avg = sum(losses_ep) / len(losses_ep)

        losses += losses_ep
        accuracies += accuracies_ep
        losses_avg += [loss_avg] * len(losses_ep)
        val_losses += [val_loss_ep] * len(losses_ep)
        val_accuracies += [val_acc_ep] * len(accuracies_ep)

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

    wandb.finish()
    return min(val_losses), max(val_accuracies)


def main():
    print(f"Using device {DEVICE}")

    training_set = GameDataset(TRAIN_DATA_PATH, DEVICE, prefetch=True)
    val_set = GameDataset(VAL_DATA_PATH, DEVICE, prefetch=True)

    config_space = {
        "n_sizes": [256, 512],
        "num_layers": [8],
        "learning_rates": [0.001],
        "epochs": [100],
        "batch_sizes": [1024, 512, 256],
        "dropouts": [0.0, 0.1, 0.3],
    }

    combinations = itertools.product(*config_space.values())

    for n_size, num_layer, learning_rate, epoch, batch_size, dropout in combinations:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        start = time.time()
        config = {
            "board_size": BOARD_SIZE,
            "n_size": n_size,
            "num_layer": num_layer,
            "learning_rate": learning_rate,
            "epoch": epoch,
            "epoch": epoch,
            "batch_size": batch_size,
            "dropout": dropout,
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
