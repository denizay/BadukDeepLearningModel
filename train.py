import time
import logging
import itertools
import torch
from torch import nn
from matplotlib import pyplot as plt
from dataset import GameDataset
from model import NeuralNetwork


BOARD_SIZE = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu")
LOG_FILE_PATH = 'logs/logs_hps.log'
PLOT_FOLDER = 'plots'
CHECKPOINT_FOLDER = 'checkpoints'
#TRAIN_DATA_PATH = "data_pt/train_data_big.pt"
TRAIN_DATA_PATH =  "data_pt/validation_data_big.pt"
VAL_DATA_PATH = "data_pt/validation_data_big.pt"



def setup_logger(log_file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


logger = setup_logger(LOG_FILE_PATH)


def train_loop(dataloader, model, loss_fn, optimizer):
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

    return losses, accuracies


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y, nm_color in dataloader:
            pred = model(X, nm_color)
            y = torch.reshape(y, (-1, 81))
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)
                        ).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = 100 * correct / size

    logger.info(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, accuracy


def plot_and_save(logs, file_name):  # claude onerisini dene
    fig = plt.figure()
    for log_vals, label in logs:
        plt.plot(log_vals, label=label)
    plt.legend()
    plt.savefig(file_name)
    fig.clf()
    plt.close(fig)


def train(
        train_set,
        test_set,
        board_size,
        n_size,
        num_layer,
        learning_rate,
        epoch,
        batch_size):
    training_generator = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True)
    test_generator = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True)

    model = NeuralNetwork(board_size, n_size, num_layer).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4)

    losses, losses_avg, accuracies, t_losses, t_accuracies = [], [], [], [], []
    fn = f"{n_size}ns_{num_layer}ls_{learning_rate}lr_{epoch}ep_{batch_size}bs"

    for t in range(epoch):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        torch.save(model.state_dict(), f'{CHECKPOINT_FOLDER}/mw_{fn}.pth')

        losses_ep, accuracies_ep = train_loop(
            training_generator, model, loss_fn, optimizer)
        test_loss_ep, test_acc_ep = test_loop(test_generator, model, loss_fn)

        loss_avg = sum(losses_ep) / len(losses_ep)

        losses += losses_ep
        accuracies += accuracies_ep
        losses_avg += [loss_avg] * len(losses_ep)
        t_losses += [test_loss_ep] * len(losses_ep)
        t_accuracies += [test_acc_ep] * len(accuracies_ep)

        plot_and_save([(losses, "Train Loss"), (losses_avg, "Train Avg Loss"),
                      (t_losses, "Test Loss")], f"{PLOT_FOLDER}/loss_{fn}.png")
        plot_and_save([(accuracies, "Train Accuracy"), (t_accuracies,
                      "Test Accuracy")], f"{PLOT_FOLDER}/accuracies_{fn}.png")

    return min(t_losses), max(t_accuracies)


def main():
    print(f"Using device {DEVICE}")

    training_set = GameDataset(TRAIN_DATA_PATH, DEVICE, prefetch=True)
    test_set = GameDataset(VAL_DATA_PATH, DEVICE, prefetch=True)

    config_space = {
        'n_sizes': [256, 512],
        'num_layers': [8],
        'learning_rates': [0.001],
        'epochs': [100],
        'batch_sizes': [1024, 512, 256]
    }

    combinations = itertools.product(*config_space.values())

    for n_size, num_layer, learning_rate, epoch, batch_size in combinations:
        start = time.time()
        config = {
            'board_size': BOARD_SIZE,
            'n_size': n_size,
            'num_layer': num_layer,
            'learning_rate': learning_rate,
            'epoch': epoch,
            'batch_size': batch_size
        }
        print(f"Running Config: {config}")
        min_t_loss, max_t_acc = train(training_set, test_set, **config)
        duration = time.time() - start
        print(
            f"Minimum test loss: {min_t_loss}, maximum test accuracy: {max_t_acc}")
        print(f"Ran in {duration} seconds")
        print()


if __name__ == "__main__":
    main()
