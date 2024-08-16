import os
from glob import glob
from tqdm import tqdm
import numpy as np
import torch
from sgfparser import sgf_to_matrix


SGF_FOLDER_PATH = "sgfs"
BOARD_POS_COUNT = 16


def get_positions(sgf_paths):
    data = []
    fail_count = 0
    for sgf_path in tqdm(sgf_paths):
        try:
            with open(sgf_path, 'r', encoding='utf-8') as f:
                sgf_data = f.read()
            for i in range(BOARD_POS_COUNT):
                board_matrix, label_board, label_color = sgf_to_matrix(
                    sgf_data)
                board_matrix = np.rot90(board_matrix, k=i).copy()
                label_board = np.rot90(label_board, k=i).copy()
                data.append(
                    (torch.tensor(
                        board_matrix, dtype=torch.float), torch.tensor(
                        label_board, dtype=torch.float), torch.tensor(
                        label_color, dtype=torch.float)))
        except Exception as e:
            fail_count += 1
    return data, fail_count


def main():
    sgf_paths = glob(os.path.join(SGF_FOLDER_PATH, "*.sgf"))
    sgf_count = len(sgf_paths)

    train_paths, val_paths, test_paths = np.split(np.random.permutation(
        sgf_paths), [int(.75 * sgf_count), int(.875 * sgf_count)])

    train_data, train_fc = get_positions(train_paths)
    val_data, val_fc = get_positions(val_paths)
    test_data, test_fc = get_positions(test_paths)

    with open('train_data_big.pkl', 'wb') as f:
        torch.save(train_data, f)

    with open('validation_data_big.pkl', 'wb') as f:
        torch.save(val_data, f)

    with open('test_data_big.pkl', 'wb') as f:
        torch.save(test_data, f)

    print(f"Train data length: {len(train_data)}")
    print(f"Val data length: {len(val_data)}")
    print(f"Test data length: {len(test_data)}")
    fail_count = train_fc + test_fc + val_fc
    print(f"Fail count: {fail_count}")


if __name__ == '__main__':
    main()
