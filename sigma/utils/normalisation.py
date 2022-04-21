import numpy as np
import itertools

def neighbour_averaging(dataset: np) -> np:
    h, w = dataset.shape[0], dataset.shape[1]
    new_dataset = np.zeros(
        shape=dataset.shape
    )  # create an empty np.array for new dataset

    for row in range(h):  # for each row
        for col in range(w):  # for each column
            row_idxs = [row - 1, row, row + 1]
            col_idxs = [
                col - 1,
                col,
                col + 1,
            ]  # get indices from the neighboring (num=3*3)

            for row_idx in row_idxs:
                if row_idx < 0 or row_idx >= h:
                    row_idxs.remove(
                        row_idx
                    )  # remove the pixels which is out ofthe boundaries

            for col_idx in col_idxs:
                if col_idx < 0 or col_idx >= w:
                    col_idxs.remove(
                        col_idx
                    )  # remove the pixels which is out ofthe boundaries

            # get positions using indices after the removal of pixels out of the boundaries
            positions = [pos for pos in itertools.product(row_idxs, col_idxs)]
            background_signal = []

            for k in positions:
                background_signal.append(dataset[k])
            background_signal = np.stack(background_signal, axis=0)
            background_signal_avg = (
                np.sum(background_signal, axis=0) / background_signal.shape[0]
            )

            new_dataset[row, col, :] = background_signal_avg

    return new_dataset


def zscore(dataset: np) -> np:
    new_dataset = dataset.copy()
    for i in range(new_dataset.shape[2]):
        mean = new_dataset[:, :, i].mean()
        std = new_dataset[:, :, i].std()
        new_dataset[:, :, i] = (new_dataset[:, :, i] - mean) / std
    return new_dataset


def softmax(dataset: np) -> np:
    exp_dataset = np.exp(dataset)
    sum_exp = np.sum(exp_dataset, axis=2)
    sum_exp = np.expand_dims(sum_exp, axis=2)
    sum_exp = np.tile(sum_exp, (1, 1, dataset.shape[2]))
    new_dataset = exp_dataset / sum_exp
    return new_dataset
