
import numpy as np


class Batcher:
    """
    Batcher class. Given a list of np.arrays of same 0-dimension, returns a
    a list of batches for these elements
    """

    def __init__(self, data, batch_size, shuffle_on_reset=False):
        """
        :param data: list containing np.arrays (type: list[np.array])
        :param batch_size: size of each batch (type: int)
        :param shuffle_on_reset: flag to shuffle data (type: bool)
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle_on_reset = shuffle_on_reset

        if type(data) == list:
            self.data_size = data[0].shape[0]
        else:
            self.data_size = data.shape[0]
        self.n_batches = int(np.ceil(self.data_size / self.batch_size))
        self.idx = np.arange(0, self.data_size, dtype=int)
        if shuffle_on_reset:
            np.random.shuffle(self.idx)
        self.current = 0

    def shuffle(self):
        """
        Re-shufle the data
        """
        np.random.shuffle(self.idx)

    def reset(self):
        """
        Reset iteration counter
        """
        if self.shuffle_on_reset:
            self.shuffle()
        self.current = 0

    def next(self):
        """
        Get next batch
        :return: list of np.arrays
        """
        i_select = self.idx[
            (self.current * self.batch_size): ((self.current + 1) * self.batch_size)
        ]
        batch = []
        for elem in self.data:
            batch.append(elem[i_select])

        if self.current < (self.n_batches - 1):
            self.current = self.current + 1
        else:
            self.reset()

        return batch
