""" Module for creating the DataLoader of the Network """

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch as th
import pickle


class LeakGANDataset(Dataset):
    """ PyTorch Dataset wrapper for the leak-gan dataset """

    def __setup_data(self):
        """
        private helper for reading and filtering the data
        :return: fil_dat, vocab, rev_vocab => filtered data, vocabulary, rev
        """
        with open(self.data_source, "rb") as ds:
            data_obj = pickle.load(ds)

        data = data_obj["text_sequences"]

        # remove zero length examples from the data
        i = 0
        while i < len(data):
            if len(data[i]) == 0:
                data.pop(i)
                continue
            i += 1

        # return the filtered data, vocabulary and the rev_vocabulary
        return data, data_obj["vocab"], data_obj["rev_vocab"]

    def __init__(self, pickle_file, sentence_length):
        """
        constructor for the class. This dataset reads the data from the pickle
        file and pads / truncates the sequences as per sentence_length
        :param pickle_file: file containing the data
        :param sentence_length: sentence lengths to be adjusted
        """

        # setup the object state
        self.len = sentence_length
        self.data_source = pickle_file

        # get the data and vocabularies
        self.data, self.vocab, self.rev_vocab = self.__setup_data()

    def __len__(self):
        """
        obtain the length of the data:
        :return: len => length of the data
        """
        return len(self.data)

    def __getitem__(self, item):
        """
        item obtaining for indexing purpose
        :param item: index
        :return: one example from the sequence
        """

        # obtain the required item from the data list
        sample = self.data[item]

        # pad / truncate the sample
        if len(sample) < self.len:
            while len(sample) != self.len:
                sample.append(self.rev_vocab["<pad>"])

        elif len(sample) > self.len:
            sample = sample[: self.len]

        # return the processed sample
        return th.tensor(sample)

    def get_text_item(self, item):
        """
        converts the numeric sequence into text sentence
        :param item: index
        :return: text sentence
        """
        sample = self.__getitem__(item)

        # return the list of converted words
        return [self.vocab[num_id.item()] for num_id in sample]


def __collate_fn(samples):
    """
    module level private function to combine the samples into a 2D tensor
    :param samples: list of tensors
    :return: batch_tensor => 2D tensor of shape [batch x seq_len]
    """
    return th.stack(samples)


def create_data_loader(dataset, batch_size, shuffle=True, num_workers=3):
    """
    obtain the PyTorch data_loader object
    :param dataset: dataset object
    :param batch_size: batch_size for training batches
    :param shuffle: whether to shuffle before every epoch
    :param num_workers: number of parallel worker processes
    :return: dLoader => PyTorch DataLoader object
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=__collate_fn
    )
