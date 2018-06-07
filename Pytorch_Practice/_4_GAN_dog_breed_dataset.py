""" Script for creating a Generative Adversarial Network (GAN)
    trained on the dog-breed dataset accompanied in the data directory
    Use of PyTorch ( :) :D )
"""

import argparse
import os
import torch as th

from torch.utils.data import Dataset

device = th.device("cuda" if th.cuda.is_available() else "cpu")

data_path = "data/dog-breed/train"


class DogBreedDataset(Dataset):
    """ pyTorch Dataset wrapper for the Dog breed dataset """

    def __setup_files(self):
        """
        private helper for setting up the files_list
        :return: files => list of paths of files
        """
        file_names = os.listdir(self.data_dir)
        files = []  # initialize to empty list

        for file_name in file_names:
            possible_file = os.path.join(self.data_dir, file_name)
            if os.path.isfile(possible_file):
                files.append(possible_file)

        # return the files list
        return files

    def __init__(self, data_dir, transform=None):
        """
        constructor for the class
        :param data_dir: path to the directory containing the data
        :param transform: transforms to be applied to the images
        """
        # define the state of the object
        self.data_dir = data_dir
        self.transform = transform

        # setup the files for reading
        self.files = self.__setup_files()

    def __len__(self):
        """
        compute the length of the dataset
        :return: len => length of dataset
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        obtain the image (read and transform)
        :param idx: index of the file required
        :return: img => image array
        """
        from PIL import Image

        # read the image:
        img = Image.open(self.files[idx])

        # apply the transforms on the image
        if self.transform is not None:
            img = self.transform(img)

        # return the image:
        return img


def setup_data(new_size, batch_size, p_readers):
    """
    setup the data for the training
    :param new_size: tuple (height, width) resized image size
    :param batch_size: size of each batch of Data
    :param p_readers: number of parallel reader processes.
    :return: d_loader => dataLoader for the Dog Breed dataset
    """

    from torchvision.transforms import Resize, ToTensor, Normalize, Compose
    from torch.utils.data import DataLoader

    # create the transforms required to the images:
    # Namely :-> resize, Convert to Tensors and Normalize
    transforms = Compose([
        Resize(new_size),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # create the dataset for the DataLoader
    ds = DogBreedDataset(data_path, transform=transforms)

    # create the dataLoader from the dataset
    d_loader = DataLoader(ds, batch_size, shuffle=True, num_workers=p_readers)

    # return the so created DataLoader
    return d_loader


def parse_arguments():
    """
    Command line argument parser
    :return: args => parsed command line arguments
    """

    parser = argparse.ArgumentParser()

    # define the arguments to parse
    parser.add_argument("--img_height", action="store", type=int, default=128,
                        help="height of the image samples generated. default = 128")
    parser.add_argument("--img_width", action="store", type=int, default=128,
                        help="width of the image samples generated. default = 128")
    parser.add_argument("--batch_size", action="store", type=int, default=32,
                        help="batch size for SGD. default = 32")
    parser.add_argument("--parallel_readers", action="store", type=int, default=3,
                        help="number of parallel processes to read data. default = 3")

    # parse the detected arguments
    args = parser.parse_args()

    # return the parsed args
    return args


def main(args):
    """
    Main function of the script
    :param args: parsed command line arguments
    :return: None
    """

    data = setup_data(
        new_size=(args.img_height, args.img_width),
        batch_size=args.batch_size,
        p_readers=args.parallel_readers
    )


if __name__ == '__main__':
    main(parse_arguments())
