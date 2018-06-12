""" Script for creating a Generative Adversarial Network (GAN)
    trained on the dog-breed dataset accompanied in the data directory
    Use of PyTorch ( :) :D )
"""

import argparse
import os
import torch as th
import pickle

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


class Generator(th.nn.Module):
    """ Generator of the GAN """

    def __init__(self, noise_size=512):
        """
        constructor of the class
        :param noise_size: dimensionality of the input prior Z
        """

        super(Generator, self).__init__()  # super constructor call

        # define the state of the object
        self.z_size = noise_size

        # define all the required modules for the generator
        from torch.nn import ConvTranspose2d, Conv2d, Upsample, LeakyReLU
        from torch.nn.functional import local_response_norm

        ch = self.z_size

        # Layer 1:
        self.conv_1_1 = ConvTranspose2d(ch, ch, (4, 4))
        self.conv_1_2 = Conv2d(ch, ch, (3, 3), padding=1)

        # Layer 2:
        self.conv_2_1 = Conv2d(ch, ch, (3, 3), padding=1)
        self.conv_2_2 = Conv2d(ch, ch, (3, 3), padding=1)

        # Layer 3:
        self.conv_3_1 = Conv2d(ch, ch, (3, 3), padding=1)
        self.conv_3_2 = Conv2d(ch, ch, (3, 3), padding=1)

        # Layer 4:
        self.conv_4_1 = Conv2d(ch, ch, (3, 3), padding=1)
        self.conv_4_2 = Conv2d(ch, ch, (3, 3), padding=1)

        # Layer 5:
        self.conv_5_1 = Conv2d(ch, ch // 2, (3, 3), padding=1)
        self.conv_5_2 = Conv2d(ch // 2, ch // 2, (3, 3), padding=1)

        # Layer 6:
        self.conv_6_1 = Conv2d(ch // 2, ch // 4, (3, 3), padding=1)
        self.conv_6_2 = Conv2d(ch // 4, ch // 4, (3, 3), padding=1)

        # Upsampler
        self.upsample = Upsample(scale_factor=2)

        # To RGB converter operation:
        self.ToRGB = Conv2d(ch // 4, 3, (1, 1), bias=False)

        # Pixelwise feature vector normalization operation
        self.pixNorm = lambda x: local_response_norm(x, 2*x.shape[1], alpha=2, beta=0.5,
                                                     k=1e-8)

        # Leaky Relu to be applied as activation
        self.lrelu = LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        """
        forward pass of the Generator
        :param x:
        :return: samps => generated Samples
        """
        from torch.nn.functional import tanh

        # Define the forward computations
        y = self.lrelu(self.conv_1_1(x))
        y = self.lrelu(self.pixNorm(self.conv_1_2(y)))

        y = self.upsample(y)
        y = self.lrelu(self.pixNorm(self.conv_2_1(y)))
        y = self.lrelu(self.pixNorm(self.conv_2_2(y)))

        y = self.upsample(y)
        y = self.lrelu(self.pixNorm(self.conv_3_1(y)))
        y = self.lrelu(self.pixNorm(self.conv_3_2(y)))

        y = self.upsample(y)
        y = self.lrelu(self.pixNorm(self.conv_4_1(y)))
        y = self.lrelu(self.pixNorm(self.conv_4_2(y)))

        y = self.upsample(y)
        y = self.lrelu(self.pixNorm(self.conv_5_1(y)))
        y = self.lrelu(self.pixNorm(self.conv_5_2(y)))

        y = self.upsample(y)
        y = self.lrelu(self.pixNorm(self.conv_6_1(y)))
        y = self.lrelu(self.pixNorm(self.conv_6_2(y)))

        # convert the output to RGB form:
        samps = tanh(self.ToRGB(y))

        # return the generated samples:
        return samps


class Discriminator(th.nn.Module):
    """ Discriminator of the GAN """

    def __init__(self):
        """
        constructor for the class
        """
        super(Discriminator, self).__init__()  # super constructor call

        # define all the required modules for the generator
        from torch.nn import Conv2d, LeakyReLU, AvgPool2d

        channels = 3  # for RGB images
        net_ch = 128

        # Layer 1:
        self.conv_1_1 = Conv2d(channels, net_ch, (1, 1))
        self.conv_1_2 = Conv2d(net_ch, net_ch, (3, 3), padding=1)
        self.conv_1_3 = Conv2d(net_ch, 2 * net_ch, (3, 3), padding=1)

        # Layer 2:
        self.conv_2_1 = Conv2d(2 * net_ch, 2 * net_ch, (3, 3), padding=1)
        self.conv_2_2 = Conv2d(2 * net_ch, 4 * net_ch, (3, 3), padding=1)

        # fixing number of channels hereon ...
        fix_channel = 4 * net_ch

        # Layer 3:
        self.conv_3_1 = Conv2d(fix_channel, fix_channel, (3, 3), padding=1)
        self.conv_3_2 = Conv2d(fix_channel, fix_channel, (3, 3), padding=1)

        # Layer 4:
        self.conv_4_1 = Conv2d(fix_channel, fix_channel, (3, 3), padding=1)
        self.conv_4_2 = Conv2d(fix_channel, fix_channel, (3, 3), padding=1)

        # Layer 5:
        self.conv_5_1 = Conv2d(fix_channel, fix_channel, (3, 3), padding=1)
        self.conv_5_2 = Conv2d(fix_channel, fix_channel, (3, 3), padding=1)

        # Layer 6:
        self.conv_6_1 = Conv2d(fix_channel, fix_channel, (3, 3), padding=1)
        self.conv_6_2 = Conv2d(fix_channel, fix_channel, (3, 3), padding=1)
        self.conv_6_3 = Conv2d(fix_channel, fix_channel, (4, 4))
        self.conv_6_4 = Conv2d(fix_channel, 1, (1, 1), bias=False)

        # Downsampler (Average pooling)
        self.downsample = AvgPool2d(2)

        # Leaky Relu to be applied as activation
        self.lrelu = LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        """
        forward pass of the Discriminator
        :param x: input images
        :return: raw_preds => raw prediction score for WGAN
        """

        # Define the Forward computations:
        y = self.lrelu(self.conv_1_1(x))
        y = self.lrelu(self.conv_1_2(y))
        y = self.lrelu(self.conv_1_3(y))
        y = self.downsample(y)

        y = self.lrelu(self.conv_2_1(y))
        y = self.lrelu(self.conv_2_2(y))
        y = self.downsample(y)

        y = self.lrelu(self.conv_3_1(y))
        y = self.lrelu(self.conv_3_2(y))
        y = self.downsample(y)

        y = self.lrelu(self.conv_4_1(y))
        y = self.lrelu(self.conv_4_2(y))
        y = self.downsample(y)

        y = self.lrelu(self.conv_5_1(y))
        y = self.lrelu(self.conv_5_2(y))
        y = self.downsample(y)

        y = self.lrelu(self.conv_6_1(y))
        y = self.lrelu(self.conv_6_2(y))
        y = self.lrelu(self.conv_6_3(y))
        y = self.conv_6_4(y)  # last layer has linear activation

        # generate the raw predictions
        raw_preds = y.view(-1)

        return raw_preds


class GAN:
    """ Wrapper around the Generator and the Discriminator """

    class WeightClipper:
        """ Simple class for implementing weight clamping """
        def __init__(self, clamp_value):
            """ constructor """
            self.clamp_val = clamp_value

        def __call__(self, module):
            """ Recursive application of weight clamping """
            # filter the variables to get the ones you want
            if hasattr(module, 'weight'):
                w = module.weight.data
                th.clamp(w, min=-self.clamp_val, max=self.clamp_val, out=w)

    def __init__(self, generator, discriminator, learning_rate=0.001, beta_1=0,
                 beta_2=0.99, eps=1e-8, clamp_value=0.01, n_critic=5):
        """
        constructor for the class
        :param generator: Generator object
        :param discriminator: Discriminator object
        :param learning_rate: learning rate for Adam
        :param beta_1: beta_1 for Adam
        :param beta_2: beta_2 for Adam
        :param eps: epsilon for Adam
        :param n_critic: number of times to update discriminator
        :param clamp_value: Clamp value for Wasserstein update
        """

        from torch.optim import Adam

        # define the state of the object
        self.gen = generator
        self.dis = discriminator
        self.n_critic = n_critic

        # define the optimizers for the discriminator and generator
        self.gen_optim = Adam(self.gen.parameters(), lr=learning_rate,
                              betas=(beta_1, beta_2), eps=eps)

        self.dis_optim = Adam(self.dis.parameters(), lr=learning_rate,
                              betas=(beta_1, beta_2), eps=eps)

        self.clamper = self.WeightClipper(clamp_value=clamp_value)

    def generate_samples(self, num):
        """
        generate samples using the generator
        :param num: number of samples required
        :return: samps => generated samples
        """

        # generate the required random noise:
        noise = th.randn(num, self.gen.z_size, 1, 1).to(device)

        # generate the samples by performing the forward pass on generator
        samps = self.gen(noise)

        return samps

    def optimize_discriminator(self, batch):
        """
        performs one step of weight update on discriminator using the batch of data
        :param batch: real samples batch
        :return: current loss (Wasserstein loss)
        """
        # rename the input for simplicity
        real_samples = batch

        loss_val = 0
        for _ in range(self.n_critic):
            # generate a batch of samples
            fake_samples = self.generate_samples(batch.shape[0])

            # define the (Wasserstein) loss
            loss = th.mean(self.dis(real_samples)) - th.mean(self.dis(fake_samples))

            # optimize discriminator
            self.dis_optim.zero_grad()
            loss.backward()
            self.dis_optim.step()

            # clamp the updated weight values
            self.dis.apply(self.clamper)

            loss_val += loss.item()

        return loss_val / self.n_critic

    def optimize_generator(self, batch_size):
        """
        performs one step of weight update on generator for the given batch_size
        :param batch_size: batch_size
        :return: current loss (Wasserstein estimate)
        """

        # generate fake samples:
        fake_samples = self.generate_samples(batch_size)

        loss = -th.mean(self.dis(fake_samples))

        # optimize the generator
        self.gen_optim.zero_grad()
        loss.backward()
        self.gen_optim.step()

        # return the loss value
        return loss.item()


def create_grid(gan, img_file, width=2):
    """
    utility funtion to create a grid of GAN samples
    :param gan: GAN object
    :param img_file: name of file to write
    :param width: width for the grid
    :return: None (saves a file)
    """
    from torchvision.utils import save_image

    # generate width^2 samples
    samples = gan.generate_samples(width * width)

    # save the images:
    save_image(samples, img_file, nrow=width)


def train_GAN(gan, data, num_epochs=21, feedback_factor=10,
              save_dir="./GAN_Models/", sample_dir="GAN_Out/",
              log_file="./GAN_Models/loss.log", checkpoint_factor=3):
    """
    train the GAN (network) using the given data
    :param gan: GAN object
    :param data: data_loader stream for training
    :param num_epochs: Number of epochs to train for
    :param feedback_factor: number of prints per epoch
    :param save_dir: directory for saving the GAN models
    :param sample_dir: directory for storing the generated samples
    :param log_file: log file for loss collection
    :param checkpoint_factor: save after every n epochs
    :return: None
    """
    total_batches = len(iter(data))

    print("Starting the GAN training ... ")
    for epoch in range(num_epochs):
        print("Epoch: %d" % (epoch + 1))

        for (i, batch) in enumerate(data, 1):
            # optimize the discriminator using the data batch:
            dis_loss = gan.optimize_discriminator(batch.to(device))

            # optimize the generator using the data batch:
            gen_loss = gan.optimize_generator(batch.shape[0])

            # provide a loss feedback
            if i % int(total_batches / feedback_factor) == 0 or i == 1:
                print("batch: %d  d_loss: %f  g_loss: %f" % (i, dis_loss, gen_loss))

                # also write the losses to the log file:
                with open(log_file, "a") as log:
                    log.write(str(dis_loss)+"\t"+str(gen_loss)+"\n")

                # create a grid of samples and save it
                img_file = os.path.join(sample_dir, str(epoch + 1) + "_" +
                                        str(i) + ".png")
                create_grid(gan, img_file, 4)

        if (epoch + 1) % checkpoint_factor == 0 or epoch == 0:
            # save the GAN
            gen_save_file = os.path.join(save_dir, "GAN_GEN"+str(epoch+1)+".pth")
            dis_save_file = os.path.join(save_dir, "GAN_DIS" + str(epoch + 1) + ".pth")
            th.save(gan.gen.state_dict(), gen_save_file, pickle)
            th.save(gan.dis.state_dict(), dis_save_file, pickle)

    print("Training completed ...")


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
    parser.add_argument("--batch_size", action="store", type=int, default=8,
                        help="batch size for SGD. default = 32")
    parser.add_argument("--parallel_readers", action="store", type=int, default=3,
                        help="number of parallel processes to read data. default = 3")
    parser.add_argument("--learning_rate", action="store", type=float, default=0.00005,
                        help="learning rate for Adam optimization")
    parser.add_argument("--beta_1", action="store", type=float, default=0,
                        help="beta_1 for Adam optimization")
    parser.add_argument("--beta_2", action="store", type=float, default=0.99,
                        help="beta_2 for Adam optimization")
    parser.add_argument("--epsilon", action="store", type=float, default=1e-8,
                        help="epsilon for Adam optimization")
    parser.add_argument("--clamp_value", action="store", type=float, default=0.01,
                        help="clamp value for Wasserstein critic")
    parser.add_argument("--n_critic", action="store", type=int, default=5,
                        help="number of times to train for Wasserstein critic per step")
    parser.add_argument("--num_epochs", action="store", type=int, default=21,
                        help="number of epochs to train the gan for")
    parser.add_argument("--feedback_factor", action="store", type=int, default=1200,
                        help="number of times to log loss and generate samples per epoch")
    parser.add_argument("--save_dir", action="store", type=str, default="./GAN_Models/",
                        help="directory to save the models")
    parser.add_argument("--sample_dir", action="store", type=str, default="GAN_Out/",
                        help="directory to save the generated samples")
    parser.add_argument("--log_file", action="store", type=str, default="./GAN_Models/loss.log",
                        help="log file for saving losses")
    parser.add_argument("--checkpoint_factor", action="store", type=int, default=1,
                        help="Save after every n epochs")

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

    # create a GAN
    gan = GAN(
        generator=Generator(512).to(device),
        discriminator=Discriminator().to(device),
        learning_rate=args.learning_rate,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        eps=args.epsilon,
        clamp_value=args.clamp_value,
        n_critic=args.n_critic
    )

    # train the gan:
    train_GAN(
        gan=gan,
        data=data,
        num_epochs=args.num_epochs,
        feedback_factor=args.feedback_factor,
        save_dir=args.save_dir,
        sample_dir=args.sample_dir,
        log_file=args.log_file,
        checkpoint_factor=args.checkpoint_factor,
    )


if __name__ == '__main__':
    main(parse_arguments())
