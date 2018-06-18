""" Script for processing the text data files """


import common.data_processor as dp
import argparse
import pickle


def save_pickle(obj, file_name):
    """
    save the given data obj as a pickle file
    :param obj: python data object
    :param file_name: path of the output file
    :return: None (writes file to disk)
    """
    with open(file_name, 'wb') as dumper:
        pickle.dump(obj, dumper, pickle.HIGHEST_PROTOCOL)


def parse_arguments():
    """
    command line arguments parser for the script
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_files", type=str, action="store", nargs="*",
                        default=[], help="space separated names of txt files")
    parser.add_argument("--vocab_size", type=int, action="store",
                        default=25000, help="vocabulary size for the data")
    parser.add_argument("--out_file", type=str, action="store", default="common/data.pkl",
                        help="path of the output file. Default = common/data.pkl")

    args = parser.parse_args()

    return args


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """

    files = args.data_files

    # collect all the data from all the text_files:
    data = []
    for file in files:
        data.extend(dp.read_and_basic_process(file))

    # calculate frequency counts for all the words from the data
    print("calculating the frequencies of the vocabulary words ...")
    freqs = dp.frequency_count(data)

    # transform data and obtain the vocabularies
    print("transforming textual data to numeric sequences ...")
    vocab, rev_vocab, data = dp.tokenize(data, freqs, args.vocab_size)

    # print some statistics and sample data
    print("Vocab_size: ", len(vocab))
    print("sample_vocabulary: ", list(vocab.items())[:10])
    print("transformed sequences: ", data[:3])

    # save the processed data:
    data_obj = {
        "text_sequences": data,
        "vocab": vocab,
        "rev_vocab": rev_vocab
    }

    print("saving data to: ", args.out_file)
    save_pickle(data_obj, args.out_file)


if __name__ == '__main__':
    main(parse_arguments())

