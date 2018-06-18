""" Module containing functionalities for processing the data
"""

import re


def read_and_basic_process(file_name):
    """
    reads the file and stores it as a list of lists
    :param file_name: input file name (path)
    :return: dat => list[lists[string]]
    """
    print("Reading file: ", file_name, "...")
    with open(file_name, "r") as fil:
        dat_blob = fil.read()

    # insert space before all the special characters
    print("performing regex substitution ...")
    db = re.sub(r"([^a-zA-Z])", r" \1 ", dat_blob)

    # split the blob into lines:
    print("performing word level tokenization ...")
    lines = db.split("\n")

    # purge any two consecutive spaces in the blob
    lines = list(map(lambda x: re.sub('\s{2,}', ' ', x), lines))

    # split every line into list of words:
    dat = list(map(lambda x: x.split(), lines))

    return dat


def frequency_count(text_data):
    """
    count the frequency of each word in data
    :param text_data: list[lists[string]]
    :return: freq_cnt => {word -> freq}
    """
    # generate the vocabulary
    total_word_list = []
    for line in text_data:
        total_word_list.extend(line)

    vocabulary = set(total_word_list)

    freq_count = dict(map(lambda x: (x, 0), vocabulary))

    # count the frequencies of the words
    for line in text_data:
        for word in line:
            freq_count[word] += 1

    # return the frequency counts
    return freq_count


def tokenize(text_data, freq_counts, vocab_size=None):
    """
    tokenize the text_data using the freq_counts
    :param text_data: list[lists[string]]
    :param freq_counts: {word -> freq}
    :param vocab_size: size of the truncated vocabulary
    :return: (rev_vocab, trunc_vocab, transformed_data
                => reverse vocabulary, truncated vocabulary, numeric sequences)
    """
    # truncate the vocabulary:
    vocab_size = len(freq_counts) if vocab_size is None else vocab_size

    trunc_vocab = dict(sorted(freq_counts.items(), key=lambda x: x[1], reverse=True)[:vocab_size])
    trunc_vocab = dict(enumerate(trunc_vocab.keys(), start=2))

    # add <unk> and <pad> tokens:
    trunc_vocab[1] = "<unk>"
    trunc_vocab[0] = "<pad>"

    # compute reverse trunc_vocab
    rev_trunc_vocab = dict(list(map(lambda x: (x[1], x[0]), trunc_vocab.items())))

    # transform the sentences:
    transformed_data = []  # initialize to empty list
    for sentence in text_data:
        transformed_sentence = []
        for word in sentence:
            numeric_code = rev_trunc_vocab[word] \
                if word in rev_trunc_vocab else rev_trunc_vocab["<unk>"]
            transformed_sentence.append(numeric_code)

        transformed_data.append(transformed_sentence)

    # return the truncated vocabulary and transformed sentences:
    return trunc_vocab, rev_trunc_vocab, transformed_data
