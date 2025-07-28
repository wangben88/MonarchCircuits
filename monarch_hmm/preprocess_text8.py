import torch
import argparse
import urllib.request
import zipfile

def encode_and_chunk_text8(data, seq_len, char2idx):
    print("Dataset shape", len(data))
    data = torch.tensor([char2idx[char] for char in data])
    data = data[:seq_len*(len(data)//seq_len)]
    data = data.reshape(-1, seq_len)
    print("Dataset shape after chunk", data.shape, data.dtype)
    return data


def encode_text8(data, char2idx):
    data = torch.tensor([char2idx[char] for char in data])
    return data


def main():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--output_path', default='', type=str)
    args = arg_parser.parse_args()

    # download
    local_filename = f'{args.output_path}/text8.zip'
    url = 'http://mattmahoney.net/dc/text8.zip'
    urllib.request.urlretrieve(url, local_filename)
    print('Download and save to {}'.format(local_filename))

    # raw
    rawdata = zipfile.ZipFile(local_filename).read('text8').decode('utf-8')

    # vocab
    vocab = sorted(list(set(rawdata)))
    char2idx, idx2char = {}, []
    for i, char in enumerate(vocab):
        char2idx[char] = i
        idx2char.append(char)

    # subset
    raw_train = rawdata[:90000000]
    raw_dev = rawdata[90000000:95000000]
    raw_test = rawdata[95000000:]

    # encode characters and chunk
    seq_len = 256
    train_data = encode_and_chunk_text8(raw_train, seq_len, char2idx)
    dev_data = encode_and_chunk_text8(raw_dev, seq_len, char2idx)
    test_data = encode_and_chunk_text8(raw_test, seq_len, char2idx)
    test_data_one_line = encode_text8(raw_test, char2idx)

    dataset = {
        "train": train_data,
        "dev": dev_data,
        "test": test_data,
        "test_one_line": test_data_one_line,
    }

    torch.save(dataset, f'{args.output_path}/text8_chunk{seq_len}')


if __name__ == '__main__':
    main()