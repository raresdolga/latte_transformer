"""
Assumes data is in large files and not separated per sample
"""

import os
import requests
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from functools import partial
from datasets import load_dataset, DatasetDict
from transformers import BatchEncoding

from preproc.toks import CharFileTokenizer
from preproc.base import DataProcessing

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
LOG = logging.getLogger(__name__)


class FullFileDataSet(Dataset):
    def __init__(self, data, seq_length, pad_id) -> None:
        """
        Efficiently read big files with continious data and sample a sequence lengtth
        Args:
            data: np.memmap. Memory maped data
            seq_length: seq. length to sample
        """
        super().__init__()
        self._data = data
        self.seq_len = seq_length
        self._max_sample = len(self._data) - self.seq_len
        self._len = len(self._data) // seq_length
        self.pad_id = pad_id

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        # no need for the index since we sample
        # Could also devide the data in idexes of seq_len and return those.
        return self.sample_stream()

    def sample_stream(self):
        i = np.random.randint(self._max_sample, size=1).item()
        input_ids = self._data[i : i + self.seq_len]
        labels = np.copy(input_ids)
        labels = np.where(labels == self.pad_id, -100, labels)
        return BatchEncoding({"input_ids": input_ids, "labels": labels})


class ShakespeareDP(DataProcessing):
    """
    Works with small data which can be loaded whole in the memory.
    """

    def __init__(self, cache_dir) -> None:
        self.cache_dir = cache_dir

    def tokenize(self, tokenizer, data, max_seq_len):
        # tokenize the entire file as one string
        tok_train = tokenizer(data["train"])["input_ids"]
        tok_val = tokenizer(data["validation"])["input_ids"]

        # export to bin files
        tok_train = np.array(tok_train, dtype=np.int16)
        tok_val = np.array(tok_val, dtype=np.int16)
        train_file = os.path.join(self.cache_dir, "train.bin")
        tok_train.tofile(train_file)
        val_file = os.path.join(self.cache_dir, "val.bin")
        tok_val.tofile(val_file)
        return {
            "train": FullFileDataSet(
                np.memmap(train_file, dtype=np.int16, mode="r"),
                max_seq_len,
                tokenizer.pad_token_id,
            ),
            "validation": FullFileDataSet(
                np.memmap(val_file, dtype=np.int16, mode="r"),
                max_seq_len,
                tokenizer.pad_token_id,
            ),
        }

    def get_raw_data(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        input_file_path = os.path.join(self.cache_dir, "input.txt")
        if not os.path.exists(input_file_path):
            data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            with open(input_file_path, "w+", encoding="utf-8") as f:
                f.write(requests.get(data_url).text)

        with open(input_file_path, "r", encoding="utf-8") as f:
            data = f.read()

        LOG.info("length of dataset in characters: %s", len(data))
        n = len(data)
        # split the dataset
        train_data = data[: int(n * 0.9)]
        val_data = data[int(n * 0.9) :]
        data = {"train": train_data, "validation": val_data}
        return data


class Enwiki8CharDP(DataProcessing):
    def __init__(self, tokenizer, cache_dir) -> None:
        self.cache_dir = cache_dir
        self.tokenizer = tokenizer

    def export(self, train, valid, test):
        # export to bin files
        train = np.array(train, dtype=np.int16)
        valid = np.array(valid, dtype=np.int16)
        test = np.array(test, dtype=np.int16)
        train_file = os.path.join(self.cache_dir, "train.bin")
        train.tofile(train_file)
        val_file = os.path.join(self.cache_dir, "val.bin")
        valid.tofile(val_file)
        test_file = os.path.join(self.cache_dir, "test.bin")
        test.tofile(test_file)
        return train_file, val_file, test_file

    def _tokenize(self, max_seq_len, add_eos=False, add_bos=False):
        # Pepare the tokenizer counts
        self.tokenizer.count_file(os.path.join(self.cache_dir, "train.txt"))
        self.tokenizer.count_file(os.path.join(self.cache_dir, "valid.txt"))
        self.tokenizer.count_file(os.path.join(self.cache_dir, "test.txt"))
        self.tokenizer.build_vocab()
        train = self.tokenizer.encode_file(
            os.path.join(self.cache_dir, "train.txt"), add_eos=add_eos, add_bos=add_bos
        )
        valid = self.tokenizer.encode_file(
            os.path.join(self.cache_dir, "valid.txt"), add_eos=add_eos, add_bos=add_bos
        )
        test = self.tokenizer.encode_file(
            os.path.join(self.cache_dir, "test.txt"), add_eos=add_eos, add_bos=add_bos
        )

        train_file, val_file, test_file = self.export(train, valid, test)
        return {
            "train": FullFileDataSet(
                np.memmap(train_file, dtype=np.int16, mode="r"),
                max_seq_len,
                self.tokenizer.pad_token_id,
            ),
            "validation": FullFileDataSet(
                np.memmap(val_file, dtype=np.int16, mode="r"),
                max_seq_len,
                self.tokenizer.pad_token_id,
            ),
            "test": FullFileDataSet(
                np.memmap(test_file, dtype=np.int16, mode="r"),
                max_seq_len,
                self.tokenizer.pad_token_id,
            ),
        }

    def tokenize(self, max_seq_len):
        return self._tokenize(max_seq_len, add_eos=False, add_bos=False)

    def get_raw_data(self):
        # TODO implement the bash version of get data
        pass


class Wiki103CharDP(Enwiki8CharDP):
    def tokenize(self, max_seq_len):
        return self._tokenize(max_seq_len, add_eos=True, add_bos=True)

    def get_raw_data(self):
        # TODO implement the bash version of get data
        pass
