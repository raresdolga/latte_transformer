"""
Contains all the data pre-processing for datasets used for language modeling (next token prediction):
    Shakespeare, OpenWebText
"""

import os
import requests
import logging
import numpy as np
import jax.numpy as jnp
import torch
from torch.utils.data import Dataset
from functools import partial
from datasets import load_dataset, DatasetDict
from transformers import BatchEncoding

from preproc.base import DataProcessing

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
LOG = logging.getLogger(__name__)


# Functions to sample from a long-tokenized sequence.
# Does Dynamic padding to the min(max_seq_len, max_batch_length). Similar to longest in huggingface DataCollators
def torch_sampl_lm_collate_fn(pad_id, max_seq_len, original_batch):
    given_lens = np.array([len(x["input_ids"]) for x in original_batch])
    tmp = given_lens - max_seq_len
    high = np.where(tmp <= 0, 1, high)
    pad_lens = np.where(tmp >= 0, 0, -tmp)

    start_idx = np.random.randint(high)
    input_ids = torch.tensor(
        [
            x["input_ids"][start_idx[i] : start_idx[i] + max_seq_len]
            + [pad_id] * pad_lens
            for i, x in enumerate(original_batch)
        ]
    )

    labels = input_ids.clone().detach()
    labels[labels == pad_id] = -100
    return BatchEncoding({"input_ids": input_ids, "labels": labels})


# TODO: might be bad for multiplrocessing to pad up to max batch size!
def np_sampl_lm_collate_fn(pad_id, max_seq_len, original_batch):
    given_lens = np.array([len(x["input_ids"]) for x in original_batch])
    tmp = given_lens - max_seq_len
    high = np.where(tmp <= 0, 1, tmp)
    pad_lens = np.where(tmp >= 0, 0, -tmp)

    start_idx = np.random.randint(high)
    input_ids = np.array(
        [
            x["input_ids"][start_idx[i] : start_idx[i] + max_seq_len]
            + [pad_id] * pad_lens[i]
            for i, x in enumerate(original_batch)
        ]
    )

    labels = np.copy(input_ids)
    labels = np.where(labels == pad_id, -100, labels)
    return BatchEncoding({"input_ids": input_ids, "labels": labels})


class OpenWebTextDP(DataProcessing):
    def __init__(self, tokenizer, num_load_procs, cache_dir) -> None:
        self.cache_dir = cache_dir
        self.num_load_procs = num_load_procs
        self.tokenizer = tokenizer

    def get_collate_fn(self, return_type="torch", max_seq_len=1024):
        if return_type == "torch":
            return partial(
                torch_sampl_lm_collate_fn, self.tokenizer.pad_token_id, max_seq_len
            )
        else:
            return partial(
                np_sampl_lm_collate_fn, self.tokenizer.pad_token_id, max_seq_len
            )

    def tokenize(self, data):
        """
        Rows are independent, so we can sample of max_len size
            in the datacollator to reduce padding.
        Args:
            data: DatsetDict
        """
        return data.map(
            lambda x: self.tokenizer(
                x["text"], add_special_tokens=False, return_attention_mask=False
            ),
            remove_columns=["text"],
            batched=True,
            batch_size=10000,
            num_proc=self.num_load_procs,
            cache_file_names={
                "train": os.path.join(self.cache_dir, "tok_openwebtxt_train.bin"),
                "validation": os.path.join(self.cache_dir, "tok_openwebtxt_val.bin"),
            },
        )

    def get_raw_data(self):
        # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
        os.makedirs(self.cache_dir, exist_ok=True)
        dataset = load_dataset(
            "openwebtext", cache_dir=self.cache_dir, num_proc=self.num_load_procs
        )
        # original data only has train
        split_dataset = dataset["train"].train_test_split(
            test_size=0.0005, seed=2357, shuffle=True
        )
        split_dataset["validation"] = split_dataset.pop(
            "test"
        )  # rename the test split to val
        return split_dataset


class Wiki103DP(DataProcessing):
    """
    Process raw data as Dataset
    """

    def __init__(self, tokenizer, num_load_procs, cache_dir):
        self.cache_dir = cache_dir
        self.num_load_procs = num_load_procs
        self.tokenizer = tokenizer

    def get_collate_fn(self, max_seq_len, return_type="torch"):
        if return_type == "torch":
            return partial(
                torch_sampl_lm_collate_fn, self.tokenizer.pad_token_id, max_seq_len
            )
        else:
            return partial(
                np_sampl_lm_collate_fn, self.tokenizer.pad_token_id, max_seq_len
            )

    def tokenize(self, data: DatasetDict):
        """
        Rows are independent, so we can sample of max_len size
            in the datacollator to reduce padding.
        Args:
            data: DatsetDict
        """
        return data.map(
            lambda x: self.tokenizer(
                x["text"], add_special_tokens=False, return_attention_mask=False
            ),
            remove_columns=["text"],
            batched=True,
            batch_size=10000,
            num_proc=self.num_load_procs,
            cache_file_names={
                "train": os.path.join(self.cache_dir, "tok_wiki103_train.bin"),
                "validation": os.path.join(self.cache_dir, "tok_wiki103_val.bin"),
                "test": os.path.join(self.cache_dir, "tok_wiki103_test.bin"),
            },
        )

    def get_raw_data(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        dataset = load_dataset(
            path="wikitext",
            name="wikitext-103-raw-v1",
            cache_dir=self.cache_dir,
            num_proc=self.num_load_procs,
        )
        dataset = dataset.filter(lambda example: len(example["text"]) > 0)
        # dataset['train'] = dataset['train'].select(np.arange(100))
        # dataset['validation'] = dataset['validation'].select(np.arange(100))
        return dataset


class Enwiki8DP(Wiki103DP):
    """
    Process raw data as Dataset
    """

    def __init__(self, tokenizer, num_load_procs, cache_dir):
        super().__init__(tokenizer, num_load_procs, cache_dir)

    def tokenize(self, data: DatasetDict):
        """
        Rows are independent, so we can sample of max_len size
            in the datacollator to reduce padding.
        Args:
            data: DatsetDict
        """
        return data.map(
            lambda x: self.tokenizer(
                x["text"], add_special_tokens=False, return_attention_mask=False
            ),
            remove_columns=["text"],
            batched=True,
            batch_size=10000,
            num_proc=self.num_load_procs,
            cache_file_names={
                "train": os.path.join(self.cache_dir, "tok_enwiki8_train.bin"),
                "validation": os.path.join(self.cache_dir, "tok_enwiki8_val.bin"),
                "test": os.path.join(self.cache_dir, "tok_enwiki8_test.bin"),
            },
        )

    def get_raw_data(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        dataset = dataset = load_dataset(
            "enwik8", cache_dir=self.cache_dir, num_proc=self.num_load_procs
        )
        dataset = dataset.filter(lambda example: len(example["text"]) > 0)
        split_dataset = dataset["train"].train_test_split(
            test_size=0.005, seed=2357, shuffle=True
        )
        split_dataset["validation"] = split_dataset.pop(
            "test"
        )  # rename the test split to val
        return split_dataset
