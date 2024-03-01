import abc
import numpy as np
from jax import numpy as jnp
import torch
from transformers import BatchEncoding


def torch_collate_fn(original_batch):
    input_ids = torch.tensor([x["input_ids"] for x in original_batch])
    labels = torch.tensor([x["labels"] for x in original_batch])
    pad_mask = None
    if "pad_mask" in original_batch[0]:
        pad_mask = torch.tensor([x["pad_mask"] for x in original_batch])
        return BatchEncoding(
            {"input_ids": input_ids, "labels": labels, "pad_mask": pad_mask}
        )

    return BatchEncoding({"input_ids": input_ids, "labels": labels})


def np_collate_fn(original_batch):
    input_ids = np.array([x["input_ids"] for x in original_batch])
    labels = np.array([x["labels"] for x in original_batch])

    pad_mask = None
    if "pad_mask" in original_batch[0]:
        pad_mask = np.array([x["pad_mask"] for x in original_batch])
        return BatchEncoding(
            {"input_ids": input_ids, "labels": labels, "pad_mask": pad_mask}
        )
    return BatchEncoding({"input_ids": input_ids, "labels": labels})


class Tokenizer(abc.ABC):
    def __init__(self) -> None:
        self.special_tokens = {
            "unk": "<unk>",
            "pad": "<pad>",
            "bos": "<bos>",
            "eos": "<eos>",
        }
        self.mapping = {
            tok: id_
            for tok, id_ in zip(range(len(self.special_tokens)), self.special_tokens)
        }

    @abc.abstractmethod
    def __call__(self):
        pass

    @property
    def pad_token_id(self):
        return self.mapping[self.special_tokens["pad"]]

    @property
    def unk_token_id(self):
        return self.mapping[self.special_tokens["unk"]]

    @property
    def bos_token_id(self):
        return self.mapping[self.special_tokens["bos"]]

    @property
    def eos_token_id(self):
        return self.mapping[self.special_tokens["eos"]]

    @property
    def vocab_size(self):
        return len(self.mapping)


class DataProcessing:
    def get_collate_fn(self, return_type="torch", **kwargs):
        if return_type == "torch":
            return torch_collate_fn
        else:
            return np_collate_fn

    @abc.abstractmethod
    def tokenize(self):
        pass

    @abc.abstractmethod
    def get_raw_data(self):
        pass
