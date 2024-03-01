"""
All the data processing for long range arena benchmark
"""

import os
from pathlib import Path
from functools import partial
import numpy as np
from transformers import BatchEncoding
from datasets import (
    load_dataset,
    DatasetDict,
    Dataset,
    Features,
    Value,
    Image,
    load_from_disk,
)

from .base import DataProcessing


class ListOpsDP(DataProcessing):
    def __init__(self, tokenizer, cache_dir) -> None:
        self.cache_dir = cache_dir
        self.tokenizer = tokenizer

    @staticmethod
    def google_clean(element):
        """
        As per google method we need to replace "]" with "X"
        and "(", ")" with "".
        We simply do this to be consistent with google.
        """
        sentence = element["Source"].translate(
            {
                ord("]"): ord("X"),
                ord("("): None,
                ord(")"): None,
            }
        )
        element["Source"] = sentence
        return element

    def tokenize(self, dataset, max_length=1000):
        tok_data = dataset.map(
            lambda x: self.tokenizer(max_length, x),
            batched=False,
            remove_columns=["Source", "Target"],
        )
        tok_data = tok_data.map(
            lambda x: self.tokenizer.pad(max_length, x), batched=False
        )
        return tok_data

    def get_raw_data(self):
        data_dir = os.path.join(self.cache_dir, "lra_release/listops-1000/")
        data_dir = Path(data_dir)
        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(data_dir / "basic_train.tsv"),
                "validation": str(data_dir / "basic_val.tsv"),
                "test": str(data_dir / "basic_test.tsv"),
            },
            delimiter="\t",
            keep_in_memory=False,
            cache_dir=os.path.join(self.cache_dir, "cache", "listops"),
        )
        # clean as google does
        dataset = dataset.map(self.google_clean, batched=False)
        return dataset


class IMBDDP(DataProcessing):
    """
    Args:
        tokenizer: ByteLevelTokenizer
        cache_dir: str
    """

    def __init__(self, tokenizer, cache_dir) -> None:
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir

    def tokenize(self, dataset, max_length=4000):
        tok_data = dataset.map(
            lambda x: self.tokenizer(max_length, x),
            batched=False,
            remove_columns=["Source", "Target"],
        )
        tok_data = tok_data.map(
            lambda x: self.tokenizer.pad(max_length, x), batched=False
        )
        return tok_data

    def get_raw_data(self):
        dataset = load_dataset(
            "imdb", cache_dir=os.path.join(self.cache_dir, "cache", "imdb")
        )
        dataset = dataset.rename_column("text", "Source")
        dataset = dataset.rename_column("label", "Target")
        dataset = DatasetDict(
            {
                "train": dataset["train"],
                "validation": dataset["test"],
                "test": dataset["test"],
            }
        )
        print(dataset)
        return dataset


class AANDP(DataProcessing):
    """
    Args:
        tokenizer: ByteLevelTokenizer
        cache_dir: str
    """

    def __init__(self, tokenizer, cache_dir) -> None:
        self.cache_dir = cache_dir
        self.tokenizer = tokenizer

    def tokenize(self, dataset, max_length=4000):
        tok_data = dataset.map(
            lambda x: self.tokenizer.tokenize_pair(max_length, x),
            batched=False,
            remove_columns=["text1", "text2"],
        )
        tok_data = tok_data.map(
            lambda x: self.tokenizer.pad_pair(max_length, x), batched=False
        )
        return tok_data

    def get_raw_data(self):
        data_dir = os.path.join(
            self.cache_dir, "lra_release", "lra_release", "tsv_data"
        )
        dataset = load_dataset(
            "csv",
            data_files={
                "train": os.path.join(data_dir, "new_aan_pairs.train.tsv"),
                "validation": os.path.join(data_dir, "new_aan_pairs.eval.tsv"),
                "test": os.path.join(data_dir, "new_aan_pairs.test.tsv"),
            },
            delimiter="\t",
            column_names=["label_float", "input1_id", "input2_id", "text1", "text2"],
            cache_dir=os.path.join(self.cache_dir, "cache", "aan"),
        )
        dataset = dataset.remove_columns(["input1_id", "input2_id"])
        dataset = dataset.map(
            lambda x: {"Target": int(x["label_float"])}, remove_columns=["label_float"]
        )
        return dataset


class PathFinderDP(DataProcessing):
    CURVE_MAP = {
        "hard": "curv_contour_length_14",
        "intermediate": "curv_contour_length_9",
        "easy": "curv_baseline",
    }

    def __init__(
        self, img_type="pathfinder32", split="hard", disable_cache=False, cache_dir=None
    ) -> None:
        self.cache_dir = cache_dir
        self.img_type = img_type
        self.split = split
        self.disable_cache = disable_cache

    def _tokenize(self, element, mean, std):
        element["image"] = np.array(element["image"])
        element["image"] = element["image"] / 255.0
        element["image"] = (element["image"] - mean) / std
        element["image"] = element["image"].reshape(-1)[..., None]  # HW-> L1
        result = {"input_ids": element["image"], "labels": element["label"]}
        return BatchEncoding(result)

    def tokenize(self, dataset):
        # dataset = dataset.filter(lambda x: len(x["image"]) > 0, batched=False)
        tok_data = dataset.map(
            partial(self._tokenize, mean=0.5, std=0.5),
            batched=False,
            remove_columns=["image", "label"],
        )
        return tok_data

    def get_raw_data(self):
        save_dir = os.path.join(self.cache_dir, "cache/", self.img_type)
        if os.path.exists(save_dir) and (not self.disable_cache):
            ds2 = load_from_disk(save_dir)
            return ds2

        data_dir = Path(self.cache_dir) / "lra_release/lra_release/" / self.img_type
        samples = {"image": [], "label": []}
        metadata_files = (data_dir / self.CURVE_MAP[self.split] / "metadata").glob(
            "*.npy"
        )
        for metadata_file in metadata_files:
            with open(metadata_file, "r", encoding="utf-8") as f:
                for metadata in f.read().splitlines():
                    metadata = metadata.split()
                    img_path = (
                        (data_dir / self.CURVE_MAP[self.split])
                        / metadata[0]
                        / metadata[1]
                    )
                    label = int(metadata[3])
                    img_path = img_path.absolute().as_posix()
                    samples["image"].append(img_path)
                    samples["label"].append(label)

        features = Features({"image": Image(), "label": Value(dtype="int32")})
        ds = Dataset.from_dict(samples, features=features)
        ds = ds.train_test_split(test_size=0.2, shuffle=True)
        # split test in validation and test
        ds2 = ds["test"].train_test_split(test_size=0.5, shuffle=True)
        ds2["validation"] = ds2["train"]
        # keep train as it was
        ds2["train"] = ds["train"]
        ds2.save_to_disk(save_dir)
        # need to load from disk to ensure cache files are used.
        # Cache files ensure that not all data is loaded when using map
        ds2 = load_from_disk(save_dir)
        return ds2


class Cifrar10DP(DataProcessing):
    def __init__(self, normalize=False, tokenizer=None, cache_dir=None) -> None:

        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.normalize = normalize
        if normalize:
            self.mean = 122.6 / 255.0
            self.std = 61.0 / 255.0
        else:
            self.mean = 122.6
            self.std = 61.0

    @staticmethod
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def _tokenize(self, element):
        element["image"] = np.array(element["image"])
        element["image"] = self.rgb2gray(element["image"])
        if self.normalize:
            element["image"] = element["image"] / 255.0

        # convert to int if an embeding layer will be used
        if self.tokenizer.vocab_size:
            element["image"] = element["image"].astype(np.int32)
            element["image"] = element["image"].reshape(-1)  # HW-> L
        else:
            element["image"] = (element["image"] - self.mean) / self.std
            element["image"] = element["image"].reshape(-1, 1)  # HW-> L

        result = {"input_ids": element["image"], "labels": element["label"]}
        return BatchEncoding(result)

    def tokenize(self, dataset):
        tok_data = dataset.map(
            self._tokenize, batched=False, remove_columns=["image", "label"]
        )
        return tok_data

    def get_raw_data(self):
        dataset = load_dataset(
            "cifar10", cache_dir=os.path.join(self.cache_dir, "cache", "cifrar10")
        )
        tmp = dataset["train"].train_test_split(test_size=0.1)
        dataset["train"] = tmp["train"]
        dataset["validation"] = tmp["test"]
        dataset = dataset.rename_column("img", "image")
        return dataset
