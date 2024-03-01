import argparse
import os
import logging

from preproc.lm_dp import OpenWebTextDP
from preproc.file_dp import ShakespeareDP, Wiki103CharDP, Enwiki8CharDP
from preproc.lra_dp import ListOpsDP, PathFinderDP, AANDP, IMBDDP, Cifrar10DP
from preproc.toks import (
    CharTokenizer,
    CharFileTokenizer,
    ListOpsTokenizer,
    ByteLevelTokenizer,
    ImageTokenizer,
    SpecialToksGPT2TokenizerFast,
)
logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
LOG = logging.getLogger(__name__)


def get_dp(config):
    if config.dataset_name == "shakespeare":
        cache_dir = os.path.join(config.base_dir, "input", "shakespeare")
        dp = ShakespeareDP(cache_dir=cache_dir)
        raw_data = dp.get_raw_data()
        tokenizer = CharTokenizer(raw_data["train"] + raw_data["validation"])
        tok_data = dp.tokenize(tokenizer, raw_data, max_seq_len=config.max_seq_len)
        return dp, tokenizer, raw_data, tok_data
    elif config.dataset_name == "openwebtext":
        cache_dir = os.path.join(config.base_dir, "input", "openwebtext")
        tokenizer = SpecialToksGPT2TokenizerFast.from_pretrained("gpt2")
        dp = OpenWebTextDP(tokenizer, num_load_procs=8, cache_dir=cache_dir)
        raw_data = dp.get_raw_data()
        tok_data = dp.tokenize(raw_data)
        return dp, tokenizer, raw_data, tok_data
    elif config.dataset_name == "enwik8-char":
        cache_dir = os.path.join(config.base_dir, "input", "chr_data", "enwik8")
        tokenizer = CharFileTokenizer(min_freq=0, max_size=None, lower_case=True)
        dp = Enwiki8CharDP(tokenizer=tokenizer, cache_dir=cache_dir)
        tok_data = dp.tokenize(max_seq_len=config.max_seq_len)
        return dp, tokenizer, None, tok_data
    elif config.dataset_name == "wiki103-char":
        cache_dir = os.path.join(config.base_dir, "input", "chr_data", "wikitext-103")
        tokenizer = CharFileTokenizer(min_freq=0, max_size=None, lower_case=True)
        dp = Enwiki8CharDP(tokenizer=tokenizer, cache_dir=cache_dir)
        tok_data = dp.tokenize(max_seq_len=config.max_seq_len)
        return dp, tokenizer, None, tok_data

    raise IOError("Dataset not configured")


def get_lra_dp(config):
    dp, tokenizer = None, None
    cache_dir = os.path.join(config.base_dir, "input/lra_data/")
    if config.dataset_name == "listops":
        tokenizer = ListOpsTokenizer.from_pretrained(config.tokenizer_path)
        dp = ListOpsDP(tokenizer=tokenizer, cache_dir=cache_dir)
        raw_data = dp.get_raw_data()
        tok_data = dp.tokenize(raw_data, max_length=config.max_seq_len)
    elif config.dataset_name.startswith("pathfinder"):
        tokenizer = ImageTokenizer(vocab_size=None)
        dp = PathFinderDP(
            img_type=config.dataset_name,
            cache_dir=cache_dir,
            disable_cache=config.disable_cache,
            split="hard",
        )
        raw_data = dp.get_raw_data()
        tok_data = dp.tokenize(raw_data)
    elif config.dataset_name == "imdb":
        tokenizer = ByteLevelTokenizer(use_bos=False, use_eos=True)
        dp = IMBDDP(tokenizer, cache_dir)
        raw_data = dp.get_raw_data()
        tok_data = dp.tokenize(raw_data, max_length=config.max_seq_len)
    elif config.dataset_name == "aan":
        tokenizer = ByteLevelTokenizer(use_bos=False, use_eos=True)
        dp = AANDP(tokenizer=tokenizer, cache_dir=cache_dir)
        raw_data = dp.get_raw_data()
        tok_data = dp.tokenize(raw_data, max_length=config.max_seq_len)
    elif config.dataset_name == "cifar10":
        if config.tokenize_img:
            tokenizer = ImageTokenizer(vocab_size=256)
        else:
            tokenizer = ImageTokenizer()
        dp = Cifrar10DP(
            cache_dir=cache_dir, normalize=config.normalize_img, tokenizer=tokenizer
        )
        LOG.info("The data processor is %s ", dp)
        raw_data = dp.get_raw_data()
        tok_data = dp.tokenize(raw_data)
    else:
        raise Exception("Specified dataset not found")
    return dp, tokenizer, raw_data, tok_data


def parse_args():
    # construct the argument parser and parser the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to the configuration",
    )
    parser.add_argument(
        "--base_dir",
        default="/mnt/c/Users/Rares/Documents/phd/diffusion_models/diffusion/data/",
        type=str,
        help="directory where to dump training output",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name of the experiment",
    )
    args = parser.parse_args()
    return args
