from typing import Literal
from flax import struct
import yaml


# TODO: Figure a way to introduce comments and options in the field of this dataclasses


@struct.dataclass
class Config:
    @classmethod
    def load(cls, yaml_file, **kwargs):
        """Read configuration from json
        Args:
        yaml_file: Union[str, os.PathLike]
            Path to the yaml config
        **kwargs: other config parameters which are not part of the config file.
            If they are both part of the config and passed in the method,
                the parameters from the config file will take precendence
        """
        with open(yaml_file, "r", encoding="utf-8") as reader:
            config = yaml.safe_load(reader)
        # update json file configs with the bash configs
        config.update(kwargs)

        config = cls.validate(config)
        return cls(**config)

    @classmethod
    def validate(cls, config):
        if not "name" in config:
            raise NotImplementedError(
                "Experiemnt must have a name. Default not supported"
            )
        if not "base_dir" in config:
            raise NotImplementedError(
                "Experiemnt must have a base_dir. Default not supported"
            )
        return config


@struct.dataclass
class LMTaskConfig(Config):
    # Name of the experiment.
    name: str
    # base directory where to dump trainign output. Experiment name will be a subfolder here.
    base_dir: str
    # The project under which run is saved
    project: str = "diffusion"
    # The team/account under which the project is saved
    entity: str = "baesian-learning"
    # tokenizer path: used for pretrained tokenizers
    tokenizer_path: str = None
    # name of the dataset used for classification
    dataset_name: str = "shakespeare"
    # number of epochs to train the VAE for
    epochs: int = 10
    # number of train steps. Should be set to None if we want to use epochs
    train_steps: int = None
    # Hidden dimension for the MLP layer
    hidden_dim: int = 128
    # Dimention for the rotation matrix
    L: int = 10
    # number of unrolls used for the scan operations in latte
    unroll: int = 100
    # number heads
    nheads: int = 4
    # number layers
    nlayers: int = 6
    # maximum sequence length:
    max_seq_len: int = 1024
    # number of steps between evaluation
    eval_steps: int = 10
    # The maximum number of checkpoints to save
    max_checkpoints: int = 3
    # dropout each layer
    dropout: float = 0.1
    # weight decay for optimizer
    weight_decay: int = 0.01
    # The learning rate"
    lr: float = 3e-4
    # number of steps to do warmup
    warmup_pc: float = 0
    # learning rate decay function
    lr_decay_fn: str = (
        "cosine"  # "constant", "linear" TODO: replace str with typing.Literal
    )
    # end value used only for linear decay learning rate
    lr_end_value: float = 0.00001
    # use batchnorm or layer norm
    batchnorm: float = True
    # normalize before or after mlp
    prenorm: bool = False
    # batch size per device
    batch_size: int = 32
    # gradient accumulation steps
    grad_accumulation_steps: int = 1
    # Path to the pretrained checkpoint, useful for resuming training
    check_path: str = None
    # Whether to use wandb logging
    wandb_log: bool = False
    # whether to process data with hugging face from scratch or not.
    # True = use cached versions
    disable_cache: bool = False
    attention_type: (
        Literal["stable_latte"]
        | Literal["latte"]
        | Literal["standard_causal"]
        | Literal["bid_latte"]
        | Literal["linformer"]
    ) = "stable_latte"
    block_type: Literal["transformer"] | Literal["glu"] = "transformer"


@struct.dataclass
class LRATaskConfig(LMTaskConfig):
    # whether to devide by 255.0 or not
    normalize_img: bool = False
    # Whether to use tokens and embeddings like in nlp for images. (vocab size = 256)
    tokenize_img: bool = False
    # whether to use convolution instead of dense embedding for images.
    # like VitTransformers
    conv_embed: bool = False
    # model of pooling: ["mean", "last"]
    pool: str = "last"
    # num_classes
    num_classes: int = 10
