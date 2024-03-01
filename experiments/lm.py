from functools import partial
import os
import logging
import wandb
import jax
import orbax
from datasets import disable_caching

from jax_trainer import Trainer as JaxTrainer, DistributedTrainer
from models.tasks import LMHead
from experiments.utils import get_dp, parse_args
from config import LMTaskConfig
from eval_utils.metric_utils import pred_acc_lm, cross_entropy_loss_lm

LOG = logging.getLogger(__name__)


class LMTask:
    def __init__(self, config) -> None:
        LOG.info("Config is %s", config)
        self.config = config
        self.report_to = "none"
        self.wandb_run = None

        self.out_dir = os.path.join(self.config.base_dir, "out_latte", self.config.name)
        os.makedirs(self.out_dir, exist_ok=True)
        self.set_logger()

        self.dp, self.tokenizer, self.raw_data, self.tokenized_data = get_dp(config)
        self.data_collator = self.dp.get_collate_fn(
            return_type="np", max_seq_len=self.config.max_seq_len
        )
        self.model = LMHead(
            config,
            vocab_size=self.tokenizer.vocab_size,
            pad_id=self.tokenizer.pad_token_id,
        )

    def set_logger(self):
        # configure wandb logs
        if self.config.wandb_log:
            resume = False
            if not self.config.check_path is None:
                resume = True
            wandb_run = wandb.init(
                project=self.config.project,
                entity=self.config.entity,
                name=self.config.name,
                dir=self.out_dir,
                config=self.config,
                resume=resume,
            )
            self.report_to = "wandb"
            self.wandb_run = wandb_run

    def train(self, train_rng):
        train_rng, init_rng = jax.random.split(train_rng, 2)
        trainer = DistributedTrainer(
            config=self.config,
            out_dir=self.out_dir,
            model=self.model,
            train_data=self.tokenized_data["train"],
            val_data=self.tokenized_data["validation"],
            data_collator=self.data_collator,
            evaluator=partial(pred_acc_lm, cross_entropy_loss_lm),
            wandb_run=self.wandb_run,
            rng=init_rng,
        )
        if not self.config.check_path is None:
            trainer.train(train_rng, self.config.check_path)
        else:
            trainer.train(train_rng)

    def generate(self, sample_rng):
        # TODO: WARNING this is in beta mode.
        out_dir = "/home/ubuntu/latte/data/out_latte/enwiki_extend2_latte"
        options = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=self.config.max_checkpoints, create=False
        )
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        mngr = orbax.checkpoint.CheckpointManager(
            os.path.join(out_dir, "checkpoints"), orbax_checkpointer, options
        )
        available_checkpoints = mngr.all_steps(read=True)
        jax.debug.print(
            "Loading last of the availabel checkpoints: {x}",
            x=str(available_checkpoints),
        )

        # prompt = [
        #     "Former secretary of state Hillary Clinton meets voters",
        #     "Today, Toyota announced changes in executives’ areas of responsibility",
        # ]
        prompt = [
            "61 61 72 105 115 116 111 114 121 32 111",
            "73 110 32 49 54 50 48 44 32 119 105 116 104 32 116 104 101 32 111 98 106 101 99 116 32 111 102 32 102 111 114 101 115",
        ]
        prompt = [s.strip().split() for s in prompt]
        state = mngr.restore("360000")["model"]
        samples = self.model.apply(
            {"params": state["params"]},
            gen_shape=(2, 1000),
            rng=sample_rng,
            tokenizer=self.tokenizer,
            promt=prompt,
            method=self.model.sample,
            temperature=0,
        )
        res = []
        for s in samples:
            s = "".join([chr(x) for x in s])  # if x > 1 else ""
            res.append(s)
        print(res)


def main():
    seed = 0
    rng = jax.random.PRNGKey(seed)
    rng, train_rng, sample_rng = jax.random.split(rng, 3)
    args = parse_args()
    config = LMTaskConfig.load(
        yaml_file=args.config_file, base_dir=args.base_dir, name=args.name
    )

    if config.disable_cache:
        LOG.info("Disabling Cache")
        disable_caching()

    task = LMTask(config)
    task.train(train_rng)
    # task.generate(sample_rng)


if __name__ == "__main__":
    main()
