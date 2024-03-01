import os
import wandb
import logging
from functools import partial
import jax
from datasets import disable_caching

from jax_trainer import Trainer as JaxTrainer
from models.tasks import Classification, Retreival
from experiments.utils import get_lra_dp, parse_args
from config import LRATaskConfig
from eval_utils.metric_utils import acc_class, cross_entropy_loss

LOG = logging.getLogger(__name__)


class LRATask:
    def __init__(self, config) -> None:
        LOG.info("Config is %s", config)
        self.config = config
        self.report_to = "none"
        self.wandb_run = None

        self.out_dir = os.path.join(
            self.config.base_dir, "out_latte/lra", self.config.name
        )
        os.makedirs(self.out_dir, exist_ok=True)
        self.set_logger()

        self.dp, self.tokenizer, self.raw_data, self.tokenized_data = get_lra_dp(config)
        self.data_collator = self.dp.get_collate_fn(return_type="np")
        print(self.raw_data)
        if config.dataset_name == "aan":
            self.model = Retreival(
                config,
                vocab_size=self.tokenizer.vocab_size,
                pad_id=self.tokenizer.pad_token_id,
            )
        else:
            self.model = Classification(
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

        trainer = JaxTrainer(
            config=self.config,
            out_dir=self.out_dir,
            model=self.model,
            train_data=self.tokenized_data["train"],
            val_data=self.tokenized_data["validation"],
            test_data=self.tokenized_data["test"],
            data_collator=self.data_collator,
            evaluator=partial(acc_class, cross_entropy_loss),
            wandb_run=self.wandb_run,
            rng=init_rng,
        )
        if not self.config.check_path is None:
            trainer.train(train_rng, self.config.check_path)
        else:
            trainer.train(train_rng)


def main():
    seed = 0
    rng = jax.random.PRNGKey(seed)
    rng, train_rng = jax.random.split(rng)
    args = parse_args()
    config = LRATaskConfig.load(
        yaml_file=args.config_file, base_dir=args.base_dir, name=args.name
    )

    if config.disable_cache:
        LOG.info("Disabling Cache")
        disable_caching()

    task = LRATask(config)
    task.train(train_rng)


if __name__ == "__main__":
    main()
