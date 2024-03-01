from typing import Any
from functools import partial
import shutil
import os
from tqdm.auto import tqdm
import numpy as np
import jax
from jax import numpy as jnp
from jax.tree_util import tree_flatten
from flax.training import orbax_utils, train_state
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from flax import linen as nn
import orbax
import optax
from torch.utils.data import DataLoader
import logging

LOG = logging.getLogger(__name__)


class TrainState(train_state.TrainState):
    key: jax.Array


class BatchNormTrainState(train_state.TrainState):
    key: jax.Array
    batch_stats: Any


# Eval function
def eval_step(model_rng, state, batch, batchnorm, compute_metrics):
    input_ids, label = batch
    if batchnorm:
        output, updates = state.apply_fn(
            {"params": state.params, "batch_stats": state.batch_stats},
            input_ids,
            label,
            train=False,
            mutable=["batch_stats"],
        )
    else:
        output = state.apply_fn(
            {"params": state.params},
            input_ids,
            label,
            train=False,
        )
    return compute_metrics(output, labels=label)


def train_step(model_rng, state: train_state.TrainState, batch: jnp.ndarray, batchnorm):
    input_ids, label = batch
    model_rng, dropout_key = jax.random.split(model_rng)

    def loss_fn(params):
        if batchnorm:
            output, updates = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                input_ids,
                label,
                train=True,
                rngs={"dropout": dropout_key},
                mutable=["batch_stats"],
            )
        else:
            output = state.apply_fn(
                {"params": params},
                input_ids,
                label,
                train=True,
                rngs={"dropout": dropout_key},
            )
            updates = None

        loss = output["loss"]
        return loss, (output, updates)

    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = gradient_fn(state.params)
    if batchnorm:
        state = state.apply_gradients(grads=grads, batch_stats=updates["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)
    return state, loss


class Trainer:
    def __init__(
        self,
        config,
        out_dir,
        model,
        train_data,
        val_data,
        test_data=None,
        data_collator=None,
        evaluator=None,  # compute_metrics functions
        wandb_run=None,
        rng=None,
    ) -> None:
        init_rng, rng = jax.random.split(rng, 2)
        self.config = config
        self._out_dir = out_dir
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.data_collator = data_collator
        self.eval_steps = self.config.eval_steps
        self.max_checkpoints = self.config.max_checkpoints
        self._evaluator = evaluator

        self._eval_metrics = []

        os.makedirs(self._out_dir, exist_ok=True)

        options = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=self.max_checkpoints,
            create=True,
            best_fn=self.best_loss,
            best_mode="min",
        )
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        self.checkpoint_manager = orbax.checkpoint.CheckpointManager(
            os.path.join(self._out_dir, "checkpoints"), orbax_checkpointer, options
        )
        self.train_dl = DataLoader(
            self.train_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )

        self.val_loader = DataLoader(
            self.val_data,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
        )

        if not test_data is None:
            self.test_loader = DataLoader(
                test_data,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=self.data_collator,
            )
        else:
            self.test_loader = None

        self.wandb_run = wandb_run
        self._optimizer, self.total_steps = self.prepare_optimizer()

        self.state = self.init_train_state(init_rng)
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(self.state.params))
        jax.debug.print("Number of parameters: {x} M", x=param_count / 1000000)
        self.show_data()
        # jit for efficiency
        self._train_step = jax.jit(train_step, static_argnums=(3,))
        self._eval_step = jax.jit(eval_step, static_argnums=(3, 4))

    def show_data(self):
        dl = DataLoader(
            self.val_data, batch_size=2, shuffle=False, collate_fn=self.data_collator
        )
        batch = next(iter(dl))
        jax.debug.print("Batch has the form: {x}", x=str(batch))

    def safe_wandb_log(self, log_data):
        if not self.wandb_run is None:
            self.wandb_run.log(log_data)

    def get_scheduler(self, total_steps):
        # it is 0 for no warmup
        warmup_steps = int(self.config.warmup_pc * total_steps)
        if self.config.lr_decay_fn == "cosine":
            lr_scheduler = optax.cosine_decay_schedule(
                self.config.lr, decay_steps=(total_steps - warmup_steps), alpha=0.0
            )
        elif self.config.lr_decay_fn == "linear":
            lr_scheduler = optax.linear_schedule(
                init_value=self.config.lr,
                end_value=self.config.lr_end_value,
                transition_steps=(total_steps - warmup_steps),
                transition_begin=int((total_steps - warmup_steps) * 0.25),
            )
        else:
            lr_scheduler = optax.constant_schedule(self.config.lr)
        jax.debug.print(
            "lr scheduler is {x} , warmup {y} Total steps: {z}",
            x=lr_scheduler,
            y=warmup_steps,
            z=total_steps,
        )
        # whether to add warmup or not
        if self.config.warmup_pc > 0:
            warmup_fn = optax.linear_schedule(
                init_value=0.0, end_value=self.config.lr, transition_steps=warmup_steps
            )
            lr_scheduler = optax.join_schedules(
                schedules=[warmup_fn, lr_scheduler], boundaries=[warmup_steps]
            )
        return lr_scheduler

    def prepare_optimizer(self):
        epochs = self.config.epochs

        if self.config.train_steps is None:
            total_steps = epochs * (
                np.ceil(len(self.train_data) / self.config.batch_size)
            )
            total_steps = int(total_steps)
        else:
            total_steps = self.config.train_steps
        if self.eval_steps == 0:
            self.eval_steps = int(
                np.ceil(len(self.train_data) / self.config.batch_size)
            )

        lr_scheduler = self.get_scheduler(total_steps=total_steps)
        self.lr_scheduler = lr_scheduler
        jax.debug.print("Optimizer lr scheduler is {x}", x=lr_scheduler)
        optimizer = optax.inject_hyperparams(optax.adamw)(
            learning_rate=lr_scheduler, weight_decay=0.01
        )

        if self.config.grad_accumulation_steps > 1:
            optimizer = optax.MultiSteps(optimizer, self.config.grad_accumulation_steps)
        # chain with norm
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optimizer)

        return optimizer, total_steps

    def train(self, train_rng, checkpoint_path=None):
        if not checkpoint_path is None:
            options = orbax.checkpoint.CheckpointManagerOptions(
                max_to_keep=self.max_checkpoints, create=True
            )
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            mngr = orbax.checkpoint.CheckpointManager(
                os.path.join(self._out_dir, "checkpoits"), orbax_checkpointer, options
            )
            available_checkpoints = mngr.all_steps(read=True)
            jax.debug.print(
                "Loading last of the availabel checkpoints: {x}",
                x=str(available_checkpoints),
            )
            self.state = mngr.restore(available_checkpoints[-1])
        else:
            save_dir = os.path.join(self._out_dir, "checkponits")
            if os.path.exists(save_dir):
                # remove previous checkpoints if existent
                shutil.rmtree(save_dir)

        jax.debug.print("Trainer total steps: {x}", x=self.total_steps)
        state = self._train(
            train_rng,
            total_steps=self.total_steps,
        )

        # list of dicts to dicts of list
        metrics = self._eval_metrics
        if len(self._eval_metrics) > 0:
            metrics = {
                key: [i[key] for i in self._eval_metrics]
                for key in self._eval_metrics[0]
            }
        return metrics, state

    def init_train_state(self, init_rng):
        batch = next(iter(self.train_dl))
        init_rng, dropout_rng = jax.random.split(init_rng, 2)
        variables = self.model.init(init_rng, **batch)
        optimizer = self._optimizer
        jax.debug.print(
            "Model train parameters: {x}",
            x=jax.tree_map(lambda x: x.shape, variables["params"]),
        )
        # Create a State
        if self.config.batchnorm:
            return BatchNormTrainState.create(
                apply_fn=self.model.apply,
                params=variables["params"],
                batch_stats=variables["batch_stats"],
                tx=optimizer,
                key=dropout_rng,
            )

        return TrainState.create(
            apply_fn=self.model.apply,
            tx=optimizer,
            params=variables["params"],
            key=dropout_rng,
        )

    @staticmethod
    def best_loss(structured):
        flat, tree = tree_flatten(structured)
        flat = [float(x) for x in flat]
        return min(flat)

    def _calc_eval_metric(self, train_rng, state, val_loader):
        eval_metrics = []
        for batch in val_loader:
            train_rng, model_rng = jax.random.split(train_rng)
            metrics = self._eval_step(
                model_rng,
                state,
                (batch["input_ids"], batch["labels"]),
                self.config.batchnorm,
                self._evaluator,
            )
            eval_metrics.append(metrics)

        eval_metrics = jax.device_get(eval_metrics)
        return eval_metrics

    def _train(self, train_rng, total_steps):
        progress_bar = tqdm(range(total_steps), position=0, leave=True)
        it = 0
        all_scores = []
        losses = []
        state = self.state
        while True:
            train_loss = []
            for batch in self.train_dl:
                train_rng, model_rng = jax.random.split(train_rng)
                state, loss = self._train_step(
                    model_rng,
                    state,
                    (batch["input_ids"], batch["labels"]),
                    self.config.batchnorm,
                )
                train_loss.append(loss)

                if (it % self.eval_steps == 0) or (it >= total_steps):
                    eval_metrics = self._calc_eval_metric(
                        train_rng, state, self.val_loader
                    )
                    scores = {
                        "eval_" + k: np.mean([m[k] for m in eval_metrics])
                        for k in eval_metrics[0]
                    }
                    # compute scores on test
                    if not self.test_loader is None:
                        test_metrics = self._calc_eval_metric(
                            train_rng, state, self.test_loader
                        )
                        for k in test_metrics[0]:
                            scores["test_" + k] = np.mean([m[k] for m in test_metrics])

                    tr_loss = jax.device_get(train_loss)
                    scores["train_loss"] = np.mean(tr_loss)
                    # [1] because we chain with grad clip
                    if self.config.grad_accumulation_steps > 1:
                        opt_state = state.opt_state[1].inner_opt_state
                    else:
                        opt_state = state.opt_state[1]
                    scores["learning_rate"] = jax.device_get(
                        opt_state.hyperparams["learning_rate"]
                    ).item()
                    jax.debug.print("Train Loss {x}", x=scores["train_loss"])
                    jax.debug.print("Evaluation scores: {x}", x=scores)
                    self.safe_wandb_log(scores)
                    all_scores.append(scores)
                    ckpt = {"model": state}
                    save_args = orbax_utils.save_args_from_target(ckpt)
                    self.checkpoint_manager.save(
                        it,
                        ckpt,
                        save_kwargs={"save_args": save_args},
                        metrics={"loss": str(scores["eval_loss"])},
                    )

                it += 1
                progress_bar.update(1)
                if it >= total_steps:
                    break
            train_loss = jax.device_get(train_loss)
            losses.append(np.mean(train_loss))
            if it >= total_steps:
                break
        return state


class DistributedTrainer(Trainer):
    def __init__(
        self,
        config,
        out_dir,
        model,
        train_data,
        val_data,
        test_data=None,
        data_collator=None,
        evaluator=None,
        wandb_run=None,
        rng=None,
    ) -> None:
        init_rng, rng = jax.random.split(rng, 2)
        self.config = config
        self._out_dir = out_dir
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.data_collator = data_collator
        self.eval_steps = self.config.eval_steps
        self.max_checkpoints = self.config.max_checkpoints
        self._evaluator = evaluator

        self._eval_metrics = []

        os.makedirs(self._out_dir, exist_ok=True)
        jax.debug.print("Train data {x} ", x=train_data)
        options = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=self.max_checkpoints,
            create=True,
            best_fn=self.best_loss,
            best_mode="min",
        )
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        self.checkpoint_manager = orbax.checkpoint.CheckpointManager(
            os.path.join(self._out_dir, "checkpoints"), orbax_checkpointer, options
        )
        self.train_dl = DataLoader(
            self.train_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,  # needs batch size to be devidable by number of gpus
        )

        self.val_loader = DataLoader(
            self.val_data,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            drop_last=True,
        )
        if not test_data is None:
            self.test_loader = DataLoader(
                test_data,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=self.data_collator,
            )
        else:
            self.test_loader = None

        self.wandb_run = wandb_run

        self._optimizer, self.total_steps = self.prepare_optimizer()

        (
            self.mesh,
            self.sharding_model,
            self.sharding_data,
            self.state,
            _,
        ) = self.prepare_on_device(init_rng)
        self.show_data()

        self._train_step = jax.jit(
            train_step,
            static_argnums=(3,),
            in_shardings=(
                NamedSharding(self.mesh, None),
                self.sharding_model,
                self.sharding_data,
            ),
            out_shardings=(self.sharding_model, NamedSharding(self.mesh, None)),
        )
        self._eval_step = jax.jit(
            eval_step,
            static_argnums=(3, 4),
            in_shardings=(
                NamedSharding(self.mesh, None),
                self.sharding_model,
                self.sharding_data,
            ),
            out_shardings=NamedSharding(self.mesh, None),
        )

    @staticmethod
    def place_on_device(batch, sharding_data):
        # jax.debug.print("Data shape {y} {x}", y=batch.items(), x=batch["input_ids"].shape)
        for k, v in batch.items():
            batch[k] = jax.device_put(v, sharding_data)
        return batch

    def prepare_on_device(self, init_rng):
        init_rng, init_rng2, state_rng = jax.random.split(init_rng, 3)
        num_devices = len(jax.devices())
        jax.debug.print("Number of devices to shard is: {}", num_devices)
        data = next(iter(self.train_dl))
        devices = mesh_utils.create_device_mesh((num_devices,))
        mesh = Mesh(devices, axis_names=("B",))
        sharding_data = NamedSharding(
            mesh,
            PartitionSpec(
                "B",
            ),
        )
        data = tuple(data.values())  # from BatchEncoding to tuple
        # model place on device
        # Empty Partition spec or with None for each axis means replicate
        sharding_model = NamedSharding(mesh, PartitionSpec())

        # flax sharding
        abstract_variables = jax.eval_shape(
            partial(
                self.init_train_state,
                model=self.model,
                optimizer=self._optimizer,
                batchnorm=self.config.batchnorm,
            ),
            init_rng,
            data,
        )
        # default replicates across all devices
        state_sharding = nn.get_sharding(abstract_variables, mesh)

        jit_init_fn = jax.jit(
            self.init_train_state,
            static_argnums=(2, 3, 4),
            in_shardings=(NamedSharding(mesh, None), sharding_data),  # PRNG key and x
            out_shardings=state_sharding,
        )

        initialized_state = jit_init_fn(
            init_rng, data, self.model, self._optimizer, self.config.batchnorm
        )

        param_count = sum(
            x.size for x in jax.tree_util.tree_leaves(initialized_state.params)
        )
        jax.debug.print("Nuber of parameters: {x} M", x=param_count / 1000000)
        return mesh, sharding_model, sharding_data, initialized_state, None  # state2

    @staticmethod
    def init_train_state(init_rng, batch, model, optimizer, batchnorm):
        init_rng, arg_rng, dropout_rng = jax.random.split(init_rng, 3)
        variables = model.init(init_rng, *batch, train=False)
        jax.debug.print(
            "Model train parameters: {x}",
            x=jax.tree_map(lambda x: x.shape, variables["params"]),
        )
        # Create a State
        if batchnorm:
            state = BatchNormTrainState.create(
                apply_fn=model.apply,
                params=variables["params"],
                batch_stats=variables["batch_stats"],
                tx=optimizer,
                key=dropout_rng,
            )
        else:
            state = TrainState.create(
                apply_fn=model.apply,
                tx=optimizer,
                params=variables["params"],
                key=dropout_rng,
                # variables=other_vars
            )
        return state

    def _train(self, train_rng, total_steps):
        progress_bar = tqdm(range(total_steps), position=0, leave=False)
        it = 0
        all_scores = []
        losses = []
        state = self.state
        while True:
            train_loss = []
            for batch in self.train_dl:
                train_rng, model_rng = jax.random.split(train_rng)
                batch = self.place_on_device(batch, self.sharding_data)
                state, loss = self._train_step(
                    model_rng,
                    state,
                    (batch["input_ids"], batch["labels"]),
                    self.config.batchnorm,
                )
                train_loss.append(loss)

                if (it % self.eval_steps == 0) or (it >= total_steps):
                    eval_metrics = self._calc_eval_metric(
                        train_rng, state, self.val_loader
                    )
                    scores = {
                        "eval_" + k: np.mean([m[k] for m in eval_metrics])
                        for k in eval_metrics[0]
                    }
                    # compute scores on test
                    if not self.test_loader is None:
                        test_metrics = self._calc_eval_metric(
                            train_rng, state, self.test_loader
                        )
                        for k in test_metrics[0]:
                            scores["test_" + k] = np.mean([m[k] for m in test_metrics])

                    tr_loss = jax.device_get(train_loss)
                    scores["train_loss"] = np.mean(tr_loss)

                    if self.config.grad_accumulation_steps > 1:
                        opt_state = state.opt_state[1].inner_opt_state
                    else:
                        opt_state = state.opt_state[1]
                    scores["learning_rate"] = jax.device_get(
                        opt_state.hyperparams["learning_rate"]
                    ).item()
                    jax.debug.print("Train Loss {x}", x=scores["train_loss"])
                    jax.debug.print("Evaluation scores: {x}", x=scores)
                    self.safe_wandb_log(scores)
                    all_scores.append(scores)
                    ckpt = {"model": state}
                    save_args = orbax_utils.save_args_from_target(ckpt)
                    self.checkpoint_manager.save(
                        it,
                        ckpt,
                        save_kwargs={"save_args": save_args},
                        metrics={"loss": str(scores["eval_loss"])},
                    )

                it += 1
                progress_bar.update(1)
                if it >= total_steps:
                    break
            train_loss = jax.device_get(train_loss)
            losses.append(np.mean(train_loss))
            if it >= total_steps:
                break
        return state
