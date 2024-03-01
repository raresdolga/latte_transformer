from typing import Dict
from flax import linen as nn
from jax import numpy as jnp
import jax

from config import Config
from eval_utils.metric_utils import cross_entropy_loss_lm, cross_entropy_loss
from eval_utils.sample import sample_tok
from .modules.common_layers import Encoder, TextImageEncoder


class LMHead(nn.Module):
    config: Config
    vocab_size: int
    pad_id: int

    @nn.compact
    def __call__(
        self, input_ids: jnp.array, labels: jnp.array = None, train: bool = False
    ) -> Dict[str, jnp.array]:
        """
        Args:
            input_ids: jnp.array(BL) - input ids
            labels: jnp.array(BL)
            train: bool - used for dropout
        Returns:
            out: Dict[str, jnp.array] - loss and logits
        """
        encoder = Encoder(
            vocab_size=self.vocab_size,
            nlayers=self.config.nlayers,
            hidden_dim=self.config.hidden_dim,
            max_seq_len=self.config.max_seq_len,
            L=self.config.L,
            unroll=self.config.unroll,
            nheads=self.config.nheads,
            dropout=self.config.dropout,
            prenorm=self.config.prenorm,
            batchnorm=self.config.batchnorm,
            block_type=self.config.block_type,
            attention_type=self.config.attention_type,
        )

        head = nn.Dense(self.vocab_size, dtype=jnp.float64)
        X = encoder(input_ids[:, :-1], train=train)  # BLH
        if self.config.prenorm:
            if self.config.batchnorm:
                X = nn.BatchNorm(use_running_average=not train, momentum=0.9)(X)
            else:
                X = nn.LayerNorm()(X)

        logits = head(X)  # BLH -> BLV
        if labels is None:
            return {"logits": logits}

        # ignore pad tokens
        labels = labels[:, 1:]
        loss = cross_entropy_loss_lm(logits=logits, target=labels, ignore_index=-100)

        return {"loss": loss, "logits": logits}

    def sample(self, gen_shape, rng, tokenizer, promt=None, temperature=0):
        """
        #TODO: WARINING: This is still in beta mode
        Batched autoregressive sampling.
        Prompt length is the smallest sequence in the batch
        Args:
            gen_shape: Tuple(int,int)
                (B,T) - batch size and total length of the maximum generated sequence,
                including the initial promt
            rng: jax.random.key
            promt: List[str]
                Incomplete promt used as a starting point.
        """

        if promt is None:
            mini = 1
            promt = jnp.ones(gen_shape, dtype=jnp.int32) * tokenizer.pad_token_id
            promt = promt.at[:, 0].set(tokenizer.bos_token_id)
        else:
            mini = float("inf")
            elems = []
            for s in promt:
                e = tokenizer(s, add_special_tokens=False, return_attention_mask=False)
                elems.append(e["input_ids"])
                mini = min(len(e["input_ids"]), mini)
            promt = jnp.array([e[:mini] for e in elems])
            pad = (
                jnp.ones(shape=(gen_shape[0], gen_shape[1] - mini), dtype=jnp.int32)
                * tokenizer.pad_token_id
            )
            promt = jnp.concatenate([promt, pad], axis=1)

        for t in range(mini, gen_shape[1]):
            rng, sample_key = jax.random.split(rng, 2)
            logits = self.__call__(input_ids=promt, labels=None, train=False)["logits"]
            logits = sample_tok(logits[:, t], sample_key, temperature=temperature)
            promt = promt.at[:, t].set(logits)

        text = tokenizer.batch_decode(promt.tolist())
        return text


class Classification(nn.Module):
    config: Config
    vocab_size: int
    pad_id: int

    def setup(self) -> None:
        self.encoder = TextImageEncoder(
            vocab_size=self.vocab_size,
            max_seq_len=self.config.max_seq_len,
            L=self.config.L,
            unroll=self.config.unroll,
            hidden_dim=self.config.hidden_dim,
            nheads=self.config.nheads,
            nlayers=self.config.nlayers,
            prenorm=self.config.prenorm,
            dropout=self.config.dropout,
            batchnorm=self.config.batchnorm,
            conv_embed=self.config.conv_embed,
            block_type=self.config.block_type,
            attention_type=self.config.attention_type,
        )

        self._head = nn.Dense(features=self.config.num_classes, dtype=jnp.float64)

    def __call__(
        self,
        input_ids: jnp.array,
        labels: jnp.array = None,
        train: bool = False,
        **kwargs
    ) -> Dict[str, jnp.array]:
        """
        Args:
            input_ids: jnp.array(BL) - input ids
            labels: jnp.array(B)
            train: bool - used for dropout
        Returns:
            out: Dict[str, jnp.array] - loss and logits
        """
        batch_size, seq_len = input_ids.shape[0], input_ids.shape[1]
        # mean pooling for non-pad tokens
        if self.vocab_size is None:  # no padding for images
            attention_mask = jnp.ones_like(
                input_ids[..., 0]
            )  # Depth is not important for checks of padding
            sequence_lengths = jnp.ones((batch_size,), dtype=jnp.int32) * seq_len
        else:
            attention_mask = (input_ids != self.pad_id).astype(jnp.int32)
            sequence_lengths = (
                jnp.asarray(jax.lax.eq(input_ids, self.pad_id), dtype=jnp.int32).argmax(
                    -1
                )
                - 1
            )
        X = self.encoder(
            input_ids, train=train, attention_mask=attention_mask, **kwargs
        )
        if self.config.pool == "mean":
            X = jnp.einsum("BSH,BS->BSH", X, attention_mask)
            pooled_x = X.sum(axis=1) / attention_mask.sum(axis=-1)[..., None]
        elif self.config.pool == "last":
            # last non-pad token
            pooled_x = X[jnp.arange(batch_size), sequence_lengths]
        else:
            raise IOError("pooling mode node recognized")
        logits = self._head(pooled_x)
        if not labels is None:
            loss = cross_entropy_loss(logits=logits, target=labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


class Retreival(nn.Module):
    config: Config
    vocab_size: int
    pad_id: int

    def setup(self) -> None:
        self.encoder = TextImageEncoder(
            vocab_size=self.vocab_size,
            max_seq_len=self.config.max_seq_len,
            L=self.config.L,
            unroll=self.config.unroll,
            hidden_dim=self.config.hidden_dim,
            nheads=self.config.nheads,
            nlayers=self.config.nlayers,
            prenorm=self.config.prenorm,
            dropout=self.config.dropout,
            batchnorm=self.config.batchnorm,
            conv_embed=self.config.conv_embed,
            block_type=self.config.block_type,
            attention_type=self.config.attention_type,
        )
        self._dense_1 = nn.Dense(features=self.config.hidden_dim, name="mlp")
        self._head = nn.Dense(features=self.config.num_classes, name="logits")

    def __call__(
        self,
        input_ids: jnp.array,
        labels: jnp.array = None,
        train: bool = False,
        **kwargs
    ) -> Dict[str, jnp.array]:
        """
        Args:
            input_ids: jnp.array(B2L) - input ids
            labels: jnp.array(B)
            train: bool - used for dropout
        Returns:
            out: Dict[str, jnp.array] - loss and logits
        """
        batch_size, _, seq_len = input_ids.shape
        input_ids = input_ids.reshape(2 * batch_size, seq_len)
        # mean pooling for non-pad tokens
        if self.vocab_size is None:  # no padding for images
            attention_mask = jnp.ones_like(
                input_ids[..., 0]
            )  # Depth is not important for checks of padding
            sequence_lengths = seq_len
        else:
            attention_mask = (input_ids != self.pad_id).astype(jnp.int32)
            sequence_lengths = (
                jnp.asarray(jax.lax.eq(input_ids, self.pad_id), dtype=jnp.int32).argmax(
                    -1
                )
                - 1
            )
        X = self.encoder(input_ids, train=train, attention_mask=attention_mask)
        if self.config.pool == "mean":
            X = jnp.einsum("BSH,BS->BSH", X, attention_mask)
            pooled_x = X.sum(axis=1) / attention_mask.sum(axis=-1)[..., None]
        elif self.config.pool == "last":
            # last non-pad token
            pooled_x = X[jnp.arange(2 * batch_size), sequence_lengths, :]
        elif self.config.pool == "CLS":
            pooled_x = X[jnp.arange(2 * batch_size), 0, :]
        else:
            raise IOError("pooling mode node recognized")

        pooled_x = pooled_x.reshape(batch_size, 2, -1)
        out0, out1 = pooled_x[:, 0, :], pooled_x[:, 1, :]
        encoded = jnp.concatenate([out0, out1, out0 - out1, out0 * out1], axis=-1)

        out = nn.gelu(self._dense_1(encoded))
        logits = self._head(out)
        if not labels is None:
            loss = cross_entropy_loss(logits=logits, target=labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
