from typing import Type
from functools import partial
import math
import jax
from flax import linen as nn
from jax import numpy as jnp, lax


class CausalSelfAttention(nn.Module):
    nr_heads: int = 4
    hidden_dim: int = 128
    max_seq_len: int = 2000
    dropout: float = 0.0

    @nn.compact
    def __call__(self, X: jnp.array, train: bool, **kwargs) -> jnp.array:
        # key, query, value projections for all heads, but in a batch
        c_attn = nn.Dense(3 * self.hidden_dim, use_bias=False)
        # output projection
        c_proj = nn.Dense(self.hidden_dim, use_bias=False)

        # regularization
        attn_dropout = nn.Dropout(rate=self.dropout, deterministic=not train)
        resid_dropout = nn.Dropout(rate=self.dropout, deterministic=not train)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        bias = jnp.tril(jnp.ones(shape=(self.max_seq_len, self.max_seq_len))).reshape(
            1, 1, self.max_seq_len, self.max_seq_len
        )

        B, T, C = (
            X.shape
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = jnp.split(c_attn(X), 3, axis=2)
        k = k.reshape(B, T, self.nr_heads, C // self.nr_heads).transpose(
            0, 2, 1, 3
        )  # (B, nh, T, hs)
        q = q.reshape(B, T, self.nr_heads, C // self.nr_heads).transpose(
            0, 2, 1, 3
        )  # (B, nh, T, hs)
        v = v.reshape(B, T, self.nr_heads, C // self.nr_heads).transpose(
            0, 2, 1, 3
        )  # (B, nh, T, hs)

        # manual implementation of attention
        att = (q @ jnp.swapaxes(k, -2, -1)) * (1.0 / math.sqrt(k.shape[-1]))

        # att = att.masked_fill(bias[:,:,:T,:T] == 0, float('-inf'))
        att = jnp.where(bias[:, :, :T, :T] == 0, float("-inf"), att)
        att = jax.nn.softmax(att, axis=-1)

        att = attn_dropout(att)

        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(0, 2, 1, 3).reshape(
            B, T, C
        )  # re-assemble all head outputs side by side

        # output projection
        y = resid_dropout(c_proj(y))
        return y


class ScanCausalSelfAttention(nn.Module):
    """
    Implement the memory efficient version of attention using scans
    """

    nr_heads: int = 4
    hidden_dim: int = 128
    max_seq_len: int = 2000
    query_chunk_attention: int = (
        1024  # Sub-sequence on which to perform normal self-attention
    )
    unroll: int = 100
    dropout: float = 0.0

    @staticmethod
    def _query_chunk_attention(
        drop_layer: Type["nn.Module"],
        query: jnp.array,
        key: jnp.array,
        value: jnp.array,
        bias: jnp.array,
        key_chunk_size: int = 4096,
        precision: Type["lax.Precision"] = lax.Precision.HIGHEST,
        dtype: Type["jnp.dtype"] = jnp.float32,
    ) -> jnp.array:
        num_kv, B, num_heads, k_features = key.shape
        T = query.shape[0]
        B = query.shape[1]
        v_features = value.shape[-1]
        key_chunk_size = min(key_chunk_size, num_kv)
        query = query / jnp.sqrt(k_features).astype(dtype)

        @partial(jax.checkpoint, prevent_cse=False)
        def summarize_chunk(drop_layer, bias, query, key, value):
            attn_weights = jnp.einsum("qbhd,kbhd->bhqk", query, key).astype(dtype)
            attn_weights = jnp.where(
                bias[None, None, ...] == 0, float("-inf"), attn_weights
            )
            attn_weights = attn_weights.transpose(2, 0, 1, 3)  # bhqk->qbhk
            max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
            max_score = jax.lax.stop_gradient(max_score)
            exp_weights = jnp.exp(attn_weights - max_score)

            exp_weights = jnp.einsum("qbhk,qk->qbhk", exp_weights, bias)
            # dropout applied only on the numerator to simulate dropout after softmax
            exp_weights_drop = drop_layer(exp_weights)
            exp_values = jnp.einsum(
                "vbhf,qbhv->qbhf", value, exp_weights_drop, precision=precision
            ).astype(dtype)
            return (
                exp_values,
                exp_weights.sum(axis=-1),
                max_score.reshape((query.shape[0], B, num_heads)),
            )

        def chunk_scanner(drop_layer, chunk_idx):
            key_chunk = lax.dynamic_slice(
                key,
                (chunk_idx, 0, 0, 0),
                slice_sizes=(key_chunk_size, B, num_heads, k_features),
            )

            value_chunk = lax.dynamic_slice(
                value,
                (chunk_idx, 0, 0, 0),
                slice_sizes=(key_chunk_size, B, num_heads, v_features),
            )

            bias_chunk = lax.dynamic_slice(
                bias, (0, chunk_idx), slice_sizes=(T, key_chunk_size)
            )

            return summarize_chunk(
                drop_layer, bias_chunk, query, key_chunk, value_chunk
            )

        fn = nn.vmap(
            chunk_scanner,
            split_rngs={"params": False, "dropout": True},
        )
        chunk_values, chunk_weights, chunk_max = fn(
            drop_layer, jnp.arange(0, num_kv, key_chunk_size)
        )
        global_max = jnp.max(chunk_max, axis=0, keepdims=True)
        max_diffs = jnp.exp(chunk_max - global_max)
        chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
        chunk_weights *= max_diffs

        all_values = chunk_values.sum(axis=0)
        all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)
        return all_values / all_weights

    def mefficient_attention(
        self,
        query: jnp.array,
        key: jnp.array,
        value: jnp.array,
        causal_mask: jnp.array,
        drop_layer: Type["nn.Module"],
        query_chunk_size: int = 1024,
        precision: Type["lax.Precision"] = jax.lax.Precision.HIGHEST,
        dtype: Type["jnp.dtype"] = jnp.float32,
    ):
        num_q, B, num_heads, q_features = query.shape

        def chunk_scanner(drop_layer, chunk_idx, _):
            query_chunk = lax.dynamic_slice(
                query,
                (chunk_idx, 0, 0, 0),
                slice_sizes=(min(query_chunk_size, num_q), B, num_heads, q_features),
            )
            causal_mask_chunk = lax.dynamic_slice(
                causal_mask,
                (chunk_idx, 0),
                slice_sizes=(min(query_chunk_size, num_q), num_q),
            )

            return (
                chunk_idx + query_chunk_size,
                self._query_chunk_attention(
                    drop_layer,
                    query_chunk,
                    key,
                    value,
                    causal_mask_chunk,
                    precision=precision,
                    dtype=dtype,
                ),
            )

        fn = nn.scan(
            chunk_scanner,
            unroll=self.unroll,
            variable_broadcast="params",
            split_rngs={"params": False, "dropout": True},
            length=math.ceil(num_q / query_chunk_size),
        )
        _, res = fn(drop_layer, 0, None)
        return res.reshape(num_q, B, num_heads, value.shape[-1])

    @nn.compact
    def __call__(self, X: jnp.array, train: bool, **kwargs) -> jnp.array:
        """
        Sequential implementation of causal attention
        Args:
            X: jnp.array(BTD) - batch size (B), seq len (T), hidden dim (D)
            train: bool - dropout flag
        Returns:
            y: jnp.array(BTD) - transformed input sequence
        """
        # key, query, value projections for all heads, but in a batch
        c_attn = nn.Dense(3 * self.hidden_dim, use_bias=False)
        # output projection
        c_proj = nn.Dense(self.hidden_dim, use_bias=False)

        # regularization
        attn_dropout = nn.Dropout(rate=self.dropout, deterministic=not train)
        resid_dropout = nn.Dropout(rate=self.dropout, deterministic=not train)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        bias = jnp.tril(jnp.ones(shape=(self.max_seq_len, self.max_seq_len)))
        B, T, C = (
            X.shape
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = jnp.split(c_attn(X), 3, axis=-1)
        head_dim = C // self.nr_heads
        k = k.reshape(B, T, self.nr_heads, head_dim).transpose(
            1, 0, 2, 3
        )  # T B H head_dim
        q = q.reshape(B, T, self.nr_heads, head_dim).transpose(
            1, 0, 2, 3
        )  # T B H head_dim
        v = v.reshape(B, T, self.nr_heads, head_dim).transpose(
            1, 0, 2, 3
        )  # T B H head_dim

        y = (
            self.mefficient_attention(
                q,
                k,
                v,
                bias,
                attn_dropout,
                query_chunk_size=self.query_chunk_attention,
                precision=None,
            )
            .transpose(1, 0, 2, 3)
            .reshape(B, T, C)
        )

        return resid_dropout(c_proj(y))
