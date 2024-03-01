"""
Notation:
    B = batch size, T = sequene length, D = embed/hidden dimension, H = number heads
"""

from typing import Tuple
import jax
from flax import linen as nn
from jax import numpy as jnp
import math


def attention_product(q, k, v, mask=None) -> Tuple[jnp.array, jnp.array]:
    """
    q,k,v: jnp.array(BHTD) - Query, Key, Value
    mask: causal or mask for bidirectional
    """
    d_d = q.shape[-1]
    att_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = att_logits / math.sqrt(d_d)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
    attention = nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values, attention


class LinformerAttention(nn.Module):
    """
    Our implemnetation of Linformer code
    """

    L: int  # THIS is L (L = 16)
    n_heads: int = 4
    hidden_dim: int = 128
    max_seq_len: int = 2000

    def setup(self) -> None:
        assert self.hidden_dim % self.n_heads == 0
        head_dim = self.hidden_dim // self.n_heads
        self.W = nn.Dense(
            features=3 * self.hidden_dim,
            use_bias=True,
            kernel_init=jax.nn.initializers.lecun_normal(),
        )
        self.low_rank_E = self.param(
            "low_rank_E",
            jax.nn.initializers.lecun_normal(),
            (self.max_seq_len, self.n_heads, self.L),
        )
        self.low_rank_V = self.param(
            "low_rank_V",
            jax.nn.initializers.lecun_normal(),
            (self.max_seq_len, self.n_heads, self.L),
        )
        self.o_proj = nn.Dense(
            features=self.hidden_dim, kernel_init=jax.nn.initializers.lecun_normal()
        )

    def __call__(self, X: jnp.array, train: bool, **kwargs) -> jnp.array:
        """
        Args:
            X: jnp.array(BTD)
            train: bool - Just part of the interface.
        Returns:
            V: jnp.array(BTD) - transformed output sequence
        """
        batch_sz, seq_len, hidden_sz = X.shape
        norm = jnp.linalg.norm(X, ord=2, axis=-1)
        X = jnp.einsum("BTD,BT->BTD", X, 1 / norm)

        X = self.W(X)  # BTD,DH->BTH

        X = X.reshape(batch_sz, seq_len, 3, self.n_heads, hidden_sz // self.n_heads)
        Q, K, V = (
            X[:, :, 0, ...],
            X[:, :, 1, ...],
            X[:, :, 2, ...],
        )  # jnp.split(X, 3, axis=2) #
        if "attention_mask" in kwargs:
            attention_mask = kwargs[
                "attention_mask"
            ]  # 0 if pad token, 1 if normal token
            K = jnp.einsum(
                "BTHD,BT->BTHD", K, attention_mask
            )  # make vector 0 for pad token so they do not count in projection
            V = jnp.einsum("BTHD,BT->BTHD", V, attention_mask)

        # project K,V
        low_rank_E = self.low_rank_E[:seq_len, ...]
        low_rank_V = self.low_rank_V[:seq_len, ...]
        K = jnp.einsum("THK,BTHD->BHKD", low_rank_E, K)  # sum over T
        V = jnp.einsum("THK,BTHD->BHKD", low_rank_V, V)
        # Q: BTHD, K: BHKD, V: BHKD
        Q = Q.transpose(0, 2, 1, 3)  # BTHD -> BHTD
        V, _ = attention_product(q=Q, k=K, v=V, mask=None)
        V = V.transpose(0, 2, 1, 3)  # BHTD -> BTHD
        V = V.reshape(batch_sz, seq_len, -1)
        return self.o_proj(V)


class ScanLatte(nn.Module):
    """
    Causal latent attention with projection of the input X to V
    Adopt notation closer to attention.
    This is not stable and requires normalisation
    """

    L: int
    n_heads: int = 4
    hidden_dim: int = 128
    dropout: float = 0.2
    unroll: int = 100

    @staticmethod
    def accumulate(carry, args):  #
        Qs_t, K_exp_t, V_t = args  # (B, H, L), (B, H, L), (B, H, D)
        # outer product between K_exp[t] and V[t]
        carry = carry + jnp.einsum("BHL,BHD->BHLD", K_exp_t, V_t)
        y = jnp.einsum("BHL,BHLD->BHD", Qs_t, carry)
        return (carry, y)

    @nn.compact
    def __call__(self, X: jnp.array, train: bool, **kwargs) -> jnp.array:
        """
        B: batch size H: nr heads, T: seq_len, D: hidden_dim. L: latent dimension
        Args:
            X: jnp.array(BTD)
            train: bool - used for dropout
        Returns:
            y: jnp.array(BTD) - transformed output sequence
        """
        Wk = self.param(
            "Wk", jax.nn.initializers.lecun_normal(), (self.hidden_dim, self.L)
        )
        Wq = self.param(
            "Wq", jax.nn.initializers.lecun_normal(), (self.hidden_dim, self.L)
        )
        Wv = self.param(
            "Wv", jax.nn.initializers.lecun_normal(), (self.hidden_dim, self.hidden_dim)
        )
        o_proj = self.param(
            "o_proj",
            jax.nn.initializers.lecun_normal(),
            (self.hidden_dim, self.hidden_dim),
        )
        B, T, D = X.shape
        L, H = self.L, self.n_heads

        Q_drop = nn.Dropout(self.dropout, deterministic=not train)
        resid_drop = nn.Dropout(self.dropout, deterministic=not train)

        # multi-head implementation
        V = jnp.einsum("DM,BTD->TBM", Wv, X).reshape(T, B, H, -1)
        Q = jnp.einsum("DL,BTD->TBL", Wq, X).reshape(T, B, H, -1)
        K = jnp.einsum("DL,BTD->TBL", Wk, X).reshape(T, B, H, -1)

        scale = jax.lax.rsqrt(jnp.array(D // H).astype(jnp.float32))
        K_exp = jnp.exp(K * scale)  # (T, B, H, L)
        alpha = K_exp.cumsum(axis=0)  # (T, B, H, L)

        Qs = jax.nn.softmax(Q * scale, axis=-1)
        Qs = Q_drop(Qs)
        Qs = Qs / alpha  # (T, B, H, L)
        _, y = jax.lax.scan(
            self.accumulate,
            unroll=self.unroll,
            init=jnp.zeros((B, H, L // H, D // H)),
            xs=[Qs, K_exp, V],
        )  # THDB
        y = y.transpose(3, 0, 1, 2)
        return resid_drop(y.reshape(B, T, D) @ o_proj)


class StableScanLatte(nn.Module):
    """
    Numerically stable causal latent attention.
    """

    L: int
    n_heads: int = 4
    hidden_dim: int = 128
    dropout: float = 0.2
    unroll: int = 100

    @staticmethod
    def accumulate(carry, args):
        csum, norm_cumsum, prev_mx = carry
        Qs_t, curr_alph, V_t, c_mx = args
        revert_maxi = jnp.exp(-c_mx + prev_mx)
        add_maxi = jnp.exp(curr_alph - c_mx)

        norm_cumsum = jnp.einsum("BHL,BHL->BHL", norm_cumsum, revert_maxi)
        norm_cumsum += add_maxi
        carry = jnp.einsum("BHLD,BHL->BHLD", csum, revert_maxi)
        carry += jnp.einsum("BHL,BHD->BHLD", add_maxi, V_t)
        y = jnp.einsum("BHL,BHLD->BHD", Qs_t / norm_cumsum, carry)
        return ((carry, norm_cumsum, c_mx), y)

    @nn.compact
    def __call__(self, X: jnp.array, train: bool, **kwargs) -> jnp.array:
        """
        B: batch size H: nr heads, T: seq_len, D: hidden_dim. L: latent dimension
        Args:
            X: jnp.array(BTD)
            train: bool - Constant used for dropout
        Returns:
            y: jnp.array(BTD) - transformed output sequence
        """
        Wk = self.param(
            "Wk", jax.nn.initializers.lecun_normal(), (self.hidden_dim, self.L)
        )
        Wq = self.param(
            "Wq", jax.nn.initializers.lecun_normal(), (self.hidden_dim, self.L)
        )
        Wv = self.param(
            "Wv", jax.nn.initializers.lecun_normal(), (self.hidden_dim, self.hidden_dim)
        )
        o_proj = self.param(
            "o_proj",
            jax.nn.initializers.lecun_normal(),
            (self.hidden_dim, self.hidden_dim),
        )

        Q_drop = nn.Dropout(self.dropout, deterministic=not train)
        resid_drop = nn.Dropout(self.dropout, deterministic=not train)

        B, T, D = X.shape
        L, H = self.L, self.n_heads

        # multi head implementation
        V = jnp.einsum("DM,BTD->TBM", Wv, X).reshape(T, B, H, -1)
        Q = jnp.einsum("DL,BTD->TBL", Wq, X).reshape(T, B, H, -1)
        K = jnp.einsum("DL,BTD->TBL", Wk, X).reshape(T, B, H, -1)
        maxi = jax.lax.cummax(K, axis=0)
        # maxi for stability should be trated as a constant - no grad is faster
        maxi = jax.lax.stop_gradient(maxi)

        init_alpha = jnp.zeros(shape=(B, H, L // H))
        init_carry = jnp.zeros((B, H, L // H, D // H))
        Qs = jax.nn.softmax(Q, axis=-1)
        Qs = Q_drop(Qs)

        _, y = jax.lax.scan(
            self.accumulate,
            unroll=self.unroll,
            init=(
                init_carry,
                init_alpha,
                K[0],
            ),
            xs=[Qs, K, V, maxi],
        )
        # TBHD -> BTHD
        y = y.transpose(1, 0, 2, 3)
        y = y.reshape(B, T, -1)
        return resid_drop(y @ o_proj)


class BidLatte(nn.Module):
    """
    Bidirectional version in which we sum to "T" instead of "t"
    """

    L: int
    n_heads: int = 4
    hidden_dim: int = 128

    def setup(self):
        assert self.hidden_dim % self.n_heads == 0
        self.Wk = self.param(
            "Wk", jax.nn.initializers.lecun_normal(), (self.hidden_dim, self.L)
        )
        self.Wq = self.param(
            "Wq", jax.nn.initializers.lecun_normal(), (self.hidden_dim, self.L)
        )
        self.Wv = self.param(
            "Wv", jax.nn.initializers.lecun_normal(), (self.hidden_dim, self.hidden_dim)
        )
        self.o_proj = self.param(
            "o_proj",
            jax.nn.initializers.lecun_normal(),
            (self.hidden_dim, self.hidden_dim),
        )

    def __call__(self, X: jnp.array, train: bool, **kwargs) -> jnp.array:
        """
        B: batch size H: nr heads, T: seq_len, D: hidden_dim. L: latent dimension
        Args:
            X: jnp.array(BTD)
            train: bool - Just to respect the interface of trainer.
        Returns:
            y: jnp.array(BTD) - transformed output sequence
        """
        B, T, D = X.shape
        L, H = self.L, self.n_heads
        # multi head implementation
        V = jnp.einsum("DM,BTD->TBM", self.Wv, X).reshape(T, B, H, -1)
        Q = jnp.einsum("DL,BTD->TBL", self.Wq, X).reshape(T, B, H, -1)
        K = jnp.einsum("DL,BTD->TBL", self.Wk, X).reshape(T, B, H, -1)

        if "attention_mask" in kwargs:
            V = jnp.einsum(
                "TBHD,BT->TBHD", V, kwargs["attention_mask"]
            )  # make vector 0 for pad token so they do not count

        Qs = jax.nn.softmax(Q, axis=-1)  # T B H L
        maxi = jnp.max(K, axis=0, keepdims=True)
        K = jnp.exp(K - maxi)
        if "attention_mask" in kwargs:
            K = jnp.einsum(
                "TBHD,BT->TBHD", K, kwargs["attention_mask"]
            )  # make vector 0 for pad token so they do not count

        Kv = jnp.einsum("TBHL,TBHD->BHLD", K, V)
        # normalize
        K = K.sum(axis=0)  # BLH
        Kv = jnp.einsum("BHL,BHLD->BHLD", 1 / K, Kv)
        y = jnp.einsum("TBHL,BHLD->BTHD", Qs, Kv)
        y = y.reshape(B, T, -1)
        return y @ self.o_proj
