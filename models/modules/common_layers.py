"""
Common layers used in different models
Notation (if not otherwise specified):
    B - batch size, T - Sequence Length, D - embed dimension, L - number latents
"""

from typing import Any, Literal, Type
import math
from flax import linen as nn
from jax import numpy as jnp

from .latte import (
    StableScanLatte,
    LinformerAttention,
    ScanLatte,
    BidLatte,
)
from .attention import CausalSelfAttention, ScanCausalSelfAttention


def mixing_layer_factory(self: Type["TransBlock"] | Type["GLUBlock"]):
    match self.attention_type:
        case "stable_latte":
            return StableScanLatte(
                n_heads=self.nheads,
                hidden_dim=self.hidden_dim,
                L=self.L,
                dropout=self.dropout,
                unroll=self.unroll,
            )
        case "bid_latte":
            return BidLatte(L=self.L, n_heads=self.nheads, hidden_dim=self.hidden_dim)
        case "standard_causal":
            return CausalSelfAttention(
                nr_heads=self.nheads,
                hidden_dim=self.hidden_dim,
                max_seq_len=self.max_seq_len,
                dropout=self.dropout,
            )
        case "linformer":
            return LinformerAttention(
                L=self.L,
                n_heads=self.nheads,
                hidden_dim=self.hidden_dim,
                max_seq_len=self.max_seq_len,
            )
        case "scan_standard_causal":
            return ScanCausalSelfAttention(
                nr_heads=self.nheads,
                hidden_dim=self.hidden_dim,
                max_seq_len=self.max_seq_len,
                dropout=self.dropout,
                unroll=self.unroll,
                query_chunk_attention=1024,
            )
        case _ as unreachable:
            raise IOError("Type of attention not supported")


# TODO use config for all layers apart from deepest ones like latte
class PositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 2000

    def setup(self):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        max_len = self.max_len
        d_model = self.d_model
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = jnp.zeros((max_len, d_model))
        position = jnp.arange(0, max_len, dtype=jnp.float32)[..., None]
        div_term = jnp.exp(
            jnp.arange(0, d_model, 2, dtype=jnp.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe = pe[None, ...]

    def __call__(self, x: jnp.array, embeds: jnp.array) -> jnp.array:
        """
        Args:
            x: jnp.array(BT) - input ids
            embeds: jnp.array(BTD) - embedding vectors
        """
        a = self.pe[:, : x.shape[1]]
        return a + embeds


class ConvEmbed(nn.Module):
    """
    Pass input through a convolution network
    """

    dropout: float
    hidden_dim: int

    @nn.compact
    def __call__(self, X: jnp.array, train: bool = False) -> jnp.array:
        """
        Args:
            X: (batch_size, (W*H), 1)
            train: bool. Used for dropout
        """
        conv_dims = (1, 24, 48, 96, self.hidden_dim)  # 192
        conv_layers = [
            nn.Conv(
                features=conv_dims[i + 1], kernel_size=(3, 3), strides=1, padding="SAME"
            )
            for i in range(0, len(conv_dims) - 1)
        ]
        norm = nn.LayerNorm()
        drop = nn.Dropout(self.dropout, deterministic=not train)

        batch_sz, seq_len, inp_ch = X.shape
        W = int(math.sqrt(seq_len))
        H = W
        X = X.reshape(batch_sz, W, H, 1)
        for l in conv_layers:
            X = l(X)
        X = X.reshape(batch_sz, seq_len, -1)
        X = norm(drop(X))
        return X


class GLUBlock(nn.Module):
    hidden_dim: int  # model dimention
    max_seq_len: int  # sequence length
    L: int  # number of latent variables
    nheads: int  # number of heads
    unroll: int  # unroll used for the scan operation
    prenorm: bool = True
    dropout: float = 0.0
    batchnorm: bool = True
    attention_type: (
        Literal["stable_latte"]
        | Literal["latte"]
        | Literal["standard_causal"]
        | Literal["linformer"]
    ) = "stable_latte"

    @nn.compact
    def __call__(self, X, train: bool, **kwargs) -> jnp.array:
        "X: B L D"
        if self.batchnorm:
            norm1 = nn.BatchNorm(use_running_average=not train, momentum=0.9)
        else:
            norm1 = nn.LayerNorm()

        drop = nn.Dropout(
            self.dropout,
            deterministic=not train,
        )
        match self.attention_type:
            case "stable_latte":
                lru = StableScanLatte(
                    n_heads=self.nheads,
                    hidden_dim=self.hidden_dim,
                    L=self.L,
                    unroll=self.unroll,
                )
            case "latte":
                lru = ScanLatte(
                    n_heads=self.nheads,
                    hidden_dim=self.hidden_dim,
                    L=self.L,
                    unroll=self.unroll,
                )
            case "bid_latte":
                lru = BidLatte(
                    L=self.L, n_heads=self.nheads, hidden_dim=self.hidden_dim
                )
            case "standard_causal":
                lru = CausalSelfAttention(
                    nr_heads=self.nheads,
                    hidden_dim=self.hidden_dim,
                    max_seq_len=self.max_seq_len,
                    dropout=self.dropout,
                )
            case "linformer":
                lru = LinformerAttention(
                    L=self.L,
                    n_heads=self.nheads,
                    hidden_dim=self.hidden_dim,
                    max_seq_len=self.max_seq_len,
                )
        out = nn.Dense(features=self.hidden_dim, use_bias=True)
        out2 = nn.Dense(features=self.hidden_dim, use_bias=True)

        skip = X
        if self.prenorm:
            X = norm1(X)
        X = lru(X, train, **kwargs)
        X = X + skip
        skip = X
        # full glu
        X = drop(nn.gelu(X))
        X = out(X) * nn.sigmoid(out2(X))
        X = drop(X)
        X = X + skip
        if not self.prenorm:
            X = norm1(X)
        return X


class TransBlock(nn.Module):
    """
    Implements a standard transformer block where the attention layer is replaced with mine
    """

    hidden_dim: int  # model dimention
    max_seq_len: int  # sequence length
    L: int  # number of latent variables
    nheads: int  # number of heads
    unroll: int  # unrolls used for the scan operation
    prenorm: bool = True
    dropout: float = 0.0
    batchnorm: bool = False
    attention_type: (
        Literal["stable_latte"]
        | Literal["latte"]
        | Literal["standard_causal"]
        | Literal["bid_latte"]
        | Literal["linformer"]
    ) = "stable_latte"

    @nn.compact
    def __call__(self, X: jnp.array, train: bool, **kwargs) -> jnp.array:
        """
        Args:
            X: jnp.array(BTD), B = Batch size, T = sequence length, D = embed dimension
            train: bool - used for dropout
        Returns:
            out: jnp.array(BTD) - transformed output sequence
        """
        if self.batchnorm:
            norm1 = nn.BatchNorm(use_running_average=not train, momentum=0.9)
            norm2 = nn.BatchNorm(use_running_average=not train, momentum=0.9)
        else:
            norm1 = nn.LayerNorm()
            norm2 = nn.LayerNorm()

        lru = mixing_layer_factory(self)

        # Two - layer MLP
        mlp = [
            nn.Dense(4 * self.hidden_dim),
            nn.gelu,
            nn.Dense(self.hidden_dim),
            nn.Dropout(self.dropout, deterministic=not train),
        ]

        drop = nn.Dropout(self.dropout, deterministic=not train)

        skip = X
        if self.prenorm:
            X = norm1(X)
        X = lru(X, train, **kwargs)  # apply a mixing layer, like attention
        X = skip + drop(X)
        if not self.prenorm:
            X = norm1(X)
        # MLP part
        skip = X
        if self.prenorm:
            X = norm2(X)
        for l in mlp:
            X = l(X)
        X = skip + X
        if not self.prenorm:
            X = norm2(X)
        return X


class Encoder(nn.Module):
    vocab_size: int
    max_seq_len: int
    L: int  # latent dimension for latte, projection for linformer
    unroll: int
    hidden_dim: int
    nheads: int
    nlayers: int
    prenorm: bool = True
    dropout: float = 0.0
    batchnorm: bool = True
    block_type: str = "transformer"
    attention_type: (
        Literal["stable_latte"]
        | Literal["latte"]
        | Literal["standard_causal"]
        | Literal["bid_latte"]
        | Literal["linformer"]
    ) = "stable_latte"

    @nn.compact
    def __call__(self, X: jnp.array, train: bool = False, **kwargs) -> jnp.array:
        """
        Args:
            X: jnp.array(BTD), B = Batch size, T = sequence length, D = embed dimension
            train: bool - used for dropout
        Returns:
            out: jnp.array(BTD) - transformed output sequence
        """
        embed = nn.Embed(
            num_embeddings=self.vocab_size, features=self.hidden_dim, dtype=jnp.float64
        )
        pos_embeds = PositionalEncoding(
            d_model=self.hidden_dim, max_len=self.max_seq_len
        )
        drop = nn.Dropout(self.dropout, deterministic=not train)
        if self.block_type == "transformer":
            block = TransBlock
        elif self.block_type == "glu":
            block = GLUBlock
        else:
            raise IOError("Block type not supported")

        enc_layers = [
            block(
                hidden_dim=self.hidden_dim,
                L=self.L,
                unroll=self.unroll,
                nheads=self.nheads,
                max_seq_len=self.max_seq_len,
                prenorm=self.prenorm,
                dropout=self.dropout,
                batchnorm=self.batchnorm,
                attention_type=self.attention_type,
            )
            for _ in range(self.nlayers)
        ]
        embeds = embed(X)
        X = pos_embeds(X, embeds)
        X = drop(X)
        for l in enc_layers:
            X = l(X, train, **kwargs)
        return X


class TextImageEncoder(nn.Module):
    """
    Deals with images and text.
    """

    vocab_size: int
    max_seq_len: int
    L: int  # latent dimension for latte, projection for linformer
    unroll: int
    hidden_dim: int
    nheads: int
    nlayers: int
    prenorm: bool = True
    dropout: float = 0.0
    batchnorm: bool = True
    conv_embed: bool = False
    block_type: str = "transformer"
    attention_type: (
        Literal["stable_latte"]
        | Literal["latte"]
        | Literal["standard_causal"]
        | Literal["bid_latte"]
        | Literal["linformer"]
    ) = "stable_latte"

    @nn.compact
    def __call__(self, X: jnp.array, train: bool = False, **kwargs) -> jnp.array:
        """
        Args:
            X: jnp.array(BTD), B = Batch size, T = sequence length, D = embed dimension
            train: bool - used for dropout
        Returns:
            out: jnp.array(BTD) - transformed output sequence
        """
        if self.vocab_size is None:  # images do not require embedding layer
            if self.conv_embed:
                embed = ConvEmbed(dropout=self.dropout, hidden_dim=self.hidden_dim)
            else:
                embed = nn.Dense(features=self.hidden_dim)  #
        else:
            embed = nn.Embed(
                num_embeddings=self.vocab_size,
                features=self.hidden_dim,
                dtype=jnp.float32,
            )

        pos_embeds = PositionalEncoding(
            d_model=self.hidden_dim, max_len=self.max_seq_len
        )

        drop = nn.Dropout(self.dropout, deterministic=not train)
        if self.block_type == "transformer":
            block = TransBlock
        elif self.block_type == "glu":
            block = GLUBlock
        else:
            raise IOError("Block type not supported")
        enc_layers = [
            block(
                hidden_dim=self.hidden_dim,
                L=self.L,
                unroll=self.unroll,
                nheads=self.nheads,
                max_seq_len=self.max_seq_len,
                prenorm=self.prenorm,
                dropout=self.dropout,
                batchnorm=self.batchnorm,
                attention_type=self.attention_type,
            )
            for _ in range(self.nlayers)
        ]
        embeds = embed(X)
        X = pos_embeds(X, embeds)
        X = drop(X)
        for l in enc_layers:
            X = l(X, train, **kwargs)
        return X
