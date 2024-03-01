"""
Used to benchmark memory
"""

from functools import partial
import math
import json
import numpy as np
import jax
from jax import numpy as jnp, lax
import time
from jax.lax import associative_scan

jax.config.update("jax_platform_name", "cuda")  # "cuda:0"


######################### Standard attention vs latte vs scan attention ###############################
@partial(jax.jit, static_argnums=(3,))
def causal_self_attention(QKV, x, bias, H):
    # key, query, value projections for all heads, but in a batch
    B, T, C = x.shape  # batch size, sequence length, embedding dimensionality (n_embd)
    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    q, k, v = jnp.split(x @ QKV, 3, axis=2)
    head_dim = C // H
    k = k.reshape(B, T, H, head_dim).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
    q = q.reshape(B, T, H, head_dim).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
    v = v.reshape(B, T, H, head_dim).transpose(0, 2, 1, 3)  # (B, nh, T, hs)

    # manual implementation of attention
    att = (q @ jnp.swapaxes(k, -2, -1)) * (1.0 / math.sqrt(k.shape[-1]))
    att = jnp.where(bias[:, :, :T, :T] == False, float("-inf"), att)
    att = jax.nn.softmax(att, axis=-1)
    y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = y.transpose(0, 2, 1, 3).reshape(
        B, T, C
    )  # re-assemble all head outputs side by side
    return y


## Google Att.
def _query_chunk_attention(
    query, key, value, bias, key_chunk_size=4096, dtype=jnp.float32
):
    num_kv, B, num_heads, k_features = key.shape
    T = query.shape[0]
    v_features = value.shape[-1]
    key_chunk_size = min(key_chunk_size, num_kv)
    query = query / jnp.sqrt(k_features).astype(dtype)

    @partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(bias, query, key, value):
        attn_weights = jnp.einsum("qbhd,kbhd->bhqk", query, key).astype(dtype)
        attn_weights = jnp.where(
            bias[None, None, ...] == 0, float("-inf"), attn_weights
        )
        attn_weights = attn_weights.transpose(2, 0, 1, 3)  # bhqk->qbhk
        max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
        max_score = jax.lax.stop_gradient(max_score)
        exp_weights = jnp.exp(attn_weights - max_score)

        # exp_weights = jnp.einsum("qbhk,qk->qbhk", exp_weights, bias)
        exp_values = jnp.einsum("vbhf,qbhv->qbhf", value, exp_weights).astype(dtype)
        return (
            exp_values,
            exp_weights.sum(axis=-1),
            max_score.reshape((query.shape[0], B, num_heads)),
        )

    def chunk_scanner(chunk_idx):
        key_chunk = jax.lax.dynamic_slice(
            key,
            (chunk_idx, 0, 0, 0),
            slice_sizes=(key_chunk_size, B, num_heads, k_features),
        )
        value_chunk = jax.lax.dynamic_slice(
            value,
            (chunk_idx, 0, 0, 0),
            slice_sizes=(key_chunk_size, B, num_heads, v_features),
        )

        bias_chunk = lax.dynamic_slice(
            bias, (0, chunk_idx), slice_sizes=(T, key_chunk_size)
        )

        return summarize_chunk(bias_chunk, query, key_chunk, value_chunk)

    chunk_values, chunk_weights, chunk_max = jax.lax.map(
        chunk_scanner, xs=jnp.arange(0, num_kv, key_chunk_size)
    )

    global_max = jnp.max(chunk_max, axis=0, keepdims=True)
    max_diffs = jnp.exp(chunk_max - global_max)
    chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(axis=0)
    all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)
    return all_values / all_weights


def mefficient_attention(
    bias, query, key, value, query_chunk_size=1024, unroll=100, dtype=jnp.float32
):
    """
    Query, Key, value: (T,B, H, haead_dim)
    """
    num_q, B, num_heads, q_features = query.shape

    def chunk_scanner(chunk_idx, _):
        query_chunk = jax.lax.dynamic_slice(
            query,
            (chunk_idx, 0, 0, 0),
            slice_sizes=(min(query_chunk_size, num_q), B, num_heads, q_features),
        )

        bias_chunk = lax.dynamic_slice(
            bias, (chunk_idx, 0), slice_sizes=(min(query_chunk_size, num_q), num_q)
        )

        return (
            chunk_idx + query_chunk_size,
            _query_chunk_attention(query_chunk, key, value, bias_chunk, dtype=dtype),
        )

    _, res = jax.lax.scan(
        chunk_scanner,
        unroll=unroll,
        init=0,
        xs=None,
        length=math.ceil(num_q / query_chunk_size),
    )
    return res.reshape(num_q, B, num_heads, q_features)


@partial(jax.jit, static_argnums=(2, 3, 4))
def google_linmem_wrp(QKV, x, H, query_chunk_size=100, unroll=100):
    B, T, C = x.shape  # batch size, sequence length, embedding dimensionality (n_embd)
    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    q, k, v = jnp.split(x @ QKV, 3, axis=-1)
    head_dim = C // H
    k = k.reshape(B, T, H, head_dim).transpose(1, 0, 2, 3)  # T B H head_dim
    q = q.reshape(B, T, H, head_dim).transpose(1, 0, 2, 3)  # T B H head_dim
    v = v.reshape(B, T, H, head_dim).transpose(1, 0, 2, 3)  # T B H head_dim

    mask = jnp.tril(jnp.ones(shape=(T, T)))
    return (
        mefficient_attention(
            mask, q, k, v, query_chunk_size=query_chunk_size, unroll=unroll
        )
        .reshape(T, B, -1)
        .transpose(1, 0, 2)
    )


#################################  my latte ########################
@jax.jit
def fn(l, r):
    beta_l, alpha_l, m_l = l
    beta_r, alpha_r, m_r = r
    maxi = jnp.exp(-m_r + m_l)
    alpha = alpha_l * maxi[..., None] + alpha_r
    beta = beta_l * maxi + beta_r
    return (beta, alpha, m_r)


## Mem ineffic latte
# @partial(jax.checkpoint, prevent_cse=False)
@partial(jax.jit, static_argnums=(3,))
def mem_ineff_latte(Wq, Wk, Wv, H, X):
    """
    Args:
        Q: jnp.array(TBHL): normalised beta
        K: jnp.array(TBHL): alpha = w^k_l x_s
        V: jnp.array(TBHD): the values
        maxi: jnp.array(TBHL) Cum maxi for K.
        K_carry: jnp.array(BHL) prev chunk's sum for alpha
        Kv_carry: jnp.array(BHL) prev chunk's sum for Kv
    """

    B, T, C = X.shape
    L = Wk.shape[-1]
    head_dim = C // H
    V = jnp.einsum("DM,BTD->TBM", Wv, X).reshape(T, B, H, head_dim)
    Q = jnp.einsum("DL,BTD->TBL", Wq, X).reshape(T, B, H, L // H)
    K = jnp.einsum("DL,BTD->TBL", Wk, X).reshape(T, B, H, L // H)
    Q = jax.nn.softmax(Q, axis=-1)
    maxi = jax.lax.cummax(K, axis=0)
    maxi = jax.lax.stop_gradient(maxi)

    K = jnp.exp(K - maxi)
    Kv = jnp.einsum("TBHL,TBHD->TBHLD", K, V)
    # stable cumsums
    K, Kv, _ = associative_scan(fn, (K, Kv, maxi), axis=0)  # [0] TBHLD
    # K = associative_scan(fn, (K, maxi), axis=0)[0]  # TBHL
    # K, Kv = K[1:], Kv[1:]
    res = jnp.einsum("TBHLD,TBHL->BTHD", Kv, Q / K)
    return res.reshape(B, T, C)


# unstable version
@partial(jax.jit, static_argnums=(3, 5))
def scan_latte(Wq, Wk, Wv, H, X, unroll=100):
    def accumulate(carry, args):  #
        Qs_t, K_exp_t, V_t = args  # (H,L,B), (H, L, B), (H, D, B)
        # outer product between K_exp[t] and V[t]
        carry = carry + jnp.einsum("HLB,HDB->HLDB", K_exp_t, V_t)
        y = jnp.einsum("HLB,HLDB->HDB", Qs_t, carry)
        return (carry, y)

    B, T, C = X.shape
    L = Wk.shape[-1]
    head_dim = C // H
    V = jnp.einsum("DM,BTD->TMB", Wv, X).reshape(T, H, head_dim, B)
    Q = jnp.einsum("DL,BTD->TLB", Wq, X).reshape(T, H, L // H, B)
    K = jnp.einsum("DL,BTD->TLB", Wk, X).reshape(T, H, L // H, B)

    K_exp = jnp.exp(K)  # (T, H, L, B)
    alpha = K_exp.cumsum(axis=0)  # (T, H, L, B)

    Qs = jax.nn.softmax(Q, axis=-2) / alpha  # (T, L B)
    _, y = jax.lax.scan(
        accumulate,
        unroll=unroll,
        init=jnp.zeros((H, L // H, head_dim, B)),
        xs=[Qs, K_exp, V],
    )  # THDB
    y = y.transpose(3, 0, 1, 2)
    return y.reshape(B, T, C)


@partial(jax.jit, static_argnums=(3, 5))
def stable_scan_latte(Wq, Wk, Wv, H, X, unroll=100):
    def accumulate(carry, args):
        csum, norm_cumsum, prev_mx = carry
        Qs_t, curr_alph, V_t, c_mx = args  # (K,H), (K,H), (H, D)

        revert_maxi = jnp.exp(-c_mx + prev_mx)
        add_maxi = jnp.exp(curr_alph - c_mx)
        norm_cumsum = jnp.einsum("HLB,HLB->HLB", norm_cumsum, revert_maxi)
        norm_cumsum += add_maxi  # BKH
        # outer product between K_exp[t] and V[t]
        # carry = carry + jnp.expand_dims(K_exp_t, 1) @ jnp.expand_dims(V_t, 0) # (K, D)
        csum = jnp.einsum("HLDB,HLB->HLDB", csum, revert_maxi)
        csum += jnp.einsum("HLB,HDB->HLDB", add_maxi, V_t)
        # y = Qs_t @ carry
        y = jnp.einsum("HLB,HLDB->HDB", Qs_t / norm_cumsum, csum)
        return ((csum, norm_cumsum, c_mx), y)

    # multi head implementation
    B, T, C = X.shape
    L = Wk.shape[-1]
    head_dim = C // H
    lat_head_dim = L // H

    V = jnp.einsum("DM,BTD->TMB", Wv, X).reshape(T, H, head_dim, B)
    Q = jnp.einsum("DL,BTD->TLB", Wq, X).reshape(T, H, lat_head_dim, B)
    K = jnp.einsum("DL,BTD->TLB", Wk, X).reshape(T, H, lat_head_dim, B)

    maxi = jax.lax.cummax(K, axis=0)
    init_alpha = jnp.zeros(shape=(H, lat_head_dim, B))
    init_carry = jnp.zeros((H, lat_head_dim, head_dim, B))
    Qs = jax.nn.softmax(Q, axis=-2)  # (T, H, L, B): m_{tk}=beta_tk

    _, y = jax.lax.scan(
        accumulate,
        unroll=unroll,
        init=(init_carry, init_alpha, K[0]),
        xs=[Qs, K, V, maxi],
    )  # T H D B
    y = y.transpose(3, 0, 1, 2)
    return y.reshape(B, T, C)


@partial(jax.jit, static_argnums=(3,))
def bid_latte(Wq, Wk, Wv, H, X):
    B, T, D = X.shape
    # multi head implementation
    V = jnp.einsum("DM,BTD->TBM", Wv, X).reshape(T, B, H, -1)
    Q = jnp.einsum("DL,BTD->TBL", Wq, X).reshape(T, B, H, -1)
    K = jnp.einsum("DL,BTD->TBL", Wk, X).reshape(T, B, H, -1)

    Qs = jax.nn.softmax(Q, axis=-1)  # T B H L
    maxi = jnp.max(K, axis=0, keepdims=True)
    K = jnp.exp(K - maxi)
    Kv = jnp.einsum("TBHL,TBHD->BHLD", K, V)
    # normalize
    K = K.sum(axis=0)  # BLH
    Kv = jnp.einsum("BHL,BHLD->BHLD", 1 / K, Kv)
    y = jnp.einsum("TBHL,BHLD->BTHD", Qs, Kv)
    y = y.reshape(B, T, -1)
    return y


def test_attentions(B=2, T=10000, C=512, L=512, H=4, limit_att=100000):
    # Batch size is fixed to 1 for now (nned to modify google att)
    UNROLL, qchz = 256, 1000
    master_key = jax.random.PRNGKey(0)
    master_key, key = jax.random.split(master_key)
    master_key, key_qkv, key_q, key_k, key_v, key_x = jax.random.split(
        master_key, num=6
    )

    QKV = jax.random.normal(key_qkv, (C, 3 * C))
    Wq, Wk, Wv = (
        jax.random.normal(key_q, (C, L)),
        jax.random.normal(key_k, (C, L)),
        jax.random.normal(key_v, (C, C)),
    )
    X = jax.random.normal(key_x, (B, T, C))

    # dummy run -- this is not necessary
    out4 = google_linmem_wrp(
        QKV, X, H, query_chunk_size=qchz, unroll=UNROLL
    ).block_until_ready()
    out1 = stable_scan_latte(Wq, Wk, Wv, H, X, unroll=UNROLL).block_until_ready()
    if T <= limit_att:
        bias = jnp.tril(jnp.ones(shape=(T, T), dtype=bool)).reshape(1, 1, T, T)
        out3 = causal_self_attention(QKV, X, bias, H).block_until_ready()

    with jax.profiler.trace(f"/tmp/tensorboard_final"):
        out4 = google_linmem_wrp(
            QKV, X, H, query_chunk_size=qchz, unroll=UNROLL
        ).block_until_ready()
        out1 = stable_scan_latte(Wq, Wk, Wv, H, X, unroll=UNROLL).block_until_ready()
        bid_lat_ = bid_latte(Wq, Wk, Wv, H, X).block_until_ready()
        if T <= limit_att:
            bias = jnp.tril(jnp.ones(shape=(T, T), dtype=bool)).reshape(1, 1, T, T)
            out3 = causal_self_attention(QKV, X, bias, H).block_until_ready()


if __name__ == "__main__":
    res = {}
    for T in [1000, 3000, 5000, 7000, 10000, 12000, 14000, 16000]:
        test_attentions(B=2, T=T, C=128, L=128, H=4, limit_att=20000)
