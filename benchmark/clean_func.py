from functools import partial
import jax
from jax import numpy as jnp

jax.config.update("jax_platform_name", "cuda")  # "cuda:0"
DATA_DIR = "/home/ubuntu/latte/data/latte_misc"


@partial(jax.jit, static_argnums=(3, 5))
def causal_latte(Wq, Wk, Wv, H, X, unroll=100):
    """
    Scan implementation of latte.
    B: batch size H: nr heads, T: seq_len, D: hidden_dim. L: latent dimension
    Args:
        Wq: jnp.array(DL), Wk:jnp.array(DL), Wv:jnp.array(DM) - parameter matrices
        H: int - nr heads
        X: jnp.array(BTD) - input
        unroll: int - unroll of the loop
    Returns:
        y: jnp.array(BTD) - transformed output sequence
    """

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

    B, T, D = X.shape
    L = Wk.shape[-1]

    V = jnp.einsum("DM,BTD->TBM", Wv, X).reshape(T, B, H, -1)
    Q = jnp.einsum("DL,BTD->TBL", Wq, X).reshape(T, B, H, -1)
    K = jnp.einsum("DL,BTD->TBL", Wk, X).reshape(T, B, H, -1)
    maxi = jax.lax.cummax(K, axis=0)

    init_alpha = jnp.zeros(shape=(B, H, L // H))
    init_carry = jnp.zeros((B, H, L // H, D // H))
    Qs = jax.nn.softmax(Q, axis=-1)
    _, y = jax.lax.scan(
        accumulate,
        unroll=unroll,
        init=(
            init_carry,
            init_alpha,
            K[0],
        ),
        xs=[Qs, K, V, maxi],
    )
    # TBHD -> BTHD
    y = y.transpose(1, 0, 2, 3)
    return y.reshape(B, T, D)


def ablation_T(B=2, T=10000, C=512, L=512, H=4, repeats=100, limit_att=100000):
    # Batch size is fixed to 1 for now (nned to modify google att)
    UNROLL, qchz = 256, 1000
    master_key = jax.random.PRNGKey(0)
    master_key, key = jax.random.split(master_key)
    master_key, key_qkv, key_q, key_k, key_v, key_x = jax.random.split(
        master_key, num=6
    )

    Wq, Wk, Wv = (
        jax.random.normal(key_q, (C, L)),
        jax.random.normal(key_k, (C, L)),
        jax.random.normal(key_v, (C, C)),
    )
    X = jax.random.normal(key_x, (B, T, C))
    print("-----" * 10)
    print(X.shape)
    out = causal_latte(Wq, Wk, Wv, H, X, unroll=UNROLL).block_until_ready()

    from benchmark.best_bench_time import stable_scan_latte

    out2 = stable_scan_latte(Wq, Wk, Wv, H, X, unroll=UNROLL).block_until_ready()
    jax.debug.print(
        "Stable Latte OK: {}",
        jnp.all(jnp.isclose(out, out2, rtol=1e-3, atol=1e-3)),
    )


if __name__ == "__main__":
    time_ms = ablation_T(B=2, T=100, C=128, L=128, H=4, repeats=100, limit_att=17000)
