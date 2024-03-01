from jax import numpy as jnp, nn as jax_nn
import distrax


def sample_tok(logits, rng, temperature=0):
    """
    Sample one word, given its logits
    Args:
        logits: torch.FloatTensor
            Tensor(Batch_size, vocab_size)
        temperature: float
            constant which flattens or peaks the distribution
    """
    # no need to normalize if we need only the peak of the dist
    if temperature == 0:
        return jnp.argmax(logits, axis=1)
    # temperature: low -> sample the original (i.e elem with high prob)
    #              high -> make a wider (closer to uniform) dist as temp increases -> explore more
    p = jax_nn.softmax(logits / temperature, axis=1)
    cat_dist = distrax.Categorical(probs=p)

    sampl = cat_dist._sample_n(rng, n=1)
    return sampl[0, :]
