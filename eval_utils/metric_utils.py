import math
from jax import numpy as jnp, tree_flatten
import jax
from flax import linen as nn


@jax.jit
def cross_entropy_loss(logits, target):
    target = nn.one_hot(target, num_classes=logits.shape[-1])
    loss = jnp.einsum("BH,BH->B", target, nn.log_softmax(logits, axis=-1))
    loss = jnp.mean(loss, axis=-1)
    return -loss


@jax.jit
def cross_entropy_loss_lm(logits, target, ignore_index=-100):
    """
    Args:
        logits: jnp.array(BLH)
        target: jnp.array(BL, dtype=long)
        ignore_index: must be a negative value
    """
    num_valid = (target != ignore_index).sum(axis=-1)
    # Indices outside the range [0, num_classes) will be encoded as zeros:
    target = nn.one_hot(target, num_classes=logits.shape[-1])
    loss = jnp.einsum("BLH,BLH->BL", target, nn.log_softmax(logits, axis=-1))
    loss = jnp.sum(loss, axis=-1) / num_valid  # mean reduction on sequene level
    loss = jnp.mean(loss, axis=-1)
    return -loss


def acc_class(loss_fn, output, labels):
    logits = output["logits"]
    loss = loss_fn(logits=logits, target=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    return metrics


def pred_acc_lm(loss_fn, output, labels):
    logits = output["logits"]
    labels = labels[:, 1:]
    n_valid = (labels != -100).sum()
    loss = loss_fn(logits=logits, target=labels, ignore_index=-100)
    accuracy = jnp.sum(jnp.argmax(logits, -1) == labels) / n_valid
    metrics = {
        "loss": loss,
        "bpc": loss / math.log(2),
        "ppl": jnp.exp(loss),
        "accuracy": accuracy,
    }
    return metrics


def best_loss(structured):
    flat, tree = tree_flatten(structured)
    flat = [float(x) for x in flat]
    return min(flat)
