import matplotlib.pyplot as plt
import os
import pandas as pd

DATA_DIR = "/home/rares/Documents/phd/latte/benchmark/data"


def plot_():
    acc = pd.read_csv(os.path.join(DATA_DIR, "acc.csv"))
    loss = pd.read_csv(os.path.join(DATA_DIR, "loss.csv"))
    plt.figure()
    plt.plot(
        loss["Step"] * 2000,
        loss["gpt_same_param - eval_loss"],
        label="Standard Causal Attention",
    )
    plt.plot(
        loss["Step"] * 2000,
        loss["stable_latte_same_param - eval_loss"],
        label="Causal Platte",
    )
    plt.xlabel("Train Step")
    plt.ylabel("Neg. Log-Likelihood")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(
        acc["Step"] * 2000,
        acc["gpt_same_param - eval_accuracy"],
        label="Standard Causal Attention",
    )
    plt.plot(
        acc["Step"] * 2000,
        acc["stable_latte_same_param - eval_accuracy"],
        label="Causal Platte",
    )
    plt.xlabel("Train Step")
    plt.ylabel("Next Token Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_()
