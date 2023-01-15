import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

RESULT_ROOT = Path("./simulation")

parser = argparse.ArgumentParser("Visualization of simulation results.")
parser.add_argument(
    "--dataset",
    type=str,
    required=False,
    default="FashionMNIST",
    choices=["CIFAR10", "FashionMNIST", "MNIST", "CelebA"],
    help="FL config: dataset name",
)


def subplots_setup(
    num_cols: int,
    num_rows: int,
    xlim: Tuple[Any, Any],
    xticks_major: List[Any],
    xticks_minor: List[Any],
    xticklabels: List[Any],
    ylim: Tuple[Any, Any],
    yticks_major: List[Any],
    yticks_minor: List[Any],
    yticklabels: List[Any],
):
    # rcParams configuration
    plt.rcParams["font.size"] = 24
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.major.width"] = 1.5
    plt.rcParams["ytick.major.width"] = 1.5
    plt.rcParams["xtick.major.size"] = 5.0
    plt.rcParams["ytick.major.size"] = 5.0
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["legend.edgecolor"] = "black"
    plt.rcParams["legend.markerscale"] = 10
    plt.rcParams["font.family"] = ["Arial"]

    # setup subplots
    fig, axes = plt.subplots(
        figsize=(6 * num_cols, 5 * num_rows),
        ncols=num_cols,
        nrows=num_rows,
        sharex=True,
        sharey=True,
    )
    plt.subplots_adjust(top=0.85, bottom=0.12, wspace=0.18, hspace=0.18)

    axes: List[Axes] = np.ravel(axes).tolist()
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_xticks(xticks_major, minor=False)
        ax.set_xticks(xticks_minor, minor=True)
        ax.set_xticklabels(xticklabels)
        ax.set_ylim(ylim)
        ax.set_yticks(yticks_major, minor=False)
        ax.set_yticks(yticks_minor, minor=True)
        ax.set_yticklabels(yticklabels)
        ax.grid(which="both")

    return fig, axes


def metrics_centralized_plot(
    fig: Figure,
    axes: List[Axes],
    dataset: str,
    metrics: str,
    config: Dict[str, Dict[str, str]],
    titles: List[str],
    save_path: str = None,
) -> None:
    root = RESULT_ROOT / dataset

    for idx, target in enumerate(config):
        for strategy, dirname in config[target].items():
            if strategy != "Solo":
                load_path = (
                    root / target / dirname / "metrics" / f"{metrics}_centralized.json"
                )
                with open(load_path, "r") as f:
                    result = json.load(f)
                result_numpy = np.array(result)
                axes[idx].plot(
                    result_numpy[:, 0],
                    result_numpy[:, 1],
                    label=strategy,
                    linewidth=2,
                    alpha=0.7,
                )
                axes[idx].set_title(titles[idx], fontsize=20)

    # legend configuration
    lines, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(lines, labels, loc="upper center", ncol=3)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(4.0)
    fig.supxlabel("Communication Rounds", y=0.05)
    if metrics == "accuracies":
        fig.supylabel("Global Accuracy", x=0.075)
    elif metrics == "losses":
        fig.supylabel("Global Loss", x=0.075)
    plt.savefig(save_path, bbox_inches="tight")


def metrics_distributed_plot(
    fig: Figure,
    axes: List[Axes],
    dataset: str,
    metrics: str,
    config: Dict[str, Dict[str, str]],
    titles: List[str],
    save_path: str = None,
) -> None:
    root = RESULT_ROOT / dataset

    for idx, target in enumerate(config):
        for strategy, dirname in config[target].items():
            load_path = (
                root / target / dirname / "metrics" / f"{metrics}_distributed.json"
            )
            with open(load_path, "r") as f:
                result = json.load(f)
            result_dict = {}
            for i in range(len(result)):
                result_dict[i] = np.array(result[i][1])
            metrics_summary = None
            for cid in result_dict:
                if metrics_summary is None:
                    metrics_summary = result_dict[cid][:, 1]
                metrics_summary = np.vstack((metrics_summary, result_dict[cid][:, 1]))
            metrics_mean = metrics_summary.mean(axis=0)
            metrics_std = metrics_summary.std(axis=0)
            result_numpy = result_dict[0]
            axes[idx].plot(
                result_numpy[:, 0],
                metrics_mean,
                label=strategy,
                linewidth=2,
                alpha=0.7,
            )
            axes[idx].fill_between(
                result_numpy[:, 0],
                metrics_mean + metrics_std,
                metrics_mean - metrics_std,
                alpha=0.15,
            )
            axes[idx].set_title(titles[idx], fontsize=20)

    # legend configuration
    lines, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(lines, labels, loc="upper center", ncol=4)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(4.0)
    fig.supxlabel("Communication Rounds", y=0.05)
    if metrics == "accuracies":
        fig.supylabel("Client Accuracy", x=0.075)
    elif metrics == "losses":
        fig.supylabel("Client Loss", x=0.075)
    plt.savefig(save_path, bbox_inches="tight")


def main():
    args = parser.parse_args()
    work_dir = Path("./visualization/simulation") / args.dataset

    config_path = work_dir / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    partitions = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    # Global Accuracy
    fig, axes = subplots_setup(
        num_cols=3,
        num_rows=2,
        xlim=(0, 500),
        xticks_major=[i * 100 for i in range(6)],
        xticks_minor=[i * 50 for i in range(11)],
        xticklabels=[0, 100, 200, 300, 400, 500],
        ylim=(0, 1.05),
        yticks_major=[0.2 * int(i) for i in range(6)],
        yticks_minor=[0.05 * int(i) for i in range(21)],
        yticklabels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    save_path = work_dir / "GA_summary.pdf"
    metrics_centralized_plot(
        fig=fig,
        axes=axes,
        dataset=args.dataset,
        metrics="accuracies",
        config=config,
        titles=partitions,
        save_path=save_path,
    )

    # Client Accuracy
    fig, axes = subplots_setup(
        num_cols=3,
        num_rows=2,
        xlim=(0, 500),
        xticks_major=[i * 100 for i in range(6)],
        xticks_minor=[i * 50 for i in range(11)],
        xticklabels=[0, 100, 200, 300, 400, 500],
        ylim=(0, 1.05),
        yticks_major=[0.2 * int(i) for i in range(6)],
        yticks_minor=[0.05 * int(i) for i in range(21)],
        yticklabels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    save_path = work_dir / "CA_summary.pdf"
    metrics_distributed_plot(
        fig=fig,
        axes=axes,
        dataset=args.dataset,
        metrics="accuracies",
        config=config,
        titles=partitions,
        save_path=save_path,
    )

    # Global Loss
    fig, axes = subplots_setup(
        num_cols=3,
        num_rows=2,
        xlim=(0, 500),
        xticks_major=[i * 100 for i in range(6)],
        xticks_minor=[i * 50 for i in range(11)],
        xticklabels=[0, 100, 200, 300, 400, 500],
        ylim=(0, 3.05),
        yticks_major=[1.0 * int(i) for i in range(6)],
        yticks_minor=[0.2 * int(i) for i in range(25)],
        yticklabels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    save_path = work_dir / "GL_summary.pdf"
    metrics_centralized_plot(
        fig=fig,
        axes=axes,
        dataset=args.dataset,
        metrics="losses",
        config=config,
        titles=partitions,
        save_path=save_path,
    )

    # Client Loss
    fig, axes = subplots_setup(
        num_cols=3,
        num_rows=2,
        xlim=(0, 500),
        xticks_major=[i * 100 for i in range(6)],
        xticks_minor=[i * 50 for i in range(11)],
        xticklabels=[0, 100, 200, 300, 400, 500],
        ylim=(0, 3.05),
        yticks_major=[1.0 * int(i) for i in range(6)],
        yticks_minor=[0.2 * int(i) for i in range(25)],
        yticklabels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    save_path = work_dir / "CL_summary.pdf"
    metrics_distributed_plot(
        fig=fig,
        axes=axes,
        dataset=args.dataset,
        metrics="losses",
        config=config,
        titles=partitions,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
