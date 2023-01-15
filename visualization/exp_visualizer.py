import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

RESULT_ROOT = Path("./exp")

parser = argparse.ArgumentParser("Visualization of experiment results.")
parser.add_argument(
    "--dataset",
    type=str,
    required=False,
    default="FashionMNIST",
    choices=["CIFAR10", "FashionMNIST"],
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
    plt.rcParams["font.size"] = 20
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
        if xlim is not None:
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
    config: Dict[str, Dict[str, str]],
    save_path: str = None,
) -> None:
    root = RESULT_ROOT / dataset

    for idx, target in enumerate(config):
        for strategy, dirname in config[target].items():
            timestamps_path = (
                root / target / dirname / "metrics" / "timestamps_centralized.json"
            )
            accuracy_path = (
                root / target / dirname / "metrics" / "accuracy_centralized.json"
            )
            with open(timestamps_path, "r") as f:
                timestamps = json.load(f)
            with open(accuracy_path, "r") as f:
                result = json.load(f)
            acc = np.array(result["accuracy"])[:, 1]
            if strategy == "FML":
                time = np.array(timestamps["eval_round"])[:, 1]
                -np.array(timestamps["client_sampling"])[0, 1]
            else:
                time = np.array(timestamps["eval_round"])[:, 1]
                -np.array(timestamps["fog_sampling"])[0, 1]
            time = np.insert(time, 0, 0)
            axes[idx].plot(
                time,
                acc,
                label=strategy,
                linewidth=2,
                alpha=0.7,
            )
            # axes[idx].set_title(titles[idx], fontsize=20)

    # legend configuration
    lines, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(lines, labels, loc="upper center", ncol=3)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(4.0)
    plt.xlabel("Tims [s]", y=0.05)
    plt.ylabel("Global Accuracy", x=0.075)
    plt.savefig(save_path, bbox_inches="tight")


def metrics_distributed_plot(
    fig: Figure,
    axes: List[Axes],
    dataset: str,
    metrics: str,
    config: Dict[str, Dict[str, str]],
    titles: List[str] = None,
    save_path: str = None,
) -> None:
    root = RESULT_ROOT / dataset

    for idx, target in enumerate(config):
        fog_res = []
        client_res = []
        fog_err = []
        client_err = []
        for strategy, dirname in config[target].items():
            timestamps_path = (
                root / target / dirname / "metrics" / "timestamps_federated.json"
            )
            with open(timestamps_path, "r") as f:
                timestamps = json.load(f)
            result = {
                metrics: {"client": [], "fog": []},
            }
            if strategy == "FML":
                for cid in timestamps:
                    result[metrics]["client"].extend(
                        list(np.array(timestamps[cid][metrics])[:, 1])
                    )
                result[metrics]["fog"] = [0.0]
            else:
                for cid in timestamps:
                    if "133" in cid:
                        result[metrics]["fog"].extend(
                            list(np.array(timestamps[cid][metrics])[:, 1])
                        )
                    elif "172" in cid:
                        result[metrics]["client"].extend(
                            list(np.array(timestamps[cid][metrics])[:, 1])
                        )

            fog_res.append(np.array(result[metrics]["fog"]).mean())
            client_res.append(np.array(result[metrics]["client"]).mean())
            fog_err.append(np.array(result[metrics]["fog"]).std())
            client_err.append(np.array(result[metrics]["client"]).std())
    columns = list(config[target].keys())
    axes[idx].bar(columns, fog_res, yerr=fog_err, capsize=10, label="Fog")
    axes[idx].bar(
        columns, client_res, yerr=client_err, capsize=10, bottom=fog_res, label="Client"
    )
    # legend configuration
    lines, labels = axes[idx].get_legend_handles_labels()
    leg = fig.legend(lines, labels, loc="upper center", ncol=3)
    if metrics == "comp":
        plt.ylabel("Computing Time / Round [s]", x=0.075)
    elif metrics == "comm":
        plt.ylabel("Communication Time / Round [s]", x=0.075)
    plt.savefig(save_path, bbox_inches="tight")


def main():
    args = parser.parse_args()
    work_dir = Path("./visualization/exp") / args.dataset

    config_path = work_dir / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    fig, axes = subplots_setup(
        num_cols=1,
        num_rows=1,
        xlim=(0, 3000),
        xticks_major=[i * 1000 for i in range(4)],
        xticks_minor=[i * 500 for i in range(7)],
        xticklabels=[0, 1000, 2000, 3000],
        ylim=(0, 1.05),
        yticks_major=[0.2 * int(i) for i in range(6)],
        yticks_minor=[0.05 * int(i) for i in range(21)],
        yticklabels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    # Global Accuracy with time
    save_path = work_dir / "GA_summary.pdf"
    metrics_centralized_plot(
        fig=fig,
        axes=axes,
        dataset=args.dataset,
        config=config,
        save_path=save_path,
    )
    plt.rcParams["image.cmap"] = "plasma"

    plt.style.use(["seaborn-darkgrid"])
    plt.style.use(["seaborn-colorblind"])
    # Communication time plot
    fig, axes = subplots_setup(
        num_cols=1,
        num_rows=1,
        xlim=None,
        xticks_major=[],
        xticks_minor=[],
        xticklabels=[],
        ylim=(0, 20),
        yticks_major=[5.0 * int(i) for i in range(5)],
        yticks_minor=[2.5 * int(i) for i in range(9)],
        yticklabels=[0.0, 5.0, 10.0, 15.0, 20.0],
    )
    save_path = work_dir / "Computing_summary.pdf"
    metrics_distributed_plot(
        fig=fig,
        axes=axes,
        dataset=args.dataset,
        metrics="comp",
        config=config,
        save_path=save_path,
    )
    # Computing time plot
    fig, axes = subplots_setup(
        num_cols=1,
        num_rows=1,
        xlim=None,
        xticks_major=[],
        xticks_minor=[],
        xticklabels=[],
        ylim=(0, 0.4),
        yticks_major=[0.1 * int(i) for i in range(5)],
        yticks_minor=[0.05 * int(i) for i in range(9)],
        yticklabels=[0.0, 0.1, 0.2, 0.3, 0.4],
    )
    save_path = work_dir / "Communication_summary.pdf"
    metrics_distributed_plot(
        fig=fig,
        axes=axes,
        dataset=args.dataset,
        metrics="comm",
        config=config,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
