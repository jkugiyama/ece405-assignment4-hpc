"""
Analysis and visualization script for all-reduce benchmark results.

This script loads benchmark results and generates comprehensive visualizations and analysis.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError:
    print("Warning: matplotlib and pandas not available. Install with: pip install matplotlib pandas")
    exit(1)


def load_results(json_path: str) -> list[dict[str, Any]]:
    """Load benchmark results from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def create_dataframe(results: list[dict[str, Any]]) -> "pd.DataFrame":
    """Convert results to pandas DataFrame."""
    df = pd.DataFrame(results)
    df["data_size_label"] = df["data_size_mb"].apply(
        lambda x: f"{int(x)}MB" if x < 1024 else f"{x/1024:.1f}GB"
    )
    return df


def plot_time_vs_data_size(df: "pd.DataFrame", output_dir: Path) -> None:
    """Plot mean time vs data size for each backend and world size."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Gloo CPU plot
    gloo_df = df[df["backend"] == "gloo"]
    ax = axes[0]
    for world_size in sorted(gloo_df["world_size"].unique()):
        subset = gloo_df[gloo_df["world_size"] == world_size].sort_values("data_size_mb")
        ax.plot(
            subset["data_size_mb"],
            subset["mean_ms"],
            marker="o",
            label=f"World Size: {world_size}",
            linewidth=2,
        )

    ax.set_xlabel("Data Size (MB)", fontsize=12)
    ax.set_ylabel("Mean All-Reduce Time (ms)", fontsize=12)
    ax.set_title("Gloo Backend (CPU)", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # NCCL GPU plot
    nccl_df = df[df["backend"] == "nccl"]
    ax = axes[1]
    for world_size in sorted(nccl_df["world_size"].unique()):
        subset = nccl_df[nccl_df["world_size"] == world_size].sort_values("data_size_mb")
        ax.plot(
            subset["data_size_mb"],
            subset["mean_ms"],
            marker="s",
            label=f"World Size: {world_size}",
            linewidth=2,
        )

    ax.set_xlabel("Data Size (MB)", fontsize=12)
    ax.set_ylabel("Mean All-Reduce Time (ms)", fontsize=12)
    ax.set_title("NCCL Backend (GPU)", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(output_dir / "01_time_vs_data_size.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / '01_time_vs_data_size.png'}")
    plt.close()


def plot_scaling_efficiency(df: "pd.DataFrame", output_dir: Path) -> None:
    """Plot scaling efficiency (time per process) across different data sizes."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {"gloo": "C0", "nccl": "C1"}
    markers = {2: "o", 4: "s", 6: "^"}

    for backend in df["backend"].unique():
        for world_size in sorted(df["world_size"].unique()):
            subset = df[(df["backend"] == backend) & (df["world_size"] == world_size)]
            subset = subset.sort_values("data_size_mb")

            # Normalize by world size to see scaling
            normalized_time = subset["mean_ms"] * subset["world_size"]
            ax.plot(
                subset["data_size_mb"],
                normalized_time,
                marker=markers[world_size],
                color=colors[backend],
                label=f"{backend.upper()} (ws={world_size})",
                linewidth=2,
            )

    ax.set_xlabel("Data Size (MB)", fontsize=12)
    ax.set_ylabel("Mean Time × World Size (ms)", fontsize=12)
    ax.set_title("Scaling Efficiency (lower is better)", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(ncol=2, fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(output_dir / "02_scaling_efficiency.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / '02_scaling_efficiency.png'}")
    plt.close()


def plot_bandwidth_analysis(df: "pd.DataFrame", output_dir: Path) -> None:
    """Plot effective bandwidth of all-reduce operation."""
    df_copy = df.copy()

    # Calculate effective bandwidth: data_size / time
    # data_size in MB, time in ms -> MB/s
    df_copy["bandwidth_mb_s"] = df_copy["data_size_mb"] / (df_copy["mean_ms"] / 1000)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Gloo CPU
    gloo_df = df_copy[df_copy["backend"] == "gloo"]
    ax = axes[0]
    for world_size in sorted(gloo_df["world_size"].unique()):
        subset = gloo_df[gloo_df["world_size"] == world_size].sort_values("data_size_mb")
        ax.plot(
            subset["data_size_mb"],
            subset["bandwidth_mb_s"],
            marker="o",
            label=f"World Size: {world_size}",
            linewidth=2,
        )

    ax.set_xlabel("Data Size (MB)", fontsize=12)
    ax.set_ylabel("Effective Bandwidth (MB/s)", fontsize=12)
    ax.set_title("Gloo Backend (CPU)", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # NCCL GPU
    nccl_df = df_copy[df_copy["backend"] == "nccl"]
    ax = axes[1]
    for world_size in sorted(nccl_df["world_size"].unique()):
        subset = nccl_df[nccl_df["world_size"] == world_size].sort_values("data_size_mb")
        ax.plot(
            subset["data_size_mb"],
            subset["bandwidth_mb_s"],
            marker="s",
            label=f"World Size: {world_size}",
            linewidth=2,
        )

    ax.set_xlabel("Data Size (MB)", fontsize=12)
    ax.set_ylabel("Effective Bandwidth (MB/s)", fontsize=12)
    ax.set_title("NCCL Backend (GPU)", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(output_dir / "03_bandwidth_analysis.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / '03_bandwidth_analysis.png'}")
    plt.close()


def plot_backend_comparison_heatmap(df: "pd.DataFrame", output_dir: Path) -> None:
    """Create heatmaps comparing backends at different sizes."""
    data_sizes = sorted(df["data_size_mb"].unique())

    for data_size in data_sizes:
        fig, ax = plt.subplots(figsize=(10, 5))

        subset = df[df["data_size_mb"] == data_size]
        pivot_table = subset.pivot_table(
            values="mean_ms", index="backend", columns="world_size", aggfunc="first"
        )

        im = ax.imshow(pivot_table.values, cmap="YlOrRd", aspect="auto")

        ax.set_xticks(np.arange(len(pivot_table.columns)))
        ax.set_yticks(np.arange(len(pivot_table.index)))
        ax.set_xticklabels(pivot_table.columns)
        ax.set_yticklabels(pivot_table.index)

        ax.set_xlabel("World Size", fontsize=12)
        ax.set_ylabel("Backend", fontsize=12)

        data_size_label = f"{int(data_size)}MB" if data_size < 1024 else f"{data_size/1024:.1f}GB"
        ax.set_title(f"Mean Time (ms) - Data Size: {data_size_label}", fontsize=13, fontweight="bold")

        # Add text annotations
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                value = pivot_table.iloc[i, j]
                text = ax.text(j, i, f"{value:.1f}", ha="center", va="center", color="black")

        plt.colorbar(im, ax=ax, label="Time (ms)")
        plt.tight_layout()
        filename = f"04_heatmap_size_{int(data_size)}mb.png"
        plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_dir / filename}")
        plt.close()


def print_summary_statistics(df: "pd.DataFrame") -> None:
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for backend in sorted(df["backend"].unique()):
        print(f"\n{backend.upper()} Backend:")
        backend_df = df[df["backend"] == backend]

        for world_size in sorted(backend_df["world_size"].unique()):
            subset = backend_df[backend_df["world_size"] == world_size]
            min_time = subset["mean_ms"].min()
            max_time = subset["mean_ms"].max()
            avg_time = subset["mean_ms"].mean()

            min_data = subset[subset["mean_ms"] == min_time]["data_size_mb"].values[0]
            max_data = subset[subset["mean_ms"] == max_time]["data_size_mb"].values[0]

            print(f"  World Size {world_size}:")
            print(f"    - Fastest: {min_time:.3f}ms ({int(min_data)}MB)")
            print(f"    - Slowest: {max_time:.3f}ms ({int(max_data)}MB)")
            print(f"    - Average: {avg_time:.3f}ms")

    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    results_file = Path("benchmark_results/allreduce_results.json")

    if not results_file.exists():
        print(f"Error: Results file not found at {results_file}")
        print("Please run allreduce_benchmark.py first to generate results.")
        exit(1)

    print(f"Loading results from {results_file}...")
    results = load_results(str(results_file))

    print(f"Loaded {len(results)} benchmark results")

    df = create_dataframe(results)

    # Print summary
    print_summary_statistics(df)

    # Create output directory for plots
    output_dir = Path("benchmark_results/plots")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate plots
    print("\nGenerating plots...")
    plot_time_vs_data_size(df, output_dir)
    plot_scaling_efficiency(df, output_dir)
    plot_bandwidth_analysis(df, output_dir)
    plot_backend_comparison_heatmap(df, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
