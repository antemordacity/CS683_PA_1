import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("conv_results.csv")


# ============================================================
# 1. Speedup vs tile size
#
# One plot for each K.
# Each line is a different matrix size.
# ============================================================

for K in sorted(df["K"].unique()):

    plt.figure(figsize=(8, 6))

    subset = df[df["K"] == K]

    for N in sorted(subset["matrix_size"].unique()):

        data = subset[subset["matrix_size"] == N]

        plt.plot(
            data["tile_size"],
            data["speedup"],
            marker="o",
            label=f"{N}x{N}"
        )

    # Naive = 1x speedup
    plt.axhline(
        1.0,
        linestyle="--",
        label="Naive"
    )

    plt.xlabel("Tile size")
    plt.ylabel("Speedup")
    plt.title(f"Speedup vs Tile Size (K={K})")

    plt.xticks(
        sorted(subset["tile_size"].unique())
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"speedup_vs_tile_K{K}.png",
        dpi=200
    )

    plt.show()


# ============================================================
# 2. Runtime vs tile size
#
# One plot for each K.
# ============================================================

for K in sorted(df["K"].unique()):

    plt.figure(figsize=(8, 6))

    subset = df[df["K"] == K]

    for N in sorted(subset["matrix_size"].unique()):

        data = subset[subset["matrix_size"] == N]

        plt.plot(
            data["tile_size"],
            data["tiled_ms"],
            marker="o",
            label=f"{N}x{N}"
        )

    plt.xlabel("Tile size")
    plt.ylabel("Runtime (ms)")
    plt.title(f"Tiled Runtime vs Tile Size (K={K})")

    plt.xticks(
        sorted(subset["tile_size"].unique())
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"runtime_vs_tile_K{K}.png",
        dpi=200
    )

    plt.show()


# ============================================================
# 3. Best tile size for every matrix size
# ============================================================

best = (
    df.loc[
        df.groupby(["matrix_size", "K"])["speedup"].idxmax()
    ]
    .sort_values(["K", "matrix_size"])
)

print("\nBest tile size for each configuration:\n")

print(
    best[
        [
            "matrix_size",
            "K",
            "tile_size",
            "speedup",
            "tiled_ms"
        ]
    ].to_string(index=False)
)