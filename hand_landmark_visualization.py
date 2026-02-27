import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── MediaPipe HAND_CONNECTIONS (landmark index pairs) ──────────────────────
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17),
    # Middle finger base to wrist
    (0, 5), (5, 9),
]

# ── Finger colour coding ────────────────────────────────────────────────────
FINGER_COLORS = {
    "wrist":  "#888888",
    "thumb":  "#E74C3C",
    "index":  "#3498DB",
    "middle": "#2ECC71",
    "ring":   "#9B59B6",
    "pinky":  "#F39C12",
    "palm":   "#555555",
}

LANDMARK_COLORS = (
    [FINGER_COLORS["wrist"]]   +   # 0
    [FINGER_COLORS["thumb"]]  * 4  +   # 1-4
    [FINGER_COLORS["index"]]  * 4  +   # 5-8
    [FINGER_COLORS["middle"]] * 4  +   # 9-12
    [FINGER_COLORS["ring"]]   * 4  +   # 13-16
    [FINGER_COLORS["pinky"]]  * 4      # 17-20
)

CONNECTION_COLORS = {
    (0,1):   FINGER_COLORS["thumb"],
    (1,2):   FINGER_COLORS["thumb"],
    (2,3):   FINGER_COLORS["thumb"],
    (3,4):   FINGER_COLORS["thumb"],
    (0,5):   FINGER_COLORS["palm"],
    (5,6):   FINGER_COLORS["index"],
    (6,7):   FINGER_COLORS["index"],
    (7,8):   FINGER_COLORS["index"],
    (9,10):  FINGER_COLORS["middle"],
    (10,11): FINGER_COLORS["middle"],
    (11,12): FINGER_COLORS["middle"],
    (13,14): FINGER_COLORS["ring"],
    (14,15): FINGER_COLORS["ring"],
    (15,16): FINGER_COLORS["ring"],
    (0,17):  FINGER_COLORS["palm"],
    (17,18): FINGER_COLORS["pinky"],
    (18,19): FINGER_COLORS["pinky"],
    (19,20): FINGER_COLORS["pinky"],
    (5,9):   FINGER_COLORS["palm"],
    (9,13):  FINGER_COLORS["palm"],
    (13,17): FINGER_COLORS["palm"],
}

# ── Landmark index labels ───────────────────────────────────────────────────
LANDMARK_NAMES = [
    "Wrist",
    "Thumb CMC", "Thumb MCP", "Thumb IP", "Thumb Tip",
    "Index MCP", "Index PIP", "Index DIP", "Index Tip",
    "Middle MCP","Middle PIP","Middle DIP","Middle Tip",
    "Ring MCP",  "Ring PIP",  "Ring DIP",  "Ring Tip",
    "Pinky MCP", "Pinky PIP", "Pinky DIP", "Pinky Tip",
]


def normalize_landmarks(row_x, row_y):
    """
    Apply the same normalization used in training:
      1. Subtract wrist (landmark 0) to recentre.
      2. Divide X and Y by mean distance to the 4 fingertips (8,12,16,20).
    """
    x = row_x.copy()
    y = row_y.copy()

    # Translation
    x -= x[0]
    y -= y[0]

    # Scale
    fingertip_idx = [8, 12, 16, 20]
    dists = np.sqrt(x[fingertip_idx] ** 2 + y[fingertip_idx] ** 2)
    scale = dists.mean()
    if scale < 1e-6:
        scale = 1e-6
    x /= scale
    y /= scale

    return x, y


def draw_hand(ax, x, y, title="", show_labels=False, alpha=0.6):
    """Draw a single hand skeleton on axis `ax`."""
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")

    # Connections
    for (i, j), color in CONNECTION_COLORS.items():
        ax.plot([x[i], x[j]], [-y[i], -y[j]],
                color=color, linewidth=2, alpha=alpha, zorder=1)

    # Landmark dots
    for idx in range(21):
        ax.scatter(x[idx], -y[idx],
                   color=LANDMARK_COLORS[idx],
                   s=60, zorder=2, edgecolors="white", linewidths=0.5)
        if show_labels:
            ax.annotate(str(idx), (x[idx], -y[idx]),
                        fontsize=6, color="white", ha="center", va="bottom",
                        xytext=(0, 5), textcoords="offset points")

    ax.set_title(title, fontsize=10, color="white", pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ════════════════════════════════════════════════════════════════════════════
# PLOT 1 – One representative sample per class (18-panel grid)
# ════════════════════════════════════════════════════════════════════════════

def plot_one_sample_per_class(df, normalized=True, samples_per_class=1,
                               save_path=None):
    """
    18-panel grid showing one representative hand skeleton per gesture class.

    Parameters
    ----------
    df          : raw DataFrame (label column + landmark columns)
    normalized  : if True, apply translation + scale normalization
    save_path   : optional file path to save the figure (e.g. "gestures.png")
    """
    classes = sorted(df["label"].unique())
    n_classes = len(classes)
    cols = 6
    rows = int(np.ceil(n_classes / cols))

    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 2.8, rows * 3.2),
                              facecolor="#0f0f1a")
    fig.suptitle("Hand Gesture Classes — Landmark Skeleton (1 sample per class)",
                 fontsize=14, color="white", y=1.01)

    axes_flat = axes.flatten()

    for idx, gesture in enumerate(classes):
        ax = axes_flat[idx]
        sample = df[df["label"] == gesture].iloc[0]

        x_raw = np.array([sample[f"x{i+1}"] for i in range(21)])
        y_raw = np.array([sample[f"y{i+1}"] for i in range(21)])

        if normalized:
            x, y = normalize_landmarks(x_raw, y_raw)
        else:
            x, y = x_raw, y_raw

        draw_hand(ax, x, y,
                  title=gesture.replace("_", " ").title(),
                  show_labels=False)

    # Hide any unused subplots
    for j in range(idx + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Legend
    legend_items = [
        mpatches.Patch(color=FINGER_COLORS["wrist"],  label="Wrist"),
        mpatches.Patch(color=FINGER_COLORS["thumb"],  label="Thumb"),
        mpatches.Patch(color=FINGER_COLORS["index"],  label="Index"),
        mpatches.Patch(color=FINGER_COLORS["middle"], label="Middle"),
        mpatches.Patch(color=FINGER_COLORS["ring"],   label="Ring"),
        mpatches.Patch(color=FINGER_COLORS["pinky"],  label="Pinky"),
    ]
    fig.legend(handles=legend_items, loc="lower center",
               ncol=6, fontsize=9, facecolor="#1a1a2e",
               labelcolor="white", framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show()
    print(f"✓ Plotted {n_classes} gesture classes.")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 2 – Overlay of N samples per class (shows intra-class variation)
# ════════════════════════════════════════════════════════════════════════════

def plot_overlay_per_class(df, n_samples=30, normalized=True, save_path=None):
    """
    18-panel grid where each subplot overlays N samples for that class.
    Shows how consistent (or variable) each gesture is.

    Parameters
    ----------
    n_samples  : number of samples to overlay per class (default 30)
    """
    classes = sorted(df["label"].unique())
    n_classes = len(classes)
    cols = 6
    rows = int(np.ceil(n_classes / cols))

    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 2.8, rows * 3.2),
                              facecolor="#0f0f1a")
    fig.suptitle(f"Hand Gesture Intra-Class Variation  ({n_samples} samples overlaid per class)",
                 fontsize=13, color="white", y=1.01)

    axes_flat = axes.flatten()

    for idx, gesture in enumerate(classes):
        ax = axes_flat[idx]
        ax.set_aspect("equal")
        ax.set_facecolor("#1a1a2e")
        ax.set_title(gesture.replace("_", " ").title(),
                     fontsize=10, color="white", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        subset = df[df["label"] == gesture].sample(
            min(n_samples, len(df[df["label"] == gesture])),
            random_state=42
        )

        for _, row in subset.iterrows():
            x_raw = np.array([row[f"x{i+1}"] for i in range(21)])
            y_raw = np.array([row[f"y{i+1}"] for i in range(21)])

            if normalized:
                x, y = normalize_landmarks(x_raw, y_raw)
            else:
                x, y = x_raw, y_raw

            # Draw connections only (light alpha for overlay)
            for (i, j), color in CONNECTION_COLORS.items():
                ax.plot([x[i], x[j]], [-y[i], -y[j]],
                        color=color, linewidth=1, alpha=0.15, zorder=1)

            # Landmark dots
            for lm_idx in range(21):
                ax.scatter(x[lm_idx], -y[lm_idx],
                           color=LANDMARK_COLORS[lm_idx],
                           s=8, zorder=2, alpha=0.3)

    for j in range(idx + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show()
    print(f"✓ Overlaid {n_samples} samples for each of {n_classes} classes.")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 3 – Annotated single gesture (landmark index labels)
# ════════════════════════════════════════════════════════════════════════════

def plot_annotated_sample(df, gesture_name, normalized=True, save_path=None):
    """
    Detailed annotated view of one gesture: landmark indices + names.

    Parameters
    ----------
    gesture_name : e.g. "call", "peace", "fist"
    """
    sample = df[df["label"] == gesture_name].iloc[0]

    x_raw = np.array([sample[f"x{i+1}"] for i in range(21)])
    y_raw = np.array([sample[f"y{i+1}"] for i in range(21)])

    if normalized:
        x, y = normalize_landmarks(x_raw, y_raw)
    else:
        x, y = x_raw, y_raw

    fig, ax = plt.subplots(figsize=(7, 8), facecolor="#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    ax.set_title(f"Landmark Anatomy — '{gesture_name.replace('_', ' ').title()}'",
                 fontsize=14, color="white", pad=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Connections
    for (i, j), color in CONNECTION_COLORS.items():
        ax.plot([x[i], x[j]], [-y[i], -y[j]],
                color=color, linewidth=2.5, alpha=0.8, zorder=1)

    # Dots + index labels
    for lm_idx in range(21):
        ax.scatter(x[lm_idx], -y[lm_idx],
                   color=LANDMARK_COLORS[lm_idx],
                   s=120, zorder=3, edgecolors="white", linewidths=1)
        ax.annotate(
            f" {lm_idx}",
            (x[lm_idx], -y[lm_idx]),
            fontsize=8, color="white", fontweight="bold",
            xytext=(6, 4), textcoords="offset points"
        )

    # Landmark name legend (right side)
    legend_text = "\n".join(
        [f"{i:2d}  {name}" for i, name in enumerate(LANDMARK_NAMES)]
    )
    ax.text(1.02, 0.98, legend_text,
            transform=ax.transAxes,
            fontsize=7.5, color="white", verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a2a3e",
                      edgecolor="#555", alpha=0.9))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show()
    print(f"✓ Annotated skeleton for gesture: '{gesture_name}'")


# ════════════════════════════════════════════════════════════════════════════
# USAGE 
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Load data (adjust path for your environment) ─────────────────────
    df = pd.read_csv("hand_landmarks_data.csv")
    print(f"Loaded: {df.shape[0]} samples, {df['label'].nunique()} classes")
    print("Classes:", sorted(df["label"].unique()))

    # ── Plot 1: One skeleton per class ───────────────────────────────────
    plot_one_sample_per_class(df, normalized=True,
                               save_path="gesture_skeletons.png")

    # ── Plot 2: Overlay 30 samples per class (variation view) ────────────
    plot_overlay_per_class(df, n_samples=30, normalized=True,
                            save_path="gesture_variation.png")

    # ── Plot 3: Annotated single gesture ─────────────────────────────────
    plot_annotated_sample(df, gesture_name="call", normalized=True,
                           save_path="gesture_annotated.png")
