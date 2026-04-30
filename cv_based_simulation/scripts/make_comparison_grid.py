import os
import cv2
import matplotlib.pyplot as plt

IMAGE_NAME = "0.jpg"

PATHS = {
    "Clean": "data/clean/camera/front",
    "Light": "results/corrupted/light/camera/front",
    "Moderate": "results/corrupted/moderate/camera/front",
    "Heavy": "results/corrupted/heavy/camera/front",
}

OUTPUT_DIR = "analysis"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "comparison_grid_labeled.png")


def load_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = {}
    for label, folder in PATHS.items():
        img_path = os.path.join(folder, IMAGE_NAME)
        images[label] = load_rgb(img_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Impact of Synthetic Weather Degradation on Waymo Front Camera Images", fontsize=16)

    labels = ["Clean", "Light", "Moderate", "Heavy"]

    for ax, label in zip(axes.ravel(), labels):
        ax.imshow(images[label])
        ax.set_title(label, fontsize=13, pad=8)
        ax.axis("off")

        ax.text(
            0.02, 0.95, label,
            transform=ax.transAxes,
            fontsize=11,
            color="white",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.6)
        )

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved comparison grid to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()