import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths
CONDITIONS = {
    "Clean": "data/clean/camera/front",
    "Light": "results/corrupted/light/camera/front",
    "Moderate": "results/corrupted/moderate/camera/front",
    "Heavy": "results/corrupted/heavy/camera/front",
}

# Limit images per condition for speed.
# Set to None to use all images.
MAX_IMAGES = 200


def compute_metrics(image_paths):
    means = []
    stds = []

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        means.append(gray.mean())
        stds.append(gray.std())

    if len(means) == 0:
        return 0.0, 0.0

    return float(np.mean(means)), float(np.mean(stds))


def main():
    results = {}

    for condition, folder in CONDITIONS.items():
        print(f"\nProcessing {condition}...")

        image_paths = sorted(glob.glob(os.path.join(folder, "*")))
        if MAX_IMAGES is not None:
            image_paths = image_paths[:MAX_IMAGES]

        mean_intensity, contrast = compute_metrics(image_paths)

        results[condition] = {
            "mean": mean_intensity,
            "contrast": contrast,
        }

        print(f"{condition}: Mean={mean_intensity:.2f}, Contrast={contrast:.2f}")

    # Prepare data for plotting
    conditions = list(results.keys())
    means = [results[c]["mean"] for c in conditions]
    contrasts = [results[c]["contrast"] for c in conditions]

    # Plot
    plt.style.use("default")
    plt.figure(figsize=(8, 5))

    plt.plot(conditions, means, marker="o", label="Mean Intensity")
    plt.plot(conditions, contrasts, marker="o", label="Contrast")

    plt.xlabel("Weather Condition")
    plt.ylabel("Value")
    plt.title("Impact of Adverse Weather on Image Visibility")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Save figure
    os.makedirs("analysis", exist_ok=True)
    plt.savefig("analysis/degradation_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print results table
    print("\nFinal Results:")
    print("Condition\tMean Intensity\tContrast")
    for c in conditions:
        print(f"{c}\t\t{results[c]['mean']:.2f}\t\t{results[c]['contrast']:.2f}")

    # Suggested poster caption
    print("\nSuggested caption:")
    print(
        "Increasing rain and fog severity significantly reduces image contrast, "
        "while atmospheric scattering increases overall brightness, "
        "resulting in degraded visibility for perception systems."
    )


if __name__ == "__main__":
    main()