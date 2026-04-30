import os, glob, json
import cv2
from tqdm import tqdm
from effects.fog import add_fog
from effects.rain import add_rain
from effects.presets import PRESETS

IN_DIR = "data/clean/camera/front"
OUT_ROOT = "results/corrupted"
MAN_ROOT = "manifests"

def process(preset_name, max_images=None):
    p = PRESETS[preset_name]
    out_dir = f"{OUT_ROOT}/{preset_name}/camera/front"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(MAN_ROOT, exist_ok=True)

    man_path = f"{MAN_ROOT}/{preset_name}.jsonl"
    image_paths = sorted(glob.glob(f"{IN_DIR}/*"))

    if max_images is not None:
        image_paths = image_paths[:max_images]

    with open(man_path, "w") as mf:
        for ip in tqdm(image_paths, desc=f"{preset_name}"):
            img = cv2.imread(ip)
            if img is None:
                print(f"Warning: could not read {ip}")
                continue

            foggy = add_fog(img, beta=p["beta"], A=p["A"])
            rainy = add_rain(
                foggy,
                density=p["density"],
                angle=p["angle"],
                length_frac=p["length_frac"],
                width=p["width"],
                alpha=p["alpha"],
            )

            fn = os.path.basename(ip)
            cv2.imwrite(os.path.join(out_dir, fn), rainy)
            mf.write(json.dumps({"filename": fn, **p}) + "\n")

if __name__ == "__main__":
    for name in ["light", "moderate", "heavy"]:
        process(name, max_images=20)   # start small, then increase