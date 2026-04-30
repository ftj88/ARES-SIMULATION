import os
import pandas as pd
from PIL import Image
import io
from tqdm import tqdm

input_dir = r"C:\Users\Dell\Downloads\data\raw\waymo"
output_dir = r"L:\ARES-SIMULATION\cv_based_simulation\data\clean\camera\front"

os.makedirs(output_dir, exist_ok=True)

image_count = 0
MAX_IMAGES = 10000

parquet_files = [f for f in os.listdir(input_dir) if f.endswith(".parquet")]

for file in parquet_files:
    file_path = os.path.join(input_dir, file)
    print(f"\nProcessing: {file}")

    df = pd.read_parquet(file_path)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        if image_count >= MAX_IMAGES:
            break

        if row["key.camera_name"] != 1:
            continue

        img_bytes = row["[CameraImageComponent].image"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        img.save(os.path.join(output_dir, f"{image_count}.jpg"))
        image_count += 1

    if image_count >= MAX_IMAGES:
        break

print(f"\nDone! Extracted {image_count} front images.")