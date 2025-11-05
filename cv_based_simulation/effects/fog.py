import numpy as np
import cv2

def add_fog(img, beta=0.03, A=220):
    h, w = img.shape[:2]
    depth = cv2.GaussianBlur(np.tile(np.linspace(0,1,w),(h,1)).astype(np.float32),(0,0),45)
    t = np.exp(-beta * (1 + 2 * depth))
    out = img.astype(np.float32) * t[..., None] + A * (1 - t)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)
