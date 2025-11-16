import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image

def compute_beta_ext(visibility_m):
    return -np.log(0.05) / float(max(0.1, visibility_m))

def add_koschmieder_fog(img_rgb, depth_map=None, visibility_m=200.0, Ls_mode='mean', d_near=1.0, d_far=100.0):
    img = img_rgb.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    if Ls_mode == 'brightest':
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        max_idx = np.unravel_index(np.argmax(gray), gray.shape)
        Ls = img[max_idx]
    else:
        Ls = img.mean(axis=(0,1))
    if depth_map is not None:
        d = depth_map.astype(np.float32)
    else:
        ys = np.linspace(0, 1, h)
        d_col = d_near + (d_far - d_near) * (1.0 - ys)
        d = np.tile(d_col[:, None], (1, w))
    beta = compute_beta_ext(visibility_m)
    exp_term = np.exp(-beta * d)[:, :, np.newaxis]
    L0 = img
    fogged = L0 * exp_term + Ls[np.newaxis, np.newaxis, :] * (1.0 - exp_term)
    fogged = np.clip(fogged * 255.0, 0, 255).astype(np.uint8)
    return fogged

def terminal_velocity_mps(D_mm):
    return 4.2 * (D_mm ** 0.66)

def generate_rain_noise_filter(img_shape, fov_deg=60.0, exposure_time_s=0.004, num_drops=3500, drop_diameter_mm_range=(0.3, 3.5), near_z=0.5, far_z=20.0, downsample=1, near_layer_fraction=0.18, wind_angle_deg=10):
    H, W = img_shape
    Hw, Ww = int(H / downsample), int(W / downsample)
    fov_rad = np.deg2rad(fov_deg)
    fx = (Ww / 2.0) / np.tan(fov_rad / 2.0)
    fy = fx
    cx = Ww / 2.0
    cy = Hw / 2.0
    num_near = max(1, int(num_drops * near_layer_fraction))
    num_far = max(1, num_drops - num_near)
    D_min, D_max = drop_diameter_mm_range
    Ds_far = (np.random.rand(num_far) ** 2.0) * (D_max - D_min) + D_min
    zs_far = np.random.uniform(low=near_z, high=far_z, size=num_far)
    x_world_far = np.random.uniform(-np.tan(fov_rad/2)*zs_far, np.tan(fov_rad/2)*zs_far)
    y_world_far = np.random.uniform(-0.5*np.tan(fov_rad/2)*zs_far, 0.5*np.tan(fov_rad/2)*zs_far)
    velocities = terminal_velocity_mps(Ds_far)
    dt = exposure_time_s
    dy_world = velocities * dt
    dx_world = dy_world * np.tan(np.deg2rad(wind_angle_deg))
    mask_far = np.zeros((Hw, Ww), dtype=np.float32)
    def project_world(x, y, z):
        u = fx * (x / z) + cx
        v = fy * (y / z) + cy
        return u, v
    for i in range(num_far):
        x0w = x_world_far[i]
        y0w = y_world_far[i]
        z0 = zs_far[i]
        x1w = x0w + dx_world[i]
        y1w = y0w + dy_world[i]
        u0, v0 = project_world(x0w, y0w, z0)
        u1, v1 = project_world(x1w, y1w, z0)
        if not ((-10 <= u0 <= Ww+10 and -10 <= v0 <= Hw+10) or (-10 <= u1 <= Ww+10 and -10 <= v1 <= Hw+10)):
            continue
        radius0 = max(1, int((fx * (Ds_far[i]/1000.0)) / max(0.001, z0)))
        pt0 = (int(round(u0)), int(round(v0)))
        pt1 = (int(round(u1)), int(round(v1)))
        cv2.line(mask_far, pt0, pt1, 1.0, max(1, int(radius0)))
        cv2.circle(mask_far, pt0, radius0, 1.0, -1)
        cv2.circle(mask_far, pt1, radius0, 1.0, -1)
    mask_far = cv2.GaussianBlur(mask_far, (5,5), sigmaX=2)
    mask_far = mask_far / mask_far.max() if mask_far.max()>0 else mask_far
    Ds_near = (np.random.rand(num_near) ** 1.5) * (D_max - D_min) + D_min
    xs_near = np.random.uniform(0, Ww, num_near)
    ys_near = np.random.uniform(0, Hw, num_near)
    stick_prob = np.random.uniform(0.6, 1.0, num_near)
    mask_near = np.zeros((Hw, Ww), dtype=np.float32)
    for i in range(num_near):
        r = max(2, int(Ds_near[i] * (fx/900.0))) 
        alpha = 1.0 if stick_prob[i] > 0.65 else 0.8
        cv2.circle(mask_near, (int(xs_near[i]), int(ys_near[i])), r, float(alpha), -1)
    mask_near = cv2.GaussianBlur(mask_near, (15,15), sigmaX=6)
    if downsample != 1:
        mask_far = cv2.resize(mask_far, (W, H), interpolation=cv2.INTER_LINEAR)
        mask_near = cv2.resize(mask_near, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        mask_far = cv2.resize(mask_far, (W, H), interpolation=cv2.INTER_LINEAR)
        mask_near = cv2.resize(mask_near, (W, H), interpolation=cv2.INTER_LINEAR)
    zs_norm = np.clip((np.linspace(1.0, 0.0, H)[:,None]), 0, 1)
    distance_atten = zs_norm
    mask_far = mask_far * (0.7 + 0.4 * distance_atten)
    combined = np.clip(1.2 * mask_far + 1.6 * mask_near, 0, 1)
    combined = cv2.GaussianBlur((combined*255).astype(np.uint8), (5,5), sigmaX=1).astype(np.float32)/255.0
    return combined.astype(np.float32)

def apply_rain_to_image(img_rgb, Ccam, muB=None, RC=None):
    img = img_rgb.astype(np.float32) / 255.0
    H, W = Ccam.shape
    if muB is None:
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        muB = float(gray.mean())
    if RC is None:
        RC = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    else:
        RC = np.array(RC, dtype=np.float32)
    alpha = (1.0 - muB) * Ccam
    alpha_3c = alpha[:, :, np.newaxis]
    VC = alpha_3c * RC[np.newaxis, np.newaxis, :] + (1.0 - alpha_3c) * img
    out = np.clip(VC * 255.0, 0, 255).astype(np.uint8)
    return out, alpha

def apply_paper_faithful_vrs(image_path_or_array, depth_map=None, visibility_m=200.0, exposure_time_s=0.004, rain_num_drops=3500, rain_drop_range=(0.3, 3.5), fov_deg=60.0, near_z=0.5, far_z=20.0, downsample_rain=1, Ls_mode='mean'):
    if isinstance(image_path_or_array, str):
        img_bgr = cv2.imread(image_path_or_array)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image {image_path_or_array}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img = image_path_or_array.copy()
    H, W = img.shape[:2]
    fogged = add_koschmieder_fog(img, depth_map=depth_map, visibility_m=visibility_m, Ls_mode=Ls_mode, d_near=1.0, d_far=100.0)
    Ccam = generate_rain_noise_filter((H, W), fov_deg=fov_deg, exposure_time_s=exposure_time_s, num_drops=rain_num_drops, drop_diameter_mm_range=rain_drop_range, near_z=near_z, far_z=far_z, downsample=downsample_rain)
    gray_fogged = cv2.cvtColor(fogged, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    muB = float(gray_fogged.mean())
    rainy_img, alpha_mask = apply_rain_to_image(fogged, Ccam, muB=muB, RC=None)
    cmap = plt.get_cmap('viridis')
    Cvis = (cmap(Ccam)[:, :, :3] * 255).astype(np.uint8)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(img); axes[0].set_title('Original'); axes[0].axis('off')
    axes[1].imshow(fogged); axes[1].set_title(f'Fogged (V={visibility_m} m)'); axes[1].axis('off')
    axes[2].imshow(Cvis); axes[2].set_title('Camera noise filter C_cam'); axes[2].axis('off')
    axes[3].imshow(rainy_img); axes[3].set_title('Fog + Rain (final)'); axes[3].axis('off')
    plt.tight_layout()
    plt.show()
    return rainy_img, fogged, Ccam, alpha_mask

def main():
    img_path = "test_image.jpg"
    depth_map = None
    out_img, fogged_img, Ccam, alpha = apply_paper_faithful_vrs(img_path, depth_map=depth_map, visibility_m=150.0, exposure_time_s=0.004, rain_num_drops=3500, rain_drop_range=(0.3, 3.5), fov_deg=60.0, near_z=0.5, far_z=20.0, downsample_rain=1, Ls_mode='mean')

if __name__ == '__main__':
    main()
