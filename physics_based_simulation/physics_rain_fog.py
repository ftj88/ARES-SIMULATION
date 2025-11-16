import cv2
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
import random
from PIL import Image

# ----------------------------
# Paper-faithful simplified VRS
# ----------------------------

def compute_beta_ext(visibility_m):
    # Koschmieder approximation (Eq. 6)
    return -np.log(0.05) / float(max(0.1, visibility_m))

def add_koschmieder_fog(img_rgb,
                        depth_map=None,
                        visibility_m=200.0,
                        Ls_mode='mean',    # 'mean' or 'brightest'
                        d_near=1.0,        # used if no depth_map: distance at bottom (meters)
                        d_far=100.0):      # used if no depth_map: distance at top (meters)
    """
    img_rgb: HxWx3 uint8 RGB
    depth_map: HxW float distances in meters (optional). If None, approximate from vertical pixel.
    visibility_m: V in meters (Koschmieder)
    Ls_mode: how to choose sky/airlight luminance: 'mean' or 'brightest'
    returns fogged image (uint8)
    """
    img = img_rgb.astype(np.float32) / 255.0
    h, w = img.shape[:2]

    # compute Ls (airlight luminance)
    if Ls_mode == 'brightest':
        # pick pixel with highest intensity (luminance) and use its color
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        max_idx = np.unravel_index(np.argmax(gray), gray.shape)
        Ls = img[max_idx]
    else:
        # mean color (works for indoor too)
        Ls = img.mean(axis=(0,1))

    # compute per-pixel distances d (meters)
    if depth_map is not None:
        d = depth_map.astype(np.float32)
    else:
        # heuristic: distance increases toward top of image (road perspective)
        # map y in [0,h-1] to distance d(y) in [d_far,d_near]
        ys = np.linspace(0, 1, h)  # 0 = top, 1 = bottom
        d_col = d_near + (d_far - d_near) * (1.0 - ys)  # top -> far, bottom -> near
        d = np.tile(d_col[:, None], (1, w))

    beta = compute_beta_ext(visibility_m)  # extinction coefficient (1/m)

    # apply Koschmieder/Beer-Lambert per color channel (Eq. 5)
    # L = L0 * exp(-beta d) + Ls * (1 - exp(-beta d))
    exp_term = np.exp(-beta * d)[:, :, np.newaxis]
    L0 = img
    fogged = L0 * exp_term + Ls[np.newaxis, np.newaxis, :] * (1.0 - exp_term)

    fogged = np.clip(fogged * 255.0, 0, 255).astype(np.uint8)
    return fogged

# ----------------------------
# Rain streak simulation
# ----------------------------

def terminal_velocity_mps(D_mm):
    # Atlas & Ulbrich Eq. (7): v(D) = 3.78 * D^0.67  (D in mm, v in m/s)
    return 3.78 * (D_mm ** 0.67)

def generate_rain_noise_filter(img_shape,
                               fov_deg=60.0,
                               exposure_time_s=0.01,
                               num_drops=2000,
                               drop_diameter_mm_range=(0.2, 4.0),
                               near_z=0.5,
                               far_z=10.0,
                               downsample=1):
    """
    Generates a camera noise filter C_cam in [0,1] for rain streaks.
    - Projects sampled 3D raindrops into image plane at two time steps and draws streaks.
    - img_shape: (H,W)
    - returns C_cam (HxW) float32
    """
    H, W = img_shape
    # use a working resolution (optionally downsample to speed up)
    Hw, Ww = int(H / downsample), int(W / downsample)

    # focal length approximation from FOV
    fov_rad = np.deg2rad(fov_deg)
    fx = (Ww / 2.0) / np.tan(fov_rad / 2.0)
    fy = fx
    cx = Ww / 2.0
    cy = Hw / 2.0

    # sample drops in camera coordinates (x right, y down, z forward)
    # sample z in [near_z, far_z] (meters)
    zs = np.random.uniform(low=near_z, high=far_z, size=num_drops)
    # sample horizontal x and vertical y positions within a frustum width proportional to z
    # approximate x in [-z*tan(hfov/2), z*tan(hfov/2)]
    x_range = np.tan(fov_rad / 2.0) * zs
    xs = np.random.uniform(-1.0, 1.0, size=num_drops) * x_range
    # vertical world coordinate: choose y so that projection falls inside sensor; approximate by same range
    ys = np.random.uniform(-0.5, 0.5, size=num_drops) * x_range  # modest vertical spread

    # sample diameters (mm) using distribution skewed toward small drops
    D_min, D_max = drop_diameter_mm_range
    Ds = (np.random.rand(num_drops) ** 1.5) * (D_max - D_min) + D_min
    velocities = terminal_velocity_mps(Ds)  # m/s (vertical fall speed)

    # compute positional change in world for exposure_time (assume drops fall along +y in camera coords)
    dt = exposure_time_s
    dy = velocities * dt  # meters fall (positive)

    # create blank mask
    mask = np.zeros((Hw, Ww), dtype=np.float32)

    # helper to project a 3D point to pixel coords
    def project(x, y, z):
        # pinhole: u = fx * (x / z) + cx, v = fy * (y / z) + cy
        u = fx * (x / z) + cx
        v = fy * (y / z) + cy
        return u, v

    # draw each drop as small circle at both t0 and t1 and connect them with a line
    for i in range(num_drops):
        x0, y0, z0 = xs[i], ys[i], zs[i]
        x1, y1, z1 = x0, y0 + dy[i], z0  # fall in +y direction; z unchanged for simplicity

        u0, v0 = project(x0, y0, z0)
        u1, v1 = project(x1, y1, z1)

        # skip if both projections are outside image
        if not (-5 <= u0 <= Ww+5 and -5 <= v0 <= Hw+5) and not (-5 <= u1 <= Ww+5 and -5 <= v1 <= Hw+5):
            continue

        # approximate pixel radius from physical diameter and distance: radius_px ~ (f * (D/1000)) / z
        radius0 = max(1, int((fx * (Ds[i]/1000.0)) / max(0.001, z0)))
        radius1 = max(1, int((fx * (Ds[i]/1000.0)) / max(0.001, z1)))
        # draw on mask (use integer coords)
        pt0 = (int(round(u0)), int(round(v0)))
        pt1 = (int(round(u1)), int(round(v1)))

        cv2.circle(mask, pt0, radius0, 1.0, -1)  # fill
        cv2.circle(mask, pt1, radius1, 1.0, -1)
        # draw connecting line (streak)
        cv2.line(mask, pt0, pt1, 1.0, max(1, int((radius0 + radius1) / 2)))

    # apply a little gaussian blur to smooth
    mask = cv2.GaussianBlur(mask, (7,7), sigmaX=3)
    # normalize to [0,1]
    if mask.max() > 0:
        mask = mask / mask.max()
    # upsample to original resolution if downsampled
    if downsample != 1:
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)

    return mask.astype(np.float32)

def apply_rain_to_image(img_rgb,
                        Ccam,
                        muB=None,
                        RC=None):
    """
    Blend rain streaks into image following Eq. 8 & 9.
    - Ccam: HxW in [0,1]
    - muB: mean brightness (grayscale) in [0,1]; if None, compute from img
    - RC: rain color (1x3), default white or mean bright color
    """
    img = img_rgb.astype(np.float32) / 255.0
    H, W = Ccam.shape

    # compute mean brightness muB
    if muB is None:
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        muB = float(gray.mean())

    # rain color RC default: bright (white) or mean if indoor
    if RC is None:
        # use slightly off-white (paper suggests bright / environment reflection)
        RC = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    else:
        RC = np.array(RC, dtype=np.float32)

    # compute alpha according to Eq. (9): alpha = (1 - muB) * Ccam
    alpha = (1.0 - muB) * Ccam
    alpha_3c = alpha[:, :, np.newaxis]

    VC = alpha_3c * RC[np.newaxis, np.newaxis, :] + (1.0 - alpha_3c) * img
    out = np.clip(VC * 255.0, 0, 255).astype(np.uint8)
    return out, alpha

# ----------------------------
# Combined function & plotting
# ----------------------------

def apply_paper_faithful_vrs(image_path_or_array,
                             depth_map=None,
                             visibility_m=200.0,
                             exposure_time_s=0.01,
                             rain_num_drops=2000,
                             rain_drop_range=(0.2, 4.0),
                             fov_deg=60.0,
                             near_z=0.5,
                             far_z=10.0,
                             downsample_rain=2,
                             Ls_mode='mean'):
    """
    Top-level function:
    - image_path_or_array: path or HxWx3 RGB numpy array
    - depth_map: optional HxW float distance map (meters)
    """
    # load image
    if isinstance(image_path_or_array, str):
        img_bgr = cv2.imread(image_path_or_array)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image {image_path_or_array}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img = image_path_or_array.copy()

    H, W = img.shape[:2]

    # 1) Fog using Koschmieder / luminance
    fogged = add_koschmieder_fog(img,
                                 depth_map=depth_map,
                                 visibility_m=visibility_m,
                                 Ls_mode=Ls_mode,
                                 d_near=1.0,
                                 d_far=100.0)

    # 2) Rain camera noise filter C_cam (near field <= 10 m)
    Ccam = generate_rain_noise_filter((H, W),
                                      fov_deg=fov_deg,
                                      exposure_time_s=exposure_time_s,
                                      num_drops=rain_num_drops,
                                      drop_diameter_mm_range=rain_drop_range,
                                      near_z=near_z,
                                      far_z=far_z,
                                      downsample=downsample_rain)

    # 3) Blend rain into fogged image using alpha model
    # compute muB from fogged image luminance
    gray_fogged = cv2.cvtColor(fogged, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    muB = float(gray_fogged.mean())
    rainy_img, alpha_mask = apply_rain_to_image(fogged, Ccam, muB=muB, RC=None)

    # 4) Optionally create a visualization of the noise filter
    cmap = plt.get_cmap('viridis')
    Cvis = (cmap(Ccam)[:, :, :3] * 255).astype(np.uint8)

    # 5) Plot original, fogged, rain filter, and final
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(img); axes[0].set_title('Original'); axes[0].axis('off')
    axes[1].imshow(fogged); axes[1].set_title(f'Fogged (V={visibility_m} m)'); axes[1].axis('off')
    axes[2].imshow(Cvis); axes[2].set_title('Camera noise filter C_cam'); axes[2].axis('off')
    axes[3].imshow(rainy_img); axes[3].set_title('Fog + Rain (final)'); axes[3].axis('off')
    plt.tight_layout()
    plt.show()

    return rainy_img, fogged, Ccam, alpha_mask

def get_random_nuscenes_image(nuscenes_root, version="v1.0-mini", camera="CAM_FRONT"):
    nusc = NuScenes(version=version, dataroot=nuscenes_root, verbose=False)

    # Filter samples that contain the requested camera
    sample_tokens = [s['token'] for s in nusc.sample]
    random_sample = random.choice(sample_tokens)

    sample = nusc.get('sample', random_sample)
    cam_data = nusc.get('sample_data', sample['data'][camera])
    img_path = nusc.get_sample_data_path(cam_data['token'])

    return Image.open(img_path), img_path

def main():
    # Load a random nuScenes image
    image, img_path = get_random_nuscenes_image("/data/sets/nuscenes")
    print(f"Loaded random nuScenes image: {img_path}")

    depth_map = None  # optional depth map

    # Parameters to explore
    fog_visibilities = [300.0, 200.0, 100.0, 50.0]
    ls_modes = ['mean', 'brightest']

    # Rain parameters (kept constant)
    rain_params = dict(
        exposure_time_s=0.02,
        rain_num_drops=100,
        rain_drop_range=(0.3, 5.0),
        fov_deg=60.0,
        near_z=0.5,
        far_z=10.0,
        downsample_rain=3,
    )

    results = []  # store (ls_mode, vis, final_img)
    
    # Generate one consistent rain filter for all
    print("\nGenerating base rain filter...")
    temp_img = np.array(image)
    H, W = temp_img.shape[:2]
    Ccam = generate_rain_noise_filter(
        (H, W),
        fov_deg=rain_params['fov_deg'],
        exposure_time_s=rain_params['exposure_time_s'],
        num_drops=rain_params['rain_num_drops'],
        drop_diameter_mm_range=rain_params['rain_drop_range'],
        near_z=rain_params['near_z'],
        far_z=rain_params['far_z'],
        downsample=rain_params['downsample_rain']
    )

    # Create a color visualization for the rain filter
    cmap = plt.get_cmap('viridis')
    Cvis = (cmap(Ccam)[:, :, :3] * 255).astype(np.uint8)

    # Generate fog + rain images
    for ls_mode in ls_modes:
        for vis in fog_visibilities:
            print(f"\n--- Generating fog+rain: visibility={vis} m, Ls_mode='{ls_mode}' ---")

            fogged_img = add_koschmieder_fog(
                np.array(image),
                depth_map=depth_map,
                visibility_m=vis,
                Ls_mode=ls_mode,
                d_near=1.0,
                d_far=100.0
            )

            gray_fogged = cv2.cvtColor(fogged_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            muB = float(gray_fogged.mean())
            rainy_img, alpha = apply_rain_to_image(fogged_img, Ccam, muB=muB, RC=None)

            results.append((ls_mode, vis, rainy_img))

    # -----------------------------------------------------
    # -----------------------------------------------------
    # Plot layout:
    #   2 rows × 5 cols
    #   Row 0: Original | Rain Filter | mean (300, 200, 100)
    #   Row 1: mean (50) | brightest (300, 200, 100, 50)
    # -----------------------------------------------------
    n_rows, n_cols = 2, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 7))

    # Row 0: Original and Rain filter
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(Cvis)
    axes[0, 1].set_title("Rain Filter (C_cam)")
    axes[0, 1].axis("off")

    # Collect mean and brightest results
    mean_results = [(vis, img) for mode, vis, img in results if mode == 'mean']
    brightest_results = [(vis, img) for mode, vis, img in results if mode == 'brightest']

    # Sort by visibility for consistent order
    mean_results.sort(key=lambda x: -x[0])
    brightest_results.sort(key=lambda x: -x[0])

    # Fill remaining columns with fog+rain results
    # Row 0 → remaining mean results (up to 3)
    # Row 1 → remaining mean (if any) + all brightest
    col = 2
    for vis, img in mean_results[:3]:
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"mean\nV={vis} m")
        axes[0, col].axis("off")
        col += 1

    # Row 1: mean (if leftover) + all brightest
    col = 0
    if len(mean_results) > 3:
        vis, img = mean_results[3]
        axes[1, col].imshow(img)
        axes[1, col].set_title(f"mean\nV={vis} m")
        axes[1, col].axis("off")
        col += 1

    for vis, img in brightest_results:
        axes[1, col].imshow(img)
        axes[1, col].set_title(f"brightest\nV={vis} m")
        axes[1, col].axis("off")
        col += 1
        if col >= n_cols:
            break

    plt.tight_layout()
    plt.show()

# ----------------------------
# Example usage:
# ----------------------------
if __name__ == '__main__':
    main()
