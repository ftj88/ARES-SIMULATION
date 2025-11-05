import numpy as np, cv2

def add_rain(img,
             density=0.02,         # fraction of pixels that spawn streaks
             angle=-15,            # degrees, negative tilts right
             length_frac=0.03,     # streak length as fraction of min(h,w)
             width=2,              # streak thickness in pixels
             alpha=0.6,            # blend strength
             brightness=240,       # streak brightness cap (0-255)
             seed=1337):
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)

    # 1) Sparse impulse map (where streaks originate)
    n = int(h * w * density)
    impulses = np.zeros((h, w), np.float32)
    xs = rng.integers(0, w, n)
    ys = rng.integers(0, h, n)
    impulses[ys, xs] = 1.0

    # 2) Motion blur kernel (rotated line)
    L = max(5, int(min(h, w) * length_frac))   # e.g., 3000px wide → ~90px streaks @ 0.03
    k = np.zeros((L, L), np.float32)
    cv2.line(k, (0, L // 2), (L - 1, L // 2), 1.0, width)  # horizontal line

    M = cv2.getRotationMatrix2D((L / 2, L / 2), angle, 1.0)
    k = cv2.warpAffine(k, M, (L, L))
    s = k.sum() + 1e-6
    k /= s

    # 3) Convolve impulses → long streaks
    streaks = cv2.filter2D(impulses, -1, k)

    # 4) Normalize, lightly blur, brighten
    streaks = cv2.normalize(streaks, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    streaks = cv2.GaussianBlur(streaks, (3, 3), 0)
    streaks_col = cv2.cvtColor(streaks, cv2.COLOR_GRAY2BGR)
    streaks_col = np.minimum(streaks_col.astype(np.float32) * (brightness / 255.0), 255).astype(np.uint8)

    # 5) Blend over image
    out = cv2.addWeighted(img, 1.0, streaks_col, alpha, 0)
    return out
