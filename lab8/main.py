import os
import glob
import colorsys
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

SRC_DIR = "pictures_src"
DST_DIR = "pictures_results"
os.makedirs(DST_DIR, exist_ok=True)

D = 1
G = 16

def rgb_to_hls_arr(img: np.ndarray):
    hls = np.zeros_like(img, dtype=float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r, g, b = img[i, j] / 255.0
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            hls[i, j] = (h, l, s)
    return hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]

def hls_to_rgb_arr(H, L, S):
    out = np.zeros((H.shape[0], H.shape[1], 3), dtype=np.uint8)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            r, g, b = colorsys.hls_to_rgb(H[i, j], L[i, j], S[i, j])
            out[i, j] = (int(r * 255), int(g * 255), int(b * 255))
    return out

def equalize_histogram(L: np.ndarray):
    flat = (L * 255).astype(int).ravel()
    hist = np.bincount(flat, minlength=256)
    cdf = hist.cumsum()
    cdf_norm = (cdf - cdf.min()) / (cdf.max() - cdf.min())
    L_eq = cdf_norm[flat].reshape(L.shape)
    return L_eq

def compute_ngldm(L_gray: np.ndarray, d=1, G=16):
    bins = np.linspace(0, 256, G + 1)
    Q = np.digitize(L_gray, bins) - 1  # 0..G-1
    h, w = Q.shape
    offsets = [(-d, -d), (-d, 0), (-d, d), (0, -d), (0, d), (d, -d), (d, 0), (d, d)]
    max_dep = len(offsets)
    S = np.zeros((G, max_dep + 1), dtype=int)

    for i in range(h):
        for j in range(w):
            g = Q[i, j]
            cnt = 0
            for dy, dx in offsets:
                y, x = i + dy, j + dx
                if 0 <= y < h and 0 <= x < w and Q[y, x] == g:
                    cnt += 1
            S[g, cnt] += 1
    return S

def compute_features(S: np.ndarray):
    P = S.sum()
    if P == 0:
        return 0.0, 0.0, 0.0
    i_idx = np.arange(S.shape[0])[:, None]
    j_idx = np.arange(S.shape[1])[None, :]

    NN = (j_idx * S).sum() / P
    SM = (S[:, 1:] / (j_idx[:, 1:] + 1)**2).sum() / P  # избегаем деления на 0
    probs = S[S > 0] / P
    ENT = -np.sum(probs * np.log2(probs))

    return NN, SM, ENT

def process_image(path: str):
    name = Path(path).stem
    img = Image.open(path).convert("RGB")
    arr = np.array(img)

    H, L, S = rgb_to_hls_arr(arr)
    L_gray = (L * 255).astype(np.uint8)

    S_ngl = compute_ngldm(L_gray, d=D, G=G)
    NN0, SM0, ENT0 = compute_features(S_ngl)

    L_eq = equalize_histogram(L)
    L_eq_gray = (L_eq * 255).astype(np.uint8)
    arr_eq = hls_to_rgb_arr(H, L_eq, S)

    S_ngl_eq = compute_ngldm(L_eq_gray, d=D, G=G)
    NN1, SM1, ENT1 = compute_features(S_ngl_eq)

    Image.fromarray(arr).save(os.path.join(DST_DIR, f"{name}_orig.png"))
    Image.fromarray(L_gray).save(os.path.join(DST_DIR, f"{name}_gray.png"))
    Image.fromarray(L_eq_gray).save(os.path.join(DST_DIR, f"{name}_gray_eq.png"))
    Image.fromarray(arr_eq).save(os.path.join(DST_DIR, f"{name}_color_eq.png"))

    plt.figure(); plt.hist(L_gray.ravel(), bins=256); plt.title("L до");
    plt.savefig(os.path.join(DST_DIR, f"{name}_hist_before.png")); plt.close()
    plt.figure(); plt.hist(L_eq_gray.ravel(), bins=256); plt.title("L после");
    plt.savefig(os.path.join(DST_DIR, f"{name}_hist_after.png")); plt.close()

    plt.figure(figsize=(10, 6))
    plt.imshow(np.log1p(S_ngl), cmap='viridis', aspect='auto')
    plt.colorbar(label='Log(count)')
    plt.title("NGLDM Matrix (log scale)")
    plt.xlabel("Dependence count")
    plt.ylabel("Gray level")
    plt.tight_layout()
    plt.savefig(os.path.join(DST_DIR, f"{name}_ngldm.png"), dpi=100)
    plt.close()

    with open(os.path.join(DST_DIR, f"{name}_features.txt"), "w", encoding="utf-8") as f:
        f.write(f"NN до: {NN0:.4f}\nSM до: {SM0:.4f}\nENT до: {ENT0:.4f}\n")
        f.write(f"NN после: {NN1:.4f}\nSM после: {SM1:.4f}\nENT после: {ENT1:.4f}\n")

    print(f"[✓] {name}: NN {NN0:.2f}->{NN1:.2f}, SM {SM0:.4f}->{SM1:.4f}, ENT {ENT0:.2f}->{ENT1:.2f}")

def main():
    files = glob.glob(os.path.join(SRC_DIR, "*.*"))
    for path in files:
        try:
            process_image(path)
        except Exception as e:
            print(f"[!] {path} — ошибка: {e}")

if __name__ == "__main__":
    main()
