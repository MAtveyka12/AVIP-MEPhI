import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

# Параметры
ALPHABET = list("аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя")
PHRASE_GT = "мен сені мәңгі сүйемін"
SIZE = (64, 64)

ALPHABET_DIR = Path("alphabet")
SRC_PATH = Path("pictures_src/phrase1.bmp")
DST_DIR = Path("pictures_results")
os.makedirs(DST_DIR, exist_ok=True)

def to_binary(img_or_path) -> np.ndarray:
    """Бинаризация изображения (1 — черный)."""
    img = Image.open(img_or_path).convert("L")
    return (np.array(img) < 128).astype(np.uint8)

def normalize_bin(arr: np.ndarray, size: tuple[int, int] = SIZE) -> np.ndarray:
    """Центрирование и масштабирование изображения символа."""
    ys, xs = np.nonzero(arr)
    if ys.size == 0:
        return np.zeros(size, dtype=np.uint8)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    crop = arr[y0:y1+1, x0:x1+1]

    h, w = crop.shape
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=np.uint8)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = crop

    img = Image.fromarray(canvas * 255)
    img = img.resize(size, Image.NEAREST)
    return (np.array(img) < 128).astype(np.uint8)

def segment_by_profiles(bin_img: np.ndarray, empty_thresh: int = 1):
    """Сегментация строки на символы."""
    h, w = bin_img.shape
    vert = bin_img.sum(axis=0)
    splits, in_char = [], False

    for x, v in enumerate(vert):
        if not in_char and v > empty_thresh:
            in_char, x0 = True, x
        elif in_char and v <= empty_thresh:
            splits.append((x0, x - 1))
            in_char = False
    if in_char:
        splits.append((x0, w - 1))

    boxes = []
    for x0, x1 in splits:
        slice_ = bin_img[:, x0:x1+1]
        horiz = slice_.sum(axis=1)
        ys = np.where(horiz > empty_thresh)[0]
        if ys.size:
            boxes.append((x0, ys[0], x1, ys[-1]))
    return boxes

def extract_features(arr: np.ndarray) -> np.ndarray:
    """Извлекает признаки: масса, центр тяжести, осевые моменты."""
    ys, xs = np.nonzero(arr)
    if ys.size == 0:
        return np.zeros(6)
    mass = len(xs)
    x_c = xs.mean()
    y_c = ys.mean()
    x_m = ((xs - x_c)**2).mean()
    y_m = ((ys - y_c)**2).mean()
    xy_m = ((xs - x_c) * (ys - y_c)).mean()
    return np.array([mass, x_c, y_c, x_m, y_m, xy_m])

def load_templates():
    feats, labels = [], []
    for ch in ALPHABET:
        path = ALPHABET_DIR / f"{ch}.bmp"
        arr = normalize_bin(to_binary(path))
        feat = extract_features(arr)
        feats.append(feat)
        labels.append(ch)
    feats = np.array(feats)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8  # нормализация
    return feats, labels

def recognise_image(path: Path, template_feats: np.ndarray, labels: list[str]):
    bin_img = to_binary(path)
    boxes = segment_by_profiles(bin_img)
    boxes.sort(key=lambda b: b[0])
    predictions, all_hypotheses = [], []

    for x0, y0, x1, y1 in boxes:
        sub = bin_img[y0:y1+1, x0:x1+1]
        arr = normalize_bin(sub)
        feat = extract_features(arr)
        feat /= np.linalg.norm(feat) + 1e-8

        dists = np.linalg.norm(template_feats - feat, axis=1)
        similarities = 1 / (1 + dists)
        top_indices = similarities.argsort()[::-1]
        top_hypotheses = [(labels[i], round(float(similarities[i]), 4)) for i in top_indices]

        predictions.append(top_hypotheses[0][0])
        all_hypotheses.append(top_hypotheses)

    return predictions, all_hypotheses, boxes

def accuracy(pred: list[str], gt: str):
    gt = [c for c in gt if c != ' ']
    pred = [c for c in pred if c != ' ']
    m = max(len(pred), len(gt))
    errs = sum(1 for a, b in zip(pred, gt) if a != b)
    return errs, 100 * (1 - errs / m)

def main():
    print("[1] Загрузка шаблонов признаков…")
    template_feats, labels = load_templates()

    print("[2] Распознавание изображения…")
    preds, all_hyps, boxes = recognise_image(SRC_PATH, template_feats, labels)
    recog_str = "".join(preds)
    errs, pct = accuracy(preds, PHRASE_GT)

    print(f"\nРаспознано : {recog_str}")
    print(f"Эталон     : {PHRASE_GT}")
    print(f"Ошибок     : {errs}/{len(PHRASE_GT)} | Точность: {pct:.2f}%")

    print("[3] Сохранение результатов…")
    hyp_path = DST_DIR / "hypotheses1.txt"
    with hyp_path.open("w", encoding="utf-8") as f:
        for i, hyp in enumerate(all_hyps, 1):
            f.write(f"{i}: {hyp}\n")

    with open(DST_DIR / "best_prediction1.txt", "w", encoding="utf-8") as f:
        f.write(recog_str)

    img = Image.open(SRC_PATH).convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=1)
    img.save(DST_DIR / "phrase_segmented1.bmp")

    print("[✓] Готово!")

if __name__ == "__main__":
    main()
