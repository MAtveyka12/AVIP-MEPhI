import os
import numpy as np
from PIL import Image, ImageDraw

SRC_PATH = "pictures_src/phrase.bmp"
DST_DIR = "pictures_results"
os.makedirs(DST_DIR, exist_ok=True)


def to_binary(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.array(img)
    return (arr < 128).astype(np.uint8)  # Текст = 1, фон = 0


def profiles(bin_img: np.ndarray):
    return bin_img.sum(axis=1), bin_img.sum(axis=0)


def segment_by_profiles(bin_img: np.ndarray, empty_thresh: int = 2):
    h, w = bin_img.shape
    vert = bin_img.sum(axis=0)
    splits, in_char = [], False

    # Улучшенный алгоритм сегментации с учетом соединенных букв
    for x, v in enumerate(vert):
        if not in_char and v > empty_thresh:
            in_char, x0 = True, x
        elif in_char and v <= empty_thresh:
            # Проверяем, не является ли это временным провалом внутри буквы
            lookahead = min(x + 3, w - 1)  # Смотрим на 3 пикселя вперед
            if vert[lookahead] > empty_thresh:
                continue  # Пропускаем временный провал
            splits.append((x0, x - 1))
            in_char = False
    if in_char:
        splits.append((x0, w - 1))

    boxes = []
    for x0, x1 in splits:
        slice_ = bin_img[:, x0:x1 + 1]
        horiz = slice_.sum(axis=1)
        ys = np.where(horiz > empty_thresh)[0]
        if ys.size:
            boxes.append((x0, ys[0], x1, ys[-1]))
    return boxes


def split_wide_boxes(boxes, bin_img, factor=1.8):  # Увеличил factor для лучшего разделения
    widths = [x1 - x0 + 1 for x0, _, x1, _ in boxes]
    if not widths:
        return boxes
    avg_w = sum(widths) / len(widths)

    out = []
    for (x0, y0, x1, y1), w in zip(boxes, widths):
        if w > avg_w * factor:
            sub = bin_img[y0:y1 + 1, x0:x1 + 1]
            vert = sub.sum(axis=0)
            margin = max(w // 5, 2)  # Увеличил margin для более точного разделения

            # Ищем самый глубокий провал в вертикальном профиле
            local = vert[margin:-margin]
            if local.size > 3:  # Минимальный размер для разделения
                min_pos = np.argmin(local)
                min_val = local[min_pos]
                mean_val = np.mean(local)

                # Разделяем только если провал достаточно глубокий
                if min_val < mean_val * 0.5:
                    cut_rel = min_pos + margin
                    cut = x0 + cut_rel
                    out += [(x0, y0, cut, y1), (cut + 1, y0, x1, y1)]
                    continue
        out.append((x0, y0, x1, y1))
    return out


def save_letter_profiles(bin_img, boxes):
    for idx, (x0, y0, x1, y1) in enumerate(boxes):
        patch = bin_img[y0:y1 + 1, x0:x1 + 1]
        char_img = Image.fromarray((1 - patch) * 255)
        bmp_name = f"letter_{idx:02d}.bmp"
        char_img.save(os.path.join(DST_DIR, bmp_name))

        h_prof, v_prof = profiles(patch)
        txt_name = f"profile_{idx:02d}.txt"
        with open(os.path.join(DST_DIR, txt_name), "w", encoding="utf-8") as f:
            f.write("horizontal:\n" + " ".join(map(str, h_prof.tolist())) + "\n")
            f.write("vertical:\n" + " ".join(map(str, v_prof.tolist())))


def draw_boxes(path: str, boxes):
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for x0, y0, x1, y1 in boxes:
        draw.rectangle([(x0, y0), (x1, y1)], outline="red", width=2)
    img.save(os.path.join(DST_DIR, "phrase_boxes.bmp"))


def main():
    bin_img = to_binary(SRC_PATH)

    h, v = profiles(bin_img)
    np.savetxt(os.path.join(DST_DIR, "horiz_profile.txt"), h, fmt="%d")
    np.savetxt(os.path.join(DST_DIR, "vert_profile.txt"), v, fmt="%d")

    boxes = segment_by_profiles(bin_img, empty_thresh=2)
    boxes = split_wide_boxes(boxes, bin_img)

    draw_boxes(SRC_PATH, boxes)
    save_letter_profiles(bin_img, boxes)

    print(f"Сегментировано символов: {len(boxes)}")
    print(f"Результаты сохранены в {DST_DIR}")


if __name__ == "__main__":
    main()