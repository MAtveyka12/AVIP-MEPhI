import cv2
import numpy as np
import os

input_folder = 'pictures_src'
output_folder = 'pictures_results'

os.makedirs(output_folder, exist_ok=True)

def process_image(image_path, output_folder):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Не удалось загрузить изображение {image_path}.")
        return

    # Структурирующий элемент — кольцо (окно 3x3)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    dilated_image = cv2.dilate(image, kernel, iterations=1)
    diff_image = cv2.absdiff(image, dilated_image)

    thresh_value = 127
    max_value = 255
    _, binary_image = cv2.threshold(image, thresh_value, max_value, cv2.THRESH_BINARY)

    base_name = os.path.basename(image_path)
    filtered_path = os.path.join(output_folder, f"dilated_{base_name}")
    diff_path = os.path.join(output_folder, f"diff_{base_name}")
    binary_path = os.path.join(output_folder, f"binary_{base_name}")

    cv2.imwrite(filtered_path, dilated_image)
    cv2.imwrite(diff_path, diff_image)
    cv2.imwrite(binary_path, binary_image)

    print(f"Изображение {base_name} обработано и сохранено в {output_folder}.")


for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    if os.path.isfile(image_path):
        process_image(image_path, output_folder)
