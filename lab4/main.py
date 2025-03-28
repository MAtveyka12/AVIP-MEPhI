import numpy as np
import cv2
import os

# Функция для перевода изображения в оттенки серого
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

# Функция для вычисления морфологического градиента
def morphological_gradient(image):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    gradient = cv2.absdiff(dilated, image)
    return gradient

# Функция нормализации изображения к диапазону 0–255
def normalize_image(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Функция бинаризации изображения
def binarize_image(image, threshold=50):
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary

# Пути к папкам с изображениями
input_folder = 'pictures_src'
output_folder = 'pictures_results'
os.makedirs(output_folder, exist_ok=True)

# Обработка всех изображений в папке
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    image = cv2.imread(image_path)  # Загружаем изображение

    if image is None:
        print(f"Ошибка: Не удалось загрузить {image_name}")
        continue

    # Преобразуем в полутоновое изображение
    gray_image = convert_to_grayscale(image)

    # Вычисление градиентов Gx, Gy с использованием оператора Собеля
    Gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # Итоговый градиент G
    G = np.sqrt(Gx**2 + Gy**2)

    # Морфологический градиент (разность исходного и расширенного изображения)
    morph_gradient = morphological_gradient(gray_image)

    # Нормализация градиентных изображений
    Gx_norm = normalize_image(Gx)
    Gy_norm = normalize_image(Gy)
    G_norm = normalize_image(G)
    morph_gradient_norm = normalize_image(morph_gradient)

    # Бинаризация градиентной матрицы G
    binary_G = binarize_image(G_norm, threshold=50)

    # Сохранение изображений
    base_name = os.path.splitext(image_name)[0]
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_gray.png"), gray_image)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_gradient_x.png"), Gx_norm)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_gradient_y.png"), Gy_norm)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_gradient.png"), G_norm)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_morph_gradient.png"), morph_gradient_norm)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_binary_gradient.png"), binary_G)

    print(f"Обработано: {image_name}")

print("Все изображения успешно обработаны!")
