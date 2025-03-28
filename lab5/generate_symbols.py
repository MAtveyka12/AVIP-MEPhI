from PIL import Image, ImageDraw, ImageFont
import os

# Создание папки для хранения изображений
output_folder = "generated_images_letters"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Казахский строчный алфавит (вариант 15)
letters = "аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя"

# Загрузка шрифта (Times New Roman, кегль 52)
font = ImageFont.truetype(r"C:\Windows\Fonts\times.ttf", 52)

for letter in letters:
    img = Image.new('L', (80, 80), color=255)  # Белый фон
    draw = ImageDraw.Draw(img)

    # Определение размеров текста
    text_width, text_height = draw.textbbox((0, 0), letter, font=font)[2:4]
    position = ((100 - text_width) // 2.5, (100 - text_height) // 7)

    # Отрисовка символа
    draw.text(position, letter, fill=0, font=font)

    # Сохранение изображения
    img.save(f"{output_folder}/{letter}.png")

print("Генерация изображений завершена!")