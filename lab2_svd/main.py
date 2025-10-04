import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Загружаем изображение через PIL
image = Image.open('1121736.jpg')

# Преобразуем изображение в оттенки серого (8-bit) и в numpy-массив [0,1]
gray_image_pil = image.convert('L')
gray_image = np.asarray(gray_image_pil, dtype=np.float32) / 255.0

# Преобразуем изображение в 2D матрицу (если оно не 2D)
U, S, Vt = np.linalg.svd(gray_image, full_matrices=False)

# Сжимаем изображение, используя только первые k сингулярных значений
k = min(1333, min(gray_image.shape))  # безопасно ограничим k
S_k = np.zeros((k, k))
S_k[:k, :k] = np.diag(S[:k])

# Реконструируем изображение с использованием только k сингулярных значений
approx_image = np.dot(U[:, :k], np.dot(S_k, Vt[:k, :]))

# Подсчет размеров хранения и коэффициента сжатия
m, n = gray_image.shape
original_elements = m * n
compressed_elements = m * k + k + k * n  # U(m×k) + S(k) + Vt(k×n)
compression_factor = original_elements / compressed_elements if compressed_elements else float('inf')

print(f"Размер исходного изображения: {m} x {n} (элементов: {original_elements})")
print(f"Хранение SVD при k={k}: U({m}x{k}) + S({k}) + Vt({k}x{n}) = {compressed_elements} элементов")
print(f"Оценочный коэффициент сжатия: {compression_factor:.2f}x")

# Показать оригинальное и сжатое изображение
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title("Оригинальное изображение")
axes[1].imshow(approx_image, cmap='gray')
axes[1].set_title(f"Сжатое изображение (k={k})")
fig.suptitle(f"{m}x{n}, k={k} — хранение: {original_elements} → {compressed_elements} элементов (≈ {compression_factor:.2f}x)")
plt.tight_layout()
plt.show()
