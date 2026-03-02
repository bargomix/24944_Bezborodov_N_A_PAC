import matplotlib.pyplot as plt
from PIL import Image

# Загрузка изображения
img = Image.open('abc.png')

# Показать изображение
plt.imshow(img)
plt.axis('off')  # Убрать оси
plt.show()
