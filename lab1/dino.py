import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

# Пути к изображениям
template_image_path = "abc.png"  # Изображение с шаблоном
search_image_path = "Ab_C.jpg"    # Изображение для поиска

# Координаты шаблона на первом изображении (y1, y2, x1, x2)
y1, y2 = 480, 560
x1, x2 = 30, 230

# Загрузка и проверка изображений
template_bgr = cv2.imread(template_image_path)
search_bgr = cv2.imread(search_image_path)

# Конвертация BGR -> RGB и вырезка шаблона
template_img_rgb = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB)
search_img_rgb = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2RGB)
template = template_img_rgb[y1:y2, x1:x2]

# Инициализация ResNet-18
model = torchvision.models.resnet18(pretrained=True)
layer4_features = None  # Признаки из layer4 [512, h, w]
avgpool_emb = None      # Эмбеддинг из avgpool [512]

# Хуки для извлечения промежуточных признаков
def get_features(module, inputs, output):
    global layer4_features
    layer4_features = output

def get_embedding(module, inputs, output):
    global avgpool_emb
    avgpool_emb = output

# Регистрация хуков и перевод модели в режим оценки
model.layer4.register_forward_hook(get_features)
model.avgpool.register_forward_hook(get_embedding)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Параметры нормализации ImageNet
mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
std  = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]

def prep(rgb: np.ndarray) -> torch.Tensor:
    """Препроцессинг: [H,W,3] -> [1,3,H,W] + нормализация"""
    x = torch.from_numpy(rgb).permute(2, 0, 1).float()[None].to(device) / 255.0
    return (x - mean) / std

# Извлечение признаков через прямой проход сети
with torch.no_grad():
    # Шаблон -> avgpool -> вектор [512]
    _ = model(prep(template))
    q = avgpool_emb.flatten()
    
    # Изображение -> layer4 -> feature map [512, h, w]
    _ = model(prep(search_img_rgb))
    fm = layer4_features.squeeze(0)

# Вычисление тепловой карты через косинусное сходство
q  = F.normalize(q, dim=0)   # Нормализация вектора запроса
fm = F.normalize(fm, dim=0)  # Нормализация каждого вектора в feature map
heat = torch.einsum("c,chw->hw", q, fm).cpu().numpy()  # Скалярное произведение

# Увеличение карты до размера исходного изображения
H, W = search_img_rgb.shape[:2]
heat_up = cv2.resize(heat, (W, H), interpolation=cv2.INTER_LINEAR)

# Поиск координат максимального совпадения
y_max, x_max = np.unravel_index(np.argmax(heat_up), heat_up.shape)
max_similarity = heat_up[y_max, x_max]

print(f"Найдено: x={x_max}, y={y_max}, сходство={max_similarity:.4f}")

# Визуализация результатов (2x2)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# [0,0] Исходное изображение с выделенным шаблоном
axes[0, 0].imshow(template_img_rgb)
rect_source = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, 
                             edgecolor='lime', facecolor='none', label='Шаблон')
axes[0, 0].add_patch(rect_source)
axes[0, 0].legend(loc='upper right', fontsize=9)
axes[0, 0].axis("off")

# [0,1] Вырезанный шаблон
axes[0, 1].imshow(template)
axes[0, 1].axis("off")

# [1,0] Изображение для поиска с найденной точкой
axes[1, 0].imshow(search_img_rgb)
axes[1, 0].scatter([x_max], [y_max], c='red', marker="x", s=300, linewidths=4, 
                   label=f'Найдено: ({x_max}, {y_max})')
axes[1, 0].legend(loc='upper right', fontsize=10)
axes[1, 0].axis("off")

# [1,1] Тепловая карта совпадений
axes[1, 1].imshow(search_img_rgb)
axes[1, 1].imshow(heat_up, cmap="jet", alpha=0.6, interpolation='bilinear')
axes[1, 1].scatter([x_max], [y_max], c='white', marker="x", s=300, linewidths=4, 
                   edgecolors='black', label='Максимум')
axes[1, 1].legend(loc='upper right', fontsize=9)
axes[1, 1].axis("off")

plt.tight_layout()
plt.show()