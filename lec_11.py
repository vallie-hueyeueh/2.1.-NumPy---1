"""
Лекция 11: Визуализация данных с помощью Matplotlib, Pandas и Scikit-learn.
"""

import numpy as np                # работа с массивами и математикой
import pandas as pd               # работа с табличными данными (CSV и др.)
import matplotlib.pyplot as plt   # построение графиков
from sklearn.datasets import load_digits       # датасет рукописных цифр
from sklearn.manifold import Isomap            # нелинейное снижение размерности

# -----------------------------------------------------------------------------
# Commit 1: Базовый график синуса и косинуса с легендой
# -----------------------------------------------------------------------------
#фрагмент был закомментирован в пользу более сложного примера.

# x = np.linspace(0, 10, 1000)          # 1000 точек от 0 до 10
# fig, ax = plt.subplots()              # создаём фигуру и оси
# ax.plot(x, np.sin(x), "-k", label="Синус")   # чёрная линия
# ax.plot(x, np.cos(x), "-r", label="Косинус") # красная линия
# ax.axis("equal")                       # одинаковый масштаб по осям
# ax.legend(frameon=True, shadow=True, borderpad=1, loc="lower center", ncol=2)
# plt.show()

# =============================================================================
# Commit 2: Построение нескольких синусоид со сдвигом фазы
# =============================================================================
# Создаём массив x (1000 точек)
x = np.linspace(0, 10, 1000)

# Формируем матрицу y: каждый столбец – синусоида со своим сдвигом.
# np.arange(0, 2, 0.3) -> [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
# Всего 7 кривых.
y = np.sin(x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.3))

# Строим все кривые одним вызовом plt.plot()
lines = plt.plot(x, y)

# Вариант с явным указанием легенды
# plt.legend(lines, ["первая", "вторая", "третья", "четвертая", "пятая", "шестая", "седьмая"])

# Альтернативный способ: задавать label при каждом plot
# for i in range(y.shape[1]):
#     plt.plot(x, y[:, i], label=f"Кривая {i+1}")
# plt.legend()

# plt.legend(lines, ["первая", "вторая", "третья", "четвертая"])

# -----------------------------------------------------------------------------
# Commit 3: Загрузка данных о городах
# -----------------------------------------------------------------------------
csv_path = "./digital_python-25-26/data/california_cities.csv"
try:
    cities = pd.read_csv(csv_path)
    print("=== Первые 5 строк файла california_cities.csv ===")
    print(cities.head())
    print("\n=== Информация о данных ===")
    print(cities.info())
except FileNotFoundError:
    print(f"[Внимание] Файл {csv_path} не найден. Пропуск блока.")
    cities = None

# =============================================================================
# Commit 4: Визуализация датасета рукописных цифр (digits) с помощью Isomap
# =============================================================================
# Загружаем датасет цифр (8x8 пикселей, 1797 образцов)
digits = load_digits()

# ---- 4.1 Отображение первых 64 цифр в виде сетки 8x8 ----
fig, axes = plt.subplots(8, 8, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="binary")
    ax.set(xticks=[], yticks=[])   # убираем координатные метки
plt.suptitle("Примеры рукописных цифр из датасета digits", fontsize=14)

# ---- 4.2 Снижение размерности с 64 до 2 компонент с помощью Isomap ----
# Isomap сохраняет геодезические расстояния между точками.
iso = Isomap(n_components=2, n_neighbors=10)
# Преобразуем данные: из (1797, 64) -> (1797, 2)
prj = iso.fit_transform(digits.data)

# ---- 4.3 Визуализация точек в новом 2D-пространстве, раскраска по истинной цифре ----
plt.figure(figsize=(10, 8))
cmap = plt.get_cmap("jet", 12)  

sc = plt.scatter(prj[:, 0], prj[:, 1], c=digits.target, cmap=cmap, alpha=0.7, edgecolors='k')
plt.colorbar(sc, ticks=range(10), label="Цифра")
plt.title("Проекция датасета digits на плоскость с помощью Isomap")
plt.xlabel("Первая компонента")
plt.ylabel("Вторая компонента")


# Commit 5: Финальный показ
plt.show()
