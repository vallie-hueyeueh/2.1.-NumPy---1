"""
Лекция 14

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# =============================================================================
# ЧАСТЬ 1. 3D-ГРАФИКА
# =============================================================================

# Определяем функцию двух переменных
def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))

# -----------------------------------------------------------------------------
# 1.1. Построение поверхности на прямоугольной сетке (декартовы координаты)
# -----------------------------------------------------------------------------

# Создаём сетку (30 точек по x, 50 по y)
x = np.linspace(-6, 6, 30)
y = np.linspace(-10, 10, 50)

# Преобразуем в двумерные массивы координат
X, Y = np.meshgrid(x, y)

# Вычисляем Z
Z = f(X, Y)

# Печатаем размеры для проверки
print("X.shape:", X.shape)   # (50, 30)
print("Y.shape:", Y.shape)   # (50, 30)

# Создаём фигуру и 3D-оси
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection="3d")

# Варианты визуализации:

# 1. Точечный график (scatter3D) – цвет точек соответствует Z
# ax.scatter3D(X, Y, Z, c=Z, cmap='viridis', s=1)

# 2. Каркасная поверхность (wireframe)
# ax.plot_wireframe(X, Y, Z, color='blue', linewidth=0.5)

# 3. Закрашенная поверхность (plot_surface)
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

# Настройка подписей и заголовка
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("f(x,y) = sin(√(x²+y²))")

plt.show()

# -----------------------------------------------------------------------------
# 1.2. Построение поверхности в цилиндрических координатах (R, angle)
# -----------------------------------------------------------------------------

# Создаём полярную сетку: угол от 0 до 2π, радиус от 0 до 6
angle = np.linspace(0, 2 * np.pi, 50)   # 50 углов
r = np.linspace(0, 6, 30)               # 30 радиусов

R, Angle = np.meshgrid(r, angle)

# Переход к декартовым координатам
X_polar = R * np.sin(Angle)
Y_polar = R * np.cos(Angle)
Z_polar = f(X_polar, Y_polar)

# Новый график
fig2 = plt.figure(figsize=(10, 6))
ax2 = plt.axes(projection="3d")

# поверхность в цилиндрических координатах
ax2.plot_surface(X_polar, Y_polar, Z_polar, cmap='plasma', edgecolor='none')
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("z")
ax2.set_title("Поверхность в цилиндрических координатах")
plt.show()

# -----------------------------------------------------------------------------
# 1.3. Дополнительные варианты сеток
# -----------------------------------------------------------------------------

angle_corr = np.linspace(0, 4 * np.pi, 50)
r_corr = np.linspace(0, 6, 30)
R_corr, Angle_corr = np.meshgrid(r_corr, angle_corr)
X_corr = R_corr * np.sin(Angle_corr)
Y_corr = R_corr * np.cos(Angle_corr)
Z_corr = f(X_corr, Y_corr)

fig3 = plt.figure(figsize=(10, 6))
ax3 = plt.axes(projection="3d")
ax3.plot_surface(X_corr, Y_corr, Z_corr, cmap='inferno')
ax3.set_title("Поверхность с углом 0..4π")
plt.show()

# =============================================================================
# ЧАСТЬ 2. ВИЗУАЛИЗАЦИЯ КАТЕГОРИАЛЬНЫХ ДАННЫХ С SEABORN (НА ПРИМЕРЕ АВТОМОБИЛЕЙ)
# =============================================================================

# небольшой датасет, чтобы код не падал
try:
    cars = pd.read_csv('cars.csv')
    print("cars.csv успешно загружен")
except FileNotFoundError:
    print("Файл cars.csv не найден. Создаю тестовый датасет")
    # Создаём демонстрационные данные
    np.random.seed(42)
    n = 200
    cars = pd.DataFrame({
        'fuel': np.random.choice(['Petrol', 'Diesel', 'CNG'], n),
        'transmission': np.random.choice(['Manual', 'Automatic'], n),
        'seller_type': np.random.choice(['Individual', 'Dealer'], n),
        'year': np.random.randint(2000, 2020, n),
        'selling_price': np.random.normal(500000, 200000, n).astype(int)
    })
    cars['selling_price'] = np.abs(cars['selling_price'])  # чтобы цены были положительными

# -----------------------------------------------------------------------------
# 2.1. Совместное распределение (jointplot)
# -----------------------------------------------------------------------------

# # 1. Обычная диаграмма рассеяния + гистограммы
# sns.jointplot(x="year", y="selling_price", data=cars)
# plt.show()

# # 2. С ядерной оценкой плотности (kde)
# sns.jointplot(x="year", y="selling_price", data=cars, kind='kde')
# plt.show()

# # 3. Шестиугольные ячейки (hex)
# sns.jointplot(x="year", y="selling_price", data=cars, kind='hex')
# plt.show()

# # 4. С разделением по цвету (hue)
# sns.jointplot(x="year", y="selling_price", data=cars, hue="transmission")
# plt.show()

# -----------------------------------------------------------------------------
# 2.2. Графики для связи категорий и чисел
# -----------------------------------------------------------------------------

# # Столбчатая диаграмма (barplot)
# sns.barplot(x='fuel', y='selling_price', data=cars, estimator=np.mean, hue='transmission')
# plt.show()

# # Более сложный catplot с несколькими панелями (col)
# g = sns.catplot(x='fuel', y='selling_price', data=cars, estimator=np.mean, hue='transmission', kind='bar', col='seller_type')
# plt.show()

# # Точечный график (pointplot)
# sns.pointplot(x='fuel', y='selling_price', data=cars, estimator=np.mean, hue='transmission')
# plt.show()

# # Ящик с усами (boxplot)
# sns.boxplot(x='fuel', y='selling_price', data=cars, hue='transmission')
# plt.show()

# # Скрипичный график (violinplot) – совмещает boxplot и kde
# sns.violinplot(x='fuel', y='selling_price', data=cars, hue='transmission')
# plt.show()

# # Полосковый график (stripplot) – точки с разбросом
# sns.stripplot(x='fuel', y='selling_price', data=cars, hue='transmission')
# plt.show()

# -----------------------------------------------------------------------------
# 2.3. Комбинированный пример: boxplot + stripplot
# -----------------------------------------------------------------------------

plt.figure(figsize=(10, 6))
g = sns.catplot(x='fuel', y='selling_price', data=cars, kind='box', hue='transmission')
sns.stripplot(x='fuel', y='selling_price', data=cars, color='black', alpha=0.3, ax=g.ax)
plt.show()
