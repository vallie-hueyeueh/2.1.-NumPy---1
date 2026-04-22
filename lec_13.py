# -*- coding: utf-8 -*-
"""
Лекция 13

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# =============================================================================
# ЧАСТЬ 1: СТИЛИ ОФОРМЛЕНИЯ И ЦВЕТА ФОНА
# =============================================================================

# --- 1.1. Список доступных стилей ---
print("Доступные стили:", plt.style.available)

# --- 1.2. Использование стиля 'grayscale' ---
# plt.style.use('grayscale')

# --- 1.3. Загрузка пользовательского стиля из файла ---
# Содержимое файла lec_13.style:
#   figure.facecolor: '#921212'
#   axes.facecolor: '#adadad'
# plt.style.use('./digital_python-25-26/lec_13.style')

# --- 1.4. Настройка цветов фигуры и осей ---
# # Через plt.figure и plt.axes
# plt.figure(facecolor='#921212')
# plt.axes(facecolor='#adadad')
# # Через rcParams (глобальные настройки)
# plt.rc("figure", facecolor='#921212')
# plt.rc("axes", facecolor='#adadad')

# Генерация случайных данных для гистограммы
x = np.random.randn(1000)

# Построение гистограммы с настройками
plt.hist(x, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
plt.title("Гистограмма случайных данных")
plt.xlabel("Значение")
plt.ylabel("Частота")
plt.show()

# =============================================================================
# ЧАСТЬ 2: ЛОКАТОРЫ И ФОРМАТЕРЫ ОСЕЙ
# =============================================================================

# --- 2.1. Пример с логарифмической шкалой и GridSpec ---
grid = plt.GridSpec(1, 2, wspace=0.3)

ax1 = plt.subplot(grid[0, 0])
ax1.set_xscale("log")
ax1.set_xlim(1, 1000)
ax1.grid(True, which="major", linestyle='--', linewidth=0.5)
ax1.set_title("Логарифмическая шкала X")

ax2 = plt.subplot(grid[0, 1])
ax2.set_yscale("log")
ax2.set(ylim=(1, 1000))
ax2.grid(True, which="both", axis="y", linestyle=':', linewidth=0.5)
ax2.set_title("Логарифмическая шкала Y")

plt.show()

# --- 2.2. Вывод информации о локаторах и форматерах ---
# Можно получить текущие настройки осей:
# print(ax1.xaxis.get_major_locator())
# print(ax1.xaxis.get_major_formatter())
# print(ax1.xaxis.get_minor_locator())
# print(ax1.xaxis.get_minor_formatter())
# Аналогично для yaxis.

# --- 2.3. Демонстрация различных локаторов (на 8 субграфиках) ---
fig, ax = plt.subplots(8, 1, figsize=(8, 10))
plt.subplots_adjust(hspace=0.5)

x_vals = np.linspace(0, 10, 10)

for i, a in enumerate(ax):
    a.plot(x_vals, np.ones_like(x_vals) * 2, 'o-', color='darkblue')
    a.set_ylim(0, 4)
    a.set_title(f"Locator {i+1}")

# Установка разных локаторов для оси X
ax[0].xaxis.set_major_locator(plt.NullLocator())           # без меток
ax[1].xaxis.set_major_locator(plt.MultipleLocator(0.8))    # шаг 0.8
ax[2].xaxis.set_major_locator(plt.FixedLocator([1, 3, 8, 9])) # фиксированные позиции
ax[3].xaxis.set_major_locator(plt.LinearLocator(numticks=4))  # 4 равномерные метки
ax[4].xaxis.set_major_locator(plt.IndexLocator(base=2, offset=1.3)) # через 2, начиная с 1.3
ax[5].xaxis.set_major_locator(plt.AutoLocator())           # автоматический
ax[6].xaxis.set_major_locator(plt.MaxNLocator(8))          # максимум 8 меток
ax[7].xaxis.set_major_locator(plt.LogLocator(base=3))      # логарифмический с основанием 3

plt.show()

# --- 2.4. Демонстрация различных форматеров (на 4 субграфиках) ---
fig, ax = plt.subplots(4, 1, figsize=(8, 6))
plt.subplots_adjust(hspace=0.5)

x_form = np.linspace(0, 10, 10)
for a in ax:
    a.plot(x_form, np.ones_like(x_form) * 2, 's-', color='darkgreen')
    a.set_ylim(0, 4)

# Установка разных форматеров
ax[0].xaxis.set_major_formatter(plt.NullFormatter())               # без подписей
ax[1].xaxis.set_major_formatter(plt.FixedFormatter(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']))
ax[2].xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f $m^2$'))  # с единицами
ax[3].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=10))        # в процентах

plt.show()

# =============================================================================
# ЧАСТЬ 3: КАСТОМНЫЙ ФОРМАТТЕР ДЛЯ МЕТОК КРАТНЫХ π/2
# =============================================================================

# Функция-форматтер, возвращающая красивые обозначения для углов
def ff(value, tick_number):
    """
    Преобразует значение x в метку вида: 0, π/2, π, 3π/2, 2π, ..

    """
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\frac{\pi}{2}$"
    elif N == 2:
        return r"$\pi$"
    elif N % 2 == 0:
        t = int(N / 2)
        return f"{t}" + r"$\pi$"
    else:
        # Нечётные, кратные π/2 (кроме 1) – например, 3π/2, 5π/2..
        t = int(N // 2)  # целая часть от деления на 2
        if N == 3:
            return r"$\frac{3\pi}{2}$"
        else:
            return f"{t+1}" + r"$\frac{\pi}{2}$"

# Создание графика синуса и косинуса с кастомными метками
x_sin = np.linspace(0, 4 * np.pi, 1000)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x_sin, np.sin(x_sin), label="Sinus", color='blue')
ax.plot(x_sin, np.cos(x_sin), label="Cosinus", color='red')
ax.grid(True, linestyle=':', alpha=0.7)
ax.legend()
ax.set_xlim(0, 4 * np.pi)

# Устанавливаем локаторы: основные метки через π/2, дополнительные через π/4
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))

# Применяем кастомный форматер
ax.xaxis.set_major_formatter(plt.FuncFormatter(ff))

plt.show()

# =============================================================================
# ЧАСТЬ 4: 3D-ГРАФИКА (CONTOUR3D)
# =============================================================================

# модуль для 3D
from mpl_toolkits import mplot3d

def f(x, y):
    # Исходная функция: sin( sqrt(x^2 + y^2) )
    return np.sin(np.sqrt(x**2 + y**2))
    # Альтернативная функция:
    # return np.sin(x * np.pi / 2 + np.sqrt(x**2 + y**2))

# Создаём сетку
x_vals = np.linspace(-6, 6, 30)
y_vals = np.linspace(-10, 10, 50)

print("Размер x:", x_vals.shape)
print("Размер y:", y_vals.shape)

X, Y = np.meshgrid(x_vals, y_vals)
print("Размер X после meshgrid:", X.shape)
print("Размер Y:", Y.shape)

# Вычисляем значение функции на сетке
Z = f(X, Y)

# Выводим форму и значения Z (для информации)
print("Форма Z:", Z.shape)
print("Z (фрагмент):\n", Z[:5, :5])

# Создаём 3D-фигуру и оси
fig = plt.figure(figsize=(10, 6))
ax_3d = plt.axes(projection="3d")

# Контурный 3D-график (contour3D) с 100 уровнями
ax_3d.contour3D(X, Y, Z, 100, cmap='viridis')
ax_3d.set_xlabel("x")
ax_3d.set_ylabel("y")
ax_3d.set_zlabel("z")
ax_3d.set_title("3D контурный график f(x,y) = sin(sqrt(x^2+y^2))")

# Поворот камеры: elev (угол возвышения), azim (азимут)
ax_3d.view_init(elev=30, azim=10)   # можно менять, например (30,10)
plt.show()

# =============================================================================
# ЧАСТЬ 5: ОТОБРАЖЕНИЕ НАБОРА ИЗОБРАЖЕНИЙ (ЛИЦА OLIVETTI)
# =============================================================================

# Пример с датасетом Olivetti faces
# from sklearn.datasets import fetch_olivetti_faces
# faces = fetch_olivetti_faces().images

# Создаём сетку 7x7 субграфиков
# fig, ax = plt.subplots(7, 7, figsize=(8, 8))
# fig.subplots_adjust(hspace=0, wspace=0)
#
# for i in range(7):
#     for j in range(7):
#         idx = 7 * i + j
#         ax[i, j].xaxis.set_major_locator(plt.NullLocator())
#         ax[i, j].yaxis.set_major_locator(plt.NullLocator())
#         ax[i, j].imshow(faces[idx], cmap="binary_r")
#
# plt.show()