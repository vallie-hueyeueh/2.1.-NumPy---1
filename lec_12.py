# -*- coding: utf-8 -*-
"""
Лекция 12

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# 0. ВСПОМОГАТЕЛЬНЫЕ ДАННЫЕ
# =============================================================================
x = np.linspace(0, 10, 50)      # для большинства графиков
x_short = np.linspace(0, 1, 20) # для коротких осей
x_long = np.linspace(0, 20, 100)

# =============================================================================
# 1. БАЗОВЫЕ СПОСОБЫ: plt.axes() и fig.add_axes()
# =============================================================================

# --- 1.1. plt.axes() без параметров (одни оси на всю фигуру) ---
# plt.axes()
# plt.plot(x, np.sin(x))
# plt.show()

# --- 1.2. plt.axes() с координатами [left, bottom, width, height] ---
# # Нижний, левый, ширина, высота (в долях от размера фигуры)
# plt.axes([0.4, 0.3, 0.2, 0.1])   # маленький график внутри
# plt.plot(x, np.cos(x))
# plt.show()

# # Вариант: горизонтальная полоса вверху
# plt.axes([0, 0.3, 1, 0.1])
# plt.plot(x, np.sin(x))
# plt.show()

# # Вариант: вертикальная полоса справа
# plt.axes([0.4, 0, 0.2, 1])
# plt.plot(x, np.cos(x))
# plt.show()

# # Вариант: почти на всю высоту
# plt.axes([0.4, 0, 0.2, 0.9])
# plt.plot(x, np.sin(x))
# plt.show()

# --- 1.3. fig.add_axes() ---
# fig = plt.figure()
# # Основные оси (вся фигура)
# ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# ax1.plot(x, np.sin(x))
# # Вложенные оси (маленький график)
# ax2 = fig.add_axes([0.4, 0.3, 0.2, 0.1])
# ax2.plot(x, np.cos(x))
# plt.show()

# # Другой вариант: две горизонтальные области
# fig = plt.figure()
# ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4])   # верхняя
# ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])   # нижняя
# ax1.plot(x, np.sin(x))
# ax2.plot(x, np.cos(x))
# plt.show()

# # Ещё вариант: оси [0,0,1,1] (вся фигура)
# fig = plt.figure()
# # ax1 = fig.add_axes([0, 0, 1, 1])   
# # ax2 = fig.add_axes([0.4, 0.3, 0.2, 0.1])
# # plt.show()

# =============================================================================
# 2. СЕТКА SUBPLOTS: plt.subplots() И fig.add_subplot()
# =============================================================================

# --- 2.1. plt.subplots() ---
# fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
# x1 = np.linspace(0, 10, 50)
# x2 = np.linspace(0, 20, 100)
# for i in range(2):
#     for j in range(3):
#         if i % 2 == 0:
#             ax[i, j].plot(np.sin(x1 + np.pi/4 * (i*2 + j)))
#         else:
#             ax[i, j].plot(np.sin(x2 + np.pi/4 * (i*2 + j)))
# plt.show()

# # Упрощённый вариант (без share)
# fig, ax = plt.subplots(2, 3)
# for i in range(2):
#     for j in range(3):
#         ax[i, j].plot(np.sin(x + np.pi/4 * i))
# plt.show()

# --- 2.2. fig.add_subplot() ---
# fig = plt.figure()
# for i in range(1, 7):
#     ax = fig.add_subplot(2, 3, i)
#     ax.plot(np.sin(x + np.pi/4 * i))
# plt.show()

# =============================================================================
# 3. НЕРАВНОМЕРНЫЕ СЕТКИ С GRIDSPEC
# =============================================================================

# --- 3.1. Простой GridSpec 2x3 с объединением ячеек ---
# grid = plt.GridSpec(2, 3, wspace=0.1, hspace=0.1)
# # Схема расположения:
# # 0 X Y Y
# # 1 Z Z K
# plt.subplot(grid[0, 0])      # X
# plt.subplot(grid[0, 1:])     # Y (объединяет столбцы 1 и 2)
# plt.subplot(grid[1, :2])     # Z (объединяет столбцы 0 и 1)
# plt.subplot(grid[1, 2])      # K
# plt.show()

# # Другой вариант схемы:
# # 0 X Y K
# # 1 Z Z K
# # plt.subplot(grid[0, 0])
# # plt.subplot(grid[0, 1])
# # plt.subplot(grid[0, 2])
# # plt.subplot(grid[1, :2])
# # plt.subplot(grid[1, 2])
# # plt.show()

# --- 3.2. Более сложный GridSpec 4x4 для marginal histograms ---
# Генерируем коррелированные данные (исправленная ковариационная матрица)
# rng = np.random.default_rng(1)
# cov = [[1, 0.8], [0.8, 1]]   # симметричная положительно определённая
# x_data, y_data = rng.multivariate_normal([0, 0], cov, 1000).T

# grid = plt.GridSpec(4, 4, wspace=0.2, hspace=0.2)

# # Основной график (рассеяние)
# main_axes = plt.subplot(grid[:-1, 1:])   # строки 0-2, столбцы 1-3
# main_axes.plot(x_data, y_data, 'ok', alpha=0.2)

# # Гистограмма для Y (слева)
# y_axes = plt.subplot(grid[:-1, 0], sharey=main_axes)
# y_axes.hist(y_data, bins=40, orientation='horizontal', color='grey')
# y_axes.invert_xaxis()

# # Гистограмма для X (снизу)
# x_axes = plt.subplot(grid[-1, 1:], sharex=main_axes)
# x_axes.hist(x_data, bins=40, orientation='vertical', color='grey')
# x_axes.invert_yaxis()

# plt.show()

# =============================================================================
# 4. ПРИМЕРЫ
# =============================================================================

# --- 4.1. Данные о рождаемости  ---
# try:
#     births = pd.read_csv('births-1969.csv', index_col='Date', parse_dates=True)
#     print(births.head())
# except FileNotFoundError:
#     print("Файл births-1969.csv не найден")
#     births = None

# if births is not None:
#     fig, ax = plt.subplots()
#     ax.plot(births.index, births['births'])
#     # Добавление текста в координатах фигуры
#     # ax.text(0.1, 0.1, "#1_2", transform=fig.transFigure)
#     # Другой вариант аннотации с xytext
#     # ax.annotate("Максимум", xy=("1969-12-31", 4500), xytext=("1969-12-1", 4500),
#     #             arrowprops=dict(arrowstyle="->"))
#     plt.show()

# --- 4.2. Данные о велосипедистах ---
# try:
#     df = pd.read_csv('FremontBridge.csv', index_col='Date', parse_dates=True)
#     df.columns = ["Total", "East", "West"]
#     df = df.dropna()
# except FileNotFoundError:
#     print("Файл FremontBridge.csv не найден")
#     df = None

# if df is not None:
#     # Среднее по дням недели
#     weekly_profile = df.groupby(df.index.dayofweek).mean()
#     weekly_profile.plot(marker='o')
#     plt.xticks(range(7), ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'])
#     plt.ylabel("Среднее количество велосипедистов в час")
#     plt.title("Среднечасовой профиль по дням недели")
#     plt.show()

# =============================================================================
# 5. ДОПОЛНИТЕЛЬНЫЕ ПРИМЕРЫ
# =============================================================================

# --- 5.1. Пример с ошибкой в ковариационной матрице ---
# # Следующий код вызовет предупреждение, т.к. матрица [[1,2],[3,4]] не симметрична
# # rng = np.random.default_rng(1)
# # x_bad, y_bad = rng.multivariate_normal([0, 0], [[1, 2], [3, 4]], 1000).T
# # print("пример выдаст RuntimeWarning (несимметричная матрица)")

# --- 5.2. Пример с add_subplot и неверным синтаксисом ---
# # fig = plt.figure()
# # ax1 = fig.add_subplot(2,3,1)
# # ax1.plot(np.sin(x \ np.pi/4))   # ошибка: обратный слэш вместо +
# # plt.show()

# --- 5.3. Пример с неправильным количеством аргументов в add_axes ---
# # fig = plt.figure()
# # ax1 = fig.add_axes([0, 0, 1])   # ошибка: нужно 4 числа
# # plt.show()

# --- 5.4. Пример с использованием sharex/sharey через GridSpec ---
# # grid = plt.GridSpec(2, 2)
# # ax1 = plt.subplot(grid[0, 0])
# # ax2 = plt.subplot(grid[0, 1], sharex=ax1, sharey=ax1)
# # ax3 = plt.subplot(grid[1, :], sharex=ax1)
# # ax1.plot(x, np.sin(x))
# # ax2.plot(x, np.cos(x))
# # ax3.plot(x, np.sin(2*x))
# # plt.show()
