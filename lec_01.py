"""
ЛЕКЦИЯ: Библиотеки Python для научных расчетов и машинного обучения
Тема: Введение в NumPy

"""

# ========== ИМПОРТ БИБЛИОТЕК ==========
import sys
import array
import numpy as np

print("=" * 60)
print("ЛЕКЦИЯ: Введение в NumPy")
print("=" * 60)

# ========== 1 СРАВНЕНИЕ СТРУКТУР ДАННЫХ ==========
print("\n1. СРАВНЕНИЕ СТРУКТУР ДАННЫХ PYTHON")
print("-" * 40)

# Обычный список Python
l = [1, 2, 3, 4, 5]
print(f"Список: {l}")
print(f"Тип списка: {type(l)}")
print(f"Размер списка в памяти: {sys.getsizeof(l)} байт")

# Пустой список
l1 = []
print(f"\nПустой список: {l1}")
print(f"Тип: {type(l1)}")
print(f"Размер: {sys.getsizeof(l1)} байт")

# Массив из модуля array
al = array.array('i', [])
print(f"\nМассив array: {al}")
print(f"Тип: {type(al)}")
print(f"Размер: {sys.getsizeof(al)} байт")

# NumPy массив
a_np = np.array(l)
print(f"\nNumPy массив: {a_np}")
print(f"Тип: {type(a_np)}")
print(f"Размер: {sys.getsizeof(a_np)} байт")

# Сравнение размеров
print("\n--- Сравнение размеров в памяти ---")
print(f"list(python): {sys.getsizeof(l)} байт")
ap = array.array('i', l)
print(f"array(python): {sys.getsizeof(ap)} байт")
print(f"array(numpy): {sys.getsizeof(a_np)} байт")

# ========== 2 СОЗДАНИЕ МАССИВОВ NUMPY ==========
print("\n\n2. СОЗДАНИЕ МАССИВОВ NUMPY")
print("-" * 40)

# Базовое создание
print("--- Базовые методы создания ---")
a1 = np.array([1, 2, 3, 4, 5])
print(f"Из списка: {a1}")

a2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\nДвумерный массив:\n{a2}")

# Автоматическое приведение типов
print("\n--- Приведение типов ---")
a_mixed = np.array([1.01, 2, 3, 4, 5, "a"])
print(f"Смешанные типы: {a_mixed}")
print(f"Тип данных: {a_mixed.dtype}")

# Явное указание типа
a_int = np.array([1.99, 2, 3, 4, 5], dtype=int)
print(f"\nС явным типом int: {a_int}")
print(f"Тип данных: {a_int.dtype}")

# Специальные функции создания
print("\n--- Специальные функции ---")
print(f"zeros(3,4):\n{np.zeros((3, 4))}")
print(f"\nones(2,2):\n{np.ones((2, 2))}")
print(f"\nfull((2,3), 7):\n{np.full((2, 3), 7)}")
print(f"\neye(3):\n{np.eye(3)}")

# Диапазоны
print("\n--- Диапазоны ---")
print(f"arange(0, 10, 2): {np.arange(0, 10, 2)}")
print(f"arange(0, 20, 3): {np.arange(0, 20, 3)}")
print(f"linspace(0, 1, 5): {np.linspace(0, 1, 5)}")

# Случайные числа
print("\n--- Случайные числа ---")
rng = np.random.default_rng(seed=42)  # Для воспроизводимости
print(f"5 случайных целых [0, 10): {rng.integers(0, 10, size=5)}")
print(f"Матрица 2x2 [0,1):\n{rng.random((2, 2))}")

# ========== 3 АТРИБУТЫ МАССИВОВ ==========
print("\n\n3. АТРИБУТЫ МАССИВОВ")
print("-" * 40)

a = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Массив:\n{a}")
print(f"Тип данных (dtype): {a.dtype}")
print(f"Размерность (ndim): {a.ndim}")
print(f"Форма (shape): {a.shape}")
print(f"Общий размер (size): {a.size}")

# Изменение типа данных
print("\n--- Изменение типа данных ---")
a_float = np.array([1, 2, 3, 4, 5])
print(f"Исходный: {a_float}, тип: {a_float.dtype}")
a_float[0] = 3.5
print(f"После a[0] = 3.5: {a_float}, тип: {a_float.dtype}")

# ========== 4 ИНДЕКСАЦИЯ ==========
print("\n\n4. ИНДЕКСАЦИЯ")
print("-" * 40)

# Одномерные массивы
print("--- Одномерные массивы ---")
a = np.array([10, 20, 30, 40, 50])
print(f"Массив: {a}")
print(f"a[0]: {a[0]}")
print(f"a[-2]: {a[-2]}")

# Изменение элемента
a[1] = 200
print(f"После a[1] = 200: {a}")

# Многомерные массивы
print("\n--- Многомерные массивы ---")
m = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"Матрица 3x4:\n{m}")
print(f"m[0, 1]: {m[0, 1]}")
print(f"m[1, :]: {m[1, :]}")
print(f"m[:, 2]: {m[:, 2]}")
print(f"m[0:2, 1:3]:\n{m[0:2, 1:3]}")

# ========== 5 СРЕЗЫ ==========
print("\n\n5. СРЕЗЫ")
print("-" * 40)

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"Массив: {a}")
print(f"a[:3]: {a[:3]}")
print(f"a[3:]: {a[3:]}")
print(f"a[1:4]: {a[1:4]}")
print(f"a[:2:2]: {a[:2:2]}")
print(f"a[1::2]: {a[1::2]}")

print("\n--- Срезы с отрицательным шагом ---")
print(f"a[::-1]: {a[::-1]}")
print(f"a[5:1:-1]: {a[5:1:-1]}")

# ========== 6 ИЗМЕНЕНИЕ ФОРМЫ МАССИВОВ ==========
print("\n\n6. ИЗМЕНЕНИЕ ФОРМЫ МАССИВОВ")
print("-" * 40)

# Создание исходного массива
a = np.arange(1, 13)
print(f"Исходный массив: {a}")
print(f"Форма: {a.shape}, Размерность: {a.ndim}")

# Метод reshape
print("\n--- Метод reshape ---")
a1 = a.reshape(1, 12)
print(f"a.reshape(1, 12):\n{a1}")
print(f"Форма: {a1.shape}, a1[0, 3]: {a1[0, 3]}")

a2 = a.reshape(2, 6)
print(f"\na.reshape(2, 6):\n{a2}")
print(f"Форма: {a2.shape}")

# Добавление новой оси
a2_newaxis = a[:, np.newaxis]
print(f"\na[:, np.newaxis]:\n{a2_newaxis}")
print(f"Форма: {a2_newaxis.shape}")

# Многомерные преобразования
a3 = a.reshape(2, 2, 3)
print(f"\na.reshape(2, 2, 3):\n{a3}")
print(f"Форма: {a3.shape}, a3[0, 1, 2]: {a3[0, 1, 2]}")

# Порядок 'F' (Fortran)
a4 = a.reshape((2, 6), order="F")
print(f"\na.reshape((2, 6), order='F'):\n{a4}")
print(f"Форма: {a4.shape}, a4[1, 4]: {a4[1, 4]}")

# ========== 7 ВЕКТОРНЫЕ ОПЕРАЦИИ ==========
print("\n\n7. ВЕКТОРНЫЕ ОПЕРАЦИИ")
print("-" * 40)

x = np.array([1, 2, 3])
y = np.array([10, 20, 30])

print(f"x: {x}, y: {y}")
print(f"Сложение (x + y): {x + y}")
print(f"Вычитание (x - y): {x - y}")
print(f"Умножение (x * y): {x * y}")
print(f"Деление (y / x): {y / x}")
print(f"Возведение в степень (x ** 2): {x ** 2}")
print(f"Квадратный корень (np.sqrt(x)): {np.sqrt(x)}")

# Операции со скаляром
print(f"\nОперации со скаляром (x * 5): {x * 5}")

# ========== 8 АГРЕГИРУЮЩИЕ ФУНКЦИИ ==========
print("\n\n8. АГРЕГИРУЮЩИЕ ФУНКЦИИ")
print("-" * 40)

data = np.arange(1, 13).reshape(3, 4)
print(f"Матрица данных 3x4:\n{data}")

print(f"\n--- Базовые агрегации ---")
print(f"Сумма всех элементов: {data.sum()}")
print(f"Среднее значение: {data.mean():.2f}")
print(f"Минимальное значение: {data.min()}")
print(f"Максимальное значение: {data.max()}")
print(f"Стандартное отклонение: {data.std():.2f}")

print(f"\n--- Агрегации по осям ---")
print(f"Сумма по строкам (axis=1): {data.sum(axis=1)}")
print(f"Сумма по столбцам (axis=0): {data.sum(axis=0)}")
print(f"Среднее по строкам: {data.mean(axis=1)}")

# ========== 9 БУЛЕВЫ МАСКИ И ФИЛЬТРАЦИЯ ==========
print("\n\n9. БУЛЕВЫ МАСКИ И ФИЛЬТРАЦИЯ")
print("-" * 40)

values = np.array([3, 7, 1, 9, 4, 12, 8, 5])
print(f"Исходный массив: {values}")

# Создание булевой маски
mask = values > 5
print(f"\nМаска (values > 5): {mask}")
print(f"Элементы > 5: {values[mask]}")

# Фильтрация по нескольким условиям
mask2 = (values > 3) & (values < 9)
print(f"\nМаска (3 < values < 9): {mask2}")
print(f"Элементы 3 < x < 9: {values[mask2]}")

# Функция np.where
result = np.where(values > 6, values * 10, values)
print(f"\nnp.where(values > 6, values*10, values): {result}")

# ========== 10 BROADCASTING ==========
print("\n\n10. BROADCASTING (РАСТЯГИВАНИЕ)")
print("-" * 40)

matrix = np.ones((3, 4))
vector_row = np.array([1, 2, 3, 4])
vector_col = np.array([[1], [2], [3]])

print(f"Матрица 3x4:\n{matrix}")
print(f"\nВектор-строка: {vector_row}")
print(f"matrix + vector_row:\n{matrix + vector_row}")

print(f"\nВектор-столбец:\n{vector_col}")
print(f"matrix + vector_col:\n{matrix + vector_col}")

# ========== 11 КОНКАТЕНАЦИЯ ==========
print("\n\n11. КОНКАТЕНАЦИЯ МАССИВОВ")
print("-" * 40)

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])

print(f"Массив a:\n{a}")
print(f"\nМассив b:\n{b}")

print(f"\nВертикальная конкатенация (vstack):\n{np.vstack([a, b])}")
print(f"\nГоризонтальная конкатенация (hstack):\n{np.hstack([a, b.T])}")

# ========== 12 РАБОТА С ФАЙЛАМИ ==========
print("\n\n12. РАБОТА С ФАЙЛАМИ")
print("-" * 40)

# Сохранение и загрузка массивов
data_to_save = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# Сохраняем в текстовый файл
np.savetxt('data.txt', data_to_save, fmt='%.2f')
print("Массив сохранён в файл 'data.txt'")

# Загружаем из файла
loaded_data = np.loadtxt('data.txt')
print(f"\nЗагруженные данные:\n{loaded_data}")

# ========== ИТОГИ ==========
Темы:
1. Сравнение структур данных Python
2. Создание массивов NumPy
3. Атрибуты массивов (dtype, shape, ndim, size)
4. Индексация и срезы
5. Изменение формы массивов
6. Векторные операции
7. Агрегирующие функции
8. Булевы маски и фильтрация
9. Broadcasting
10.Конкатенация массивов
11.Работа с файлами