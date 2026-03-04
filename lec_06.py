"""
================================================================================
ЛЕКЦИЯ 6: ПРОПУЩЕННЫЕ ДАННЫЕ И ИЕРАРХИЧЕСКАЯ ИНДЕКСАЦИЯ В PANDAS
================================================================================
Содержание:
1. Представление отсутствующих данных в NumPy и pandas
   - NaN, None, pd.NA
   - Влияние на типы данных
   - Проверка на пропуски (isnull, notnull)
   - Удаление пропусков (dropna) и заполнение (fillna, ffill, bfill)
2. Иерархическая индексация (MultiIndex)
   - Способы создания MultiIndex
   - Индексация в Series и DataFrame с MultiIndex
   - Срезы с pd.IndexSlice
   - Переиндексация (reindex)
   - Преобразование между wide и long форматами (stack / unstack)
   - Работа с уровнями индекса (swaplevel, sort_index)
   - Агрегация с groupby по уровням
3. Примеры совместного использования – DataFrame с MultiIndex по строкам и столбцам
   - Генерация случайных данных
   - Выборки и операции
================================================================================
"""

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# 1. ОТСУТСТВУЮЩИЕ ДАННЫЕ
# ------------------------------------------------------------------------------

# 1.1. NaN в NumPy
# -----------------
# NaN (Not a Number) – специальное значение с плавающей точкой, обозначающее пропуск.
# Операции с NaN всегда дают NaN.
print("\n--- NumPy: NaN ---")
a = np.array([1, 2, 3])
print(f"a.sum() = {a.sum()}")                     # 6

# Попытка использовать None в массиве NumPy приводит к объектному типу
b = np.array([1, None, 3])
print(f"b = {b}, dtype = {b.dtype}")              # object
# b.sum() -> TypeError, так как None нельзя сложить с int

c = np.array([1, np.nan, 3])
print(f"c = {c}")                                  # [ 1. nan  3.]
print(f"c.sum() = {c.sum()}")                      # nan
print(f"np.sum(c) = {np.sum(c)}")                  # nan
print(f"np.nansum(c) = {np.nansum(c)}")            # 4.0 (игнорирует NaN)

# Арифметика с NaN
print(f"1 + np.nan = {1 + np.nan}")                # nan
print(f"1 * np.nan = {1 * np.nan}")                # nan

# 1.2. None и NaN в pandas.Series
# --------------------------------
# pandas автоматически преобразует типы при вставке пропусков.
print("\n--- pandas.Series: пропуски и типы ---")

# Целочисленная серия
s_int = pd.Series([1, 2, 3, 4, 5])
print("s_int (исходная):\n", s_int)

# Присваиваем None – серия становится float64 (так как NaN – это float)
s_int[0] = None
s_int[1] = np.nan
print("s_int после замены:\n", s_int)               # dtype: float64

# Серия строк
s_str = pd.Series(["1", "2", "3", "4", "5"])
s_str[0] = None
s_str[1] = np.nan
print("s_str (object):\n", s_str)                   # dtype: object

# Булева серия – при вставке None/NaN становится object
s_bool = pd.Series([True, False, True])
s_bool[0] = None
s_bool[1] = np.nan
print("s_bool (boolean -> object):\n", s_bool)

# 1.3. Использование pd.NA и nullable-типов
# ------------------------------------------
# pandas предоставляет собственное значение pd.NA, которое может использоваться
# в целочисленных, булевых и строковых типах без потери исходного типа.
print("\n--- pd.NA и nullable-типы ---")

# Целочисленный nullable-тип 'Int64' (обратите внимание на заглавную I)
s_int_na = pd.Series([1, np.nan, None, 5, pd.NA], dtype="Int64")
print("s_int_na (Int64):\n", s_int_na)
print("s_int_na.isnull():\n", s_int_na.isnull())

# Числовой nullable-тип с плавающей точкой 'Float64'
s_float_na = pd.Series([1, np.nan, None, 5, pd.NA], dtype="Float64")
print("s_float_na (Float64):\n", s_float_na)

# Булев nullable-тип 'boolean'
s_bool_na = pd.Series([True, False, None, pd.NA], dtype="boolean")
print("s_bool_na (boolean):\n", s_bool_na)

# Строковой nullable-тип 'string' (отличается от object)
s_str_na = pd.Series(["a", "b", None, pd.NA], dtype="string")
print("s_str_na (string):\n", s_str_na)

# 1.4. Работа с пропусками в DataFrame
# -------------------------------------
print("\n--- DataFrame: удаление и заполнение пропусков ---")

# Создадим DataFrame с пропусками
df1 = pd.DataFrame([[1, np.nan, None],
                    [1, 2, 3],
                    [2, np.nan, 3]])
print("Исходный df1:\n", df1)

# Удаление строк с хотя бы одним пропуском (how='any' по умолчанию)
print("df1.dropna(axis=0):\n", df1.dropna(axis=0))

# Удаление столбцов с пропусками
print("df1.dropna(axis=1):\n", df1.dropna(axis=1))

# how='all' – удаляем только те строки, где все значения пропущены
df2 = pd.DataFrame([[None, np.nan, None],
                    [1, 2, 3],
                    [2, np.nan, 3]])
print("df2.dropna(how='all'):\n", df2.dropna(how='all'))

# thresh – оставляем строки, где не менее N не-NaN значений
print("df2.dropna(thresh=2):\n", df2.dropna(thresh=2))

# Заполнение пропусков фиксированным значением
print("df1.fillna(-4):\n", df1.fillna(-4))

# Заполнение предыдущим (ffill) и следующим (bfill) значениями
df3 = pd.DataFrame([[None, np.nan, None],
                    [1, 2, 3],
                    [5, np.nan, 3]])
print("Исходный df3:\n", df3)

# По строкам (axis=1) – заполняем слева направо
print("ffill по строкам:\n", df3.ffill(axis=1))
print("bfill по строкам:\n", df3.bfill(axis=1))

# По столбцам (axis=0) – заполняем сверху вниз
print("ffill по столбцам:\n", df3.ffill(axis=0))
print("bfill по столбцам:\n", df3.bfill(axis=0))

# Комбинированные методы: можно сначала ffill, потом bfill
print("ffill затем bfill:\n", df3.ffill().bfill())

# ------------------------------------------------------------------------------
# 2. ИЕРАРХИЧЕСКАЯ ИНДЕКСАЦИЯ (MULTIINDEX)
# ------------------------------------------------------------------------------

# 2.1. Способы создания MultiIndex
# ---------------------------------
print("\n--- Создание MultiIndex ---")

# Способ 1: из списка кортежей (from_tuples)
tuples = [("A1", 2025), ("A1", 2026),
          ("A2", 2025), ("A2", 2026),
          ("A3", 2025), ("A3", 2026)]
mi_tuples = pd.MultiIndex.from_tuples(tuples, names=["shop", "year"])
print("from_tuples:\n", mi_tuples)

# Способ 2: из массивов (from_arrays)
arrays = [["A1", "A1", "A2", "A2"], [2025, 2026, 2025, 2026]]
mi_arrays = pd.MultiIndex.from_arrays(arrays, names=["shop", "year"])
print("from_arrays:\n", mi_arrays)

# Способ 3: декартово произведение (from_product) – наиболее удобен для комбинаторных индексов
mi_product = pd.MultiIndex.from_product([["A1", "A2"], [2025, 2026]],
                                        names=["property", "year"])
print("from_product:\n", mi_product)

# Способ 4: явное указание уровней и кодов (низкоуровневый, но гибкий)
mi_explicit = pd.MultiIndex(levels=[["A1", "A2"], [2025, 2026]],
                            codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
                            names=["property", "year"])
print("явное задание levels/codes:\n", mi_explicit)

# 2.2. Series с MultiIndex
# -------------------------
print("\n--- Series с иерархическим индексом ---")

# Создадим данные для трёх магазинов, двух лет и двух месяцев (1 и 2)
index_tuples = [("A1", 2025, 1), ("A1", 2025, 2),
                ("A1", 2026, 1), ("A1", 2026, 2),
                ("A2", 2025, 1), ("A2", 2025, 2),
                ("A2", 2026, 1), ("A2", 2026, 2),
                ("A3", 2025, 1), ("A3", 2025, 2),
                ("A3", 2026, 1), ("A3", 2026, 2)]
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

s = pd.Series(data, index=pd.MultiIndex.from_tuples(index_tuples,
                                                     names=["shop", "year", "month"]))
print("Исходная Series:\n", s)

# Индексация по уровням
print("\n--- Индексация ---")
print("s['A1'] (первый уровень):\n", s["A1"])
print("s[:, 2025] (второй уровень):\n", s[:, 2025])
print("s['A1', 2025, 1] (конкретный элемент):\n", s["A1", 2025, 1])

# Использование pd.IndexSlice для сложных срезов
idx = pd.IndexSlice
print("s.loc[idx['A1':'A2', 2025, 1]]:\n", s.loc[idx["A1":"A2", 2025, 1]])

# Переиндексация (reindex) – расширение индекса, добавление пропусков
full_mi = pd.MultiIndex.from_product([["A1", "A2", "A3"], [2025, 2026], [1, 2]],
                                      names=["shop", "year", "month"])
s_reindexed = s.reindex(full_mi)
print("После reindex (появились NaN):\n", s_reindexed)

# Удаление пропусков после reindex
print("s_reindexed.dropna():\n", s_reindexed.dropna())

# Превращение Series в DataFrame через unstack
# Уровень month уходит в столбцы
df_unstacked = s.unstack(level="month")
print("unstack (month -> columns):\n", df_unstacked)

# Можно unstack несколько уровней сразу
df_unstacked_both = s.unstack(level=["year", "month"])
print("unstack year и month:\n", df_unstacked_both)

# Обратная операция stack – из DataFrame в Series
s_stacked = df_unstacked.stack()
print("stack обратно в Series:\n", s_stacked)

# Работа с уровнями: swaplevel, sort_index
s_swapped = s.swaplevel(0, 1)   # поменять местами shop и year
print("swaplevel(0,1):\n", s_swapped)
print("sort_index(level='year'):\n", s_swapped.sort_index(level='year'))

# 2.3. DataFrame с MultiIndex
# ----------------------------
print("\n--- DataFrame с иерархическим индексом ---")

# Используем Series s для создания DataFrame с колонками jan, feb, mar
df_multi = pd.DataFrame({
    "jan": s.loc[:, :, 1],          # все магазины и годы, месяц 1
    "feb": s.loc[:, :, 2],          # месяц 2
    "mar": s.loc[:, :, 1] + s.loc[:, :, 2]   # сумма января и февраля
})
print("DataFrame из Series:\n", df_multi)

# Индексация по уровням в DataFrame
print("Столбец 'mar':\n", df_multi["mar"])
print("Строка для магазина A1:\n", df_multi.loc["A1"])
print("Конкретный элемент (A1, 2025, 'feb'):\n", df_multi.loc[("A1", 2025), "feb"])

# Выбор нескольких строк и столбцов с помощью списков
print("A1 и A2, столбцы feb и jan:\n",
      df_multi.loc[["A1", "A2"], ["feb", "jan"]])

# Индексация по iloc (позиционная)
print("iloc: строки 1 и 0, столбцы 1 и 1 (повтор):\n",
      df_multi.iloc[[1, 0], [1, 1]])

# 2.4. Агрегация с groupby по уровням MultiIndex
# -----------------------------------------------
print("\n--- groupby по уровням ---")

# Сумма по магазинам (уровень 0)
print("Сумма по магазинам:\n", s.groupby(level=0).sum())

# Сумма по годам (уровень 1)
print("Сумма по годам:\n", s.groupby(level="year").sum())

# Сумма по магазинам и годам (несколько уровней)
print("Сумма по магазинам и годам:\n", s.groupby(level=[0, 1]).sum())

# Среднее по месяцам (уровень 2)
print("Среднее по месяцам:\n", s.groupby(level="month").mean())

# ------------------------------------------------------------------------------
# 3. ПРИМЕР: СЛУЧАЙНЫЕ ДАННЫЕ С MULTIINDEX ПО СТРОКАМ И СТОЛБЦАМ
# ------------------------------------------------------------------------------

# Генератор случайных чисел для воспроизводимости
rng = np.random.default_rng(1)

# Индексы
mi_rows = pd.MultiIndex.from_product([["A1", "A2"], [2025, 2026]],
                                      names=["property", "year"])
mi_cols = pd.MultiIndex.from_product([["B1", "B2", "B3"], ["jan", "feb"]],
                                      names=["shop", "month"])

# Случайные данные (4 строки, 6 столбцов)
data = rng.random((4, 6))
print("Сгенерированные данные (4x6):\n", data)

# Создаём DataFrame
df_random = pd.DataFrame(data, index=mi_rows, columns=mi_cols)
print("Случайный DataFrame:\n", df_random)

# Работа с этим DataFrame – примеры
print("\n--- Выборки ---")
print("Выбор строки A1:\n", df_random.loc["A1"])
print("Выбор столбца 'B1' (все подстолбцы):\n", df_random["B1"])
print("Выбор конкретного столбца ('B1', 'jan'):\n", df_random[("B1", "jan")])
print("Срез по строкам и столбцам с помощью loc:\n",
      df_random.loc[("A1", 2025):("A2", 2025), ("B1", "jan"):("B2", "feb")])

# Транспонирование
print("Транспонированный DataFrame:\n", df_random.T)

# Агрегация по строкам (среднее по магазинам)
print("Среднее по строкам (уровень property):\n",
      df_random.groupby(level="property").mean())

# Агрегация по столбцам (сумма по месяцам)
print("Сумма по столбцам (уровень month):\n",
      df_random.groupby(level="month", axis=1).sum())
