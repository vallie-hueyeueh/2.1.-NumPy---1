"""
lec_05.py - Библиотека Pandas
- Series (создание, индексация, срезы, маскирование, операции)
- Index (свойства, операции над индексами)
- DataFrame (создание, доступ к строкам/столбцам, добавление столбцов,
  индексация loc/iloc, условное присваивание)
- Работа с пропущенными значениями (NaN) и fill_value
- Взаимодействие NumPy и Pandas
- Арифметические операции с выравниванием по индексам
"""
import numpy as np
import pandas as pd

# ============================================================================
# 1. Series – одномерная структура с метками
# ============================================================================

# 1.1. Создание Series из списка (индекс по умолчанию – от 0 до n-1)
data = pd.Series([0.25, 0.5, 0.75, 1.0])
print("\n1.1. Series из списка:")
print(data)
print("Тип объекта:", type(data))

# Атрибуты values (массив NumPy) и index
print("\nvalues:\n", data.values)
print("index:", data.index)
print("Тип индекса:", type(data.index))   # pandas.core.indexes.range.RangeIndex

# Доступ по позиции (используется целочисленный индекс)
print("\nЭлемент с позицией 1 (data[1]):", data[1])
print("Срез data[1:3] (позиции 1 и 2):")
print(data[1:3])

# 1.2. Series с собственными метками индекса
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
print("\n1.2. Series с пользовательским индексом:")
print(data)
print("Тип индекса первого элемента:", type(data.index[0]))   # str

# Доступ по метке и срез по меткам (включает правую границу!)
print("\ndata['a'] =", data['a'])
print("Срез data['b':'d'] (включительно):")
print(data['b':'d'])

# 1.3. Series из скаляра (все элементы равны указанному значению)
data_scalar = pd.Series(5, index=[10, 20, 30])
print("\n1.3. Series из скаляра (все =5):")
print(data_scalar)

# 1.4. Series из словаря (ключи становятся индексом)
d = {'A': 10, 'B': 20, 'C': 30, 'D': 40, 'E': 50}
data_dict = pd.Series(d)
print("\n1.4. Series из словаря:")
print(data_dict)

# Доступ по метке и срез по меткам
print("\ndata_dict['B'] =", data_dict['B'])
print("Срез data_dict['B':'D']:")
print(data_dict['B':'D'])

# 1.5. Series из словаря с частичным перечнем индексов (берутся только указанные ключи)
data_subset = pd.Series(d, index=['A', 'D'])
print("\n1.5. Series только с ключами ['A', 'D']:")
print(data_subset)

# 1.6. Изменение существующего элемента и добавление нового
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
data['a'] = 99
data['a1'] = 990        # новая метка добавляется автоматически
print("\n1.6. После изменения 'a' и добавления 'a1':")
print(data)

# Проверка наличия метки в индексе
print("\n'а' in data?", 'a' in data)
print("'a1' in data?", 'a1' in data)

# Преобразование в список пар (метка, значение)
print("list(data.items()):", list(data.items()))

# 1.7. Маскирование (фильтрация по условию)
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
print("\n1.7. Маскирование:")
print("data > 0.3:\n", data[data > 0.3])
print("(data > 0.3) & (data < 0.8):\n", data[(data > 0.3) & (data < 0.8)])

# Индексация по списку меток
print("\ndata[['c', 'a']] (порядок может быть произвольным):")
print(data[['c', 'a']])

# Индексация по позициям (устаревший способ, вызывает FutureWarning)
print("\nПопытка индексации по позициям data[[2,0]] (не рекомендуется):")
try:
    # Следующая строка выдаст предупреждение, но результат будет получен
    print(data[[2, 0]])
except Exception as e:
    print("Ошибка:", e)

# Правильный способ доступа по позициям – .iloc
print("\nПравильная индексация через .iloc[[2,0]]:")
print(data.iloc[[2, 0]])

# ============================================================================
# 2. Index – неизменяемый массив меток
# ============================================================================

idx = pd.Index([2, 5, 3, 5, 71])
print("\nСоздан индекс:", idx)
print("Тип:", type(idx))
print("Элемент по позиции 1 (idx[1]):", idx[1])
print("Срез idx[:2]:", idx[:2])
print("Срез idx[::2]:", idx[::2])

# Попытка изменить элемент (Index неизменяем)
try:
    idx[1] = 21
except TypeError as e:
    print("\nОшибка при попытке изменения индекса:", e)

# Операции над индексами
idx1 = pd.Index([2, 5, 3, 5, 71])
idx2 = pd.Index([1, 2, 51, 71, 4])

print("\nidx1:", idx1)
print("idx2:", idx2)
print("Пересечение idx1.intersection(idx2):", idx1.intersection(idx2))
print("Объединение idx1.union(idx2):", idx1.union(idx2))
print("Симметрическая разность idx1.symmetric_difference(idx2):",
      idx1.symmetric_difference(idx2))

# ============================================================================
# 3. DataFrame – двумерная таблица с метками строк и столбцов
# ============================================================================

# 3.1. Создание DataFrame из двух Series
dict1 = {'A': 10, 'B': 20, 'C': 30, 'D': 40, 'E': 50}
dict2 = {'A': 11, 'B': 21, 'C': 31, 'D': 41, 'E': 51}

data_dict1 = pd.Series(dict1)
data_dict2 = pd.Series(dict2)

df = pd.DataFrame({'dict_01': data_dict1, 'dict_02': data_dict2})
print("\n3.1. DataFrame из двух Series:")
print(df)

# Атрибуты
print("\nvalues (NumPy массив):\n", df.values)
print("Тип values:", type(df.values))
print("columns (имена столбцов):", df.columns)
print("index (метки строк):", df.index)

# Доступ к столбцу (возвращает Series)
print("\nСтолбец 'dict_01':")
print(df['dict_01'])
print("Тип столбца:", type(df['dict_01']))

# Доступ к строке по позиции
print("\nПервая строка через .iloc[0]:")
print(df.iloc[0])

# 3.2. Добавление новых столбцов
df['new'] = df['dict_01']                     # копия столбца
df['new1'] = df['dict_01'] / df['dict_02']    # поэлементное деление
print("\n3.2. После добавления столбцов 'new' и 'new1':")
print(df)

# 3.3. Индексация loc (по меткам) и iloc (по позициям)
print("\n3.3. loc['A':'C', :'dict_02'] (строки A-C, столбцы до dict_02 включительно):")
print(df.loc['A':'C', :'dict_02'])

print("\niloc[:3, :2] (первые 3 строки, первые 2 столбца):")
print(df.iloc[:3, :2])

# 3.4. Условный отбор строк и присваивание значений
print("\n3.4. Строки, где dict_02 > 30 (только столбцы 'new1' и 'dict_01'):")
print(df.loc[df['dict_02'] > 30, ['new1', 'dict_01']])

# Присваивание значения всем отобранным строкам в указанных столбцах
df.loc[df['dict_02'] > 30, ['new1', 'dict_01']] = 36
print("\nПосле присваивания 36 в эти ячейки:")
print(df)

# ============================================================================
# 4. Работа с пропущенными значениями (NaN)
# ============================================================================

# Расширим словари, чтобы появились несовпадающие индексы
dict1 = {'A': 10, 'B': 20, 'C': 30, 'D': 40, 'E': 50, 'F': 60}
dict2 = {'A': 11, 'B': 21, 'C': 31, 'D': 41, 'E': 51, 'H': 71}

data_dict1 = pd.Series(dict1)
data_dict2 = pd.Series(dict2)

df_nan = pd.DataFrame({'dict_01': data_dict1, 'dict_02': data_dict2})
print("\n4.1. DataFrame с несовпадающими индексами (появляются NaN):")
print(df_nan)

# Арифметические операции с NaN дают NaN, если один из операндов отсутствует
print("\nСложение Series (data_dict1 + data_dict2):")
print(data_dict1 + data_dict2)

# Метод add с fill_value – подставляет указанное значение вместо NaN перед операцией
print("\nСложение Series с fill_value=5 (data_dict1.add(data_dict2, fill_value=5)):")
print(data_dict1.add(data_dict2, fill_value=5))

# ============================================================================
# 5. Другие способы создания DataFrame
# ============================================================================

# 5.1. Из списка словарей (каждый словарь – отдельная строка)
df_list = pd.DataFrame([{'a': i, 'b': 2*i} for i in range(4)])
print("\n5.1. DataFrame из списка словарей:")
print(df_list)

# 5.2. Из NumPy массива случайных чисел
rng = np.random.default_rng(1)          # генератор с фиксированным зерном для воспроизводимости
A = rng.integers(10, size=(3, 4))
print("\n5.2. NumPy массив случайных целых (3x4):")
print(A)

# Вычитание из массива его первой строки (broadcasting)
print("\nA - A[0] (вычитание первой строки):")
print(A - A[0])

# Преобразование массива в DataFrame с именами столбцов
df_from_np = pd.DataFrame(A, columns=['A', 'B', 'C', 'D'])
print("\nDataFrame из NumPy массива:")
print(df_from_np)

# 5.3. Вычитание первой строки из DataFrame
print("\ndf - df.iloc[0]:")
print(df_from_np - df_from_np.iloc[0])

# Использование метода subtract с указанием оси (axis=0 – вычитание по столбцам)
print("\ndf.subtract(df['A'], axis=0) (вычитаем столбец 'A' из всех строк):")
print(df_from_np.subtract(df_from_np['A'], axis=0))

# ============================================================================
# 6. Сложение DataFrame с разными индексами и использование fill_value
# ============================================================================

# Создадим два DataFrame с перекрывающимися, но не идентичными индексами и столбцами
# Используем те же data_dict1, data_dict2 (содержат F и H)
df1 = pd.DataFrame({"dict_10": data_dict1, "dict_02": data_dict2})
print("\n6.1. Новый DataFrame df1:")
print(df1)

# Сложение df (из начала раздела 3) и df1 – индексы и столбцы выравниваются,
# на месте отсутствующих пар появляются NaN
print("\n6.2. df + df1 (без fill_value):")
print(df + df1)

# Сложение с fill_value – вместо NaN подставляется сумма всех элементов исходного df
total = df.values.sum()
print("\nСумма всех элементов df (df.values.sum()):", total)
print("df.add(df1, fill_value=total):")
print(df.add(df1, fill_value=total))

# ============================================================================
# 7. Дополнительные примеры с NumPy и Pandas
# ============================================================================

# 7.1. Series из случайных чисел
s = pd.Series(rng.integers(0, 10, 6))
print("\n7.1. Series из 6 случайных целых от 0 до 9:")
print(s)

# Применение универсальной функции NumPy (экспонента)
print("\nnp.exp(s):")
print(np.exp(s))

# 7.2. Случайный DataFrame 3x4
df_random = pd.DataFrame(rng.integers(0, 10, (3, 4)), columns=['A', 'B', 'C', 'D'])
print("\n7.2. Случайный DataFrame (3x4):")
print(df_random)

