"""
ЛЕКЦИЯ: Библиотеки Python для научных расчетов и машинного обучения
Тема №2

"""


# ========== 1. СОЗДАНИЕ И БАЗОВЫЕ ОПЕРАЦИИ С МАССИВАМИ ==========

print("=" * 60)
print("1. СОЗДАНИЕ И БАЗОВЫЕ ОПЕРАЦИИ С МАССИВАМИ")
print("=" * 60)

# Создание массивов разными способами
print("\n1.1 Создание массивов:")
arr1 = np.array([1, 2, 3, 4, 5])
print(f"np.array([1,2,3,4,5]) = {arr1}")

arr2 = np.arange(10)
print(f"np.arange(10) = {arr2}")

arr3 = np.arange(2, 20, 3)  # от 2 до 20 с шагом 3
print(f"np.arange(2, 20, 3) = {arr3}")

arr4 = np.linspace(0, 1, 5)  # 5 чисел от 0 до 1 равномерно распределенных
print(f"np.linspace(0, 1, 5) = {arr4}")

# Специальные массивы
zeros_arr = np.zeros(5)
print(f"\nnp.zeros(5) = {zeros_arr}")

ones_arr = np.ones((3, 3))  # матрица 3x3 из единиц
print(f"np.ones((3, 3)) =\n{ones_arr}")

empty_arr = np.empty(4)  # массив без инициализации (может содержать "мусор")
print(f"np.empty(4) = {empty_arr}")

# ========== 2. СЛИЯНИЕ И РАЗБИЕНИЕ МАССИВОВ ==========

print("\n" + "=" * 60)
print("2. СЛИЯНИЕ И РАЗБИЕНИЕ МАССИВОВ")
print("=" * 60)

# Одномерные массивы
x = np.array([1, 2, 3])
y = np.array([4, 5])
z = np.array([6])
xyz = np.concatenate([x, y, z])
print(f"\n2.1 Конкатенация одномерных массивов: {xyz}")

# Двумерные массивы
print("\n2.2 Двумерные массивы:")
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6], [7, 8]])
print(f"Матрица A:\n{matrix_a}")
print(f"Матрица B:\n{matrix_b}")

# Вертикальное слияние (по строкам)
vstack_result = np.vstack([matrix_a, matrix_b])
print(f"\nВертикальное слияние (vstack):\n{vstack_result}")

# Горизонтальное слияние (по столбцам)
hstack_result = np.hstack([matrix_a, matrix_b])
print(f"Горизонтальное слияние (hstack):\n{hstack_result}")

# Разбиение массивов
print("\n2.3 Разбиение массивов:")
vertical_split = np.vsplit(vstack_result, [2])
print(f"Вертикальное разбиение (vsplit) на 2 части: {vertical_split}")

horizontal_split = np.hsplit(hstack_result, [2])
print(f"Горизонтальное разбиение (hsplit) на 2 части: {horizontal_split}")

# Трехмерные массивы
print("\n2.4 Трехмерные массивы:")
stack_3d = np.dstack([matrix_a, matrix_b])
print(f"Глубинное слияние (dstack):\n{stack_3d}")
print(f"Форма массива: {stack_3d.shape}")

deep_split = np.dsplit(stack_3d, [1])
print(f"Глубинное разбиение (dsplit): {deep_split}")

# ========== 3. ТРАНСЛИРОВАНИЕ (BROADCASTING) ==========

print("\n" + "=" * 60)
print("3. ТРАНСЛИРОВАНИЕ (BROADCASTING)")
print("=" * 60)

print("\n3.1 Транслирование скаляра к массиву:")
arr = np.array([1, 2, 3, 4, 5])
print(f"Массив: {arr}")
print(f"arr + 10 = {arr + 10}")  # 10 транслируется ко всем элементам
print(f"arr * 2 = {arr * 2}")
print(f"arr ** 2 = {arr ** 2}")

print("\n3.2 Транслирование вектора к матрице:")
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
vector = np.array([10, 20, 30])
print(f"Матрица:\n{matrix}")
print(f"Вектор: {vector}")
print(f"Матрица + Вектор:\n{matrix + vector}")

print("\n3.3 Транслирование с разными размерностями:")
# Матрица 3x1 + Вектор 1x3 = Матрица 3x3
col_vector = np.array([[1], [2], [3]])  # 3x1
row_vector = np.array([10, 20, 30])     # 1x3
print(f"Вектор-столбец (3x1):\n{col_vector}")
print(f"Вектор-строка (1x3): {row_vector}")
print(f"Транслирование 3x1 + 1x3 = 3x3:\n{col_vector + row_vector}")

# ========== 4. УНИВЕРСАЛЬНЫЕ ФУНКЦИИ (UFUNC) ==========

print("\n" + "=" * 60)
print("4. УНИВЕРСАЛЬНЫЕ ФУНКЦИИ (UFUNC)")
print("=" * 60)

print("\n4.1 Базовые арифметические операции:")
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])
print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")
print(f"b / a = {b / a}")
print(f"a ** 2 = {a ** 2}")

print("\n4.2 Агрегирующие функции:")
print(f"Сумма элементов a: {np.sum(a)}")
print(f"Среднее значение a: {np.mean(a)}")
print(f"Минимальное значение a: {np.min(a)}")
print(f"Максимальное значение a: {np.max(a)}")
print(f"Стандартное отклонение a: {np.std(a)}")
print(f"Медиана a: {np.median(a)}")

print("\n4.3 Тригонометрические функции:")
angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
print(f"Углы (радианы): {angles}")
print(f"sin(углы): {np.sin(angles)}")
print(f"cos(углы): {np.cos(angles)}")
print(f"tan(углы): {np.tan(angles)}")

print("\n4.4 Экспоненциальные и логарифмические функции:")
x = np.array([0, 0.0001, 0.001, 0.01, 0.1, 1, 10])
print(f"x = {x}")
print(f"exp(x) = {np.exp(x)}")
print(f"expm1(x) = exp(x)-1 = {np.expm1(x)}")  # Более точный для малых x
print(f"log(x) = {np.log(x)}")  # Будет предупреждение для log(0)
print(f"log2(x) = {np.log2(x)}")
print(f"log10(x) = {np.log10(x)}")

print("\n4.5 Комплексные числа:")
complex_arr = np.array([3 + 4j, 4 - 3j, -2 + 2j])
print(f"Комплексный массив: {complex_arr}")
print(f"Модуль (abs): {np.abs(complex_arr)}")
print(f"Угол (angle): {np.angle(complex_arr)}")
print(f"Действительная часть (real): {np.real(complex_arr)}")
print(f"Мнимая часть (imag): {np.imag(complex_arr)}")

# ========== 5. АГРЕГИРОВАНИЕ И СВОДНЫЕ ПОКАЗАТЕЛИ ==========

print("\n" + "=" * 60)
print("5. АГРЕГИРОВАНИЕ И СВОДНЫЕ ПОКАЗАТЕЛИ")
print("=" * 60)

x = np.arange(1, 6)  # [1, 2, 3, 4, 5]
print(f"\nИсходный массив x = {x}")

print("\n5.1 Суммирование:")
print(f"np.sum(x) = {np.sum(x)}")
print(f"np.add.reduce(x) = {np.add.reduce(x)}")  # То же что sum
print(f"np.cumsum(x) = {np.cumsum(x)}")  # Накопленная сумма

print("\n5.2 Произведение:")
print(f"np.prod(x) = {np.prod(x)}")
print(f"np.multiply.reduce(x) = {np.multiply.reduce(x)}")  # То же что prod
print(f"np.cumprod(x) = {np.cumprod(x)}")  # Накопленное произведение

print("\n5.3 Операции с аккумуляцией:")
print(f"np.add.accumulate(x) = {np.add.accumulate(x)}")  # Накопленная сумма
print(f"np.multiply.accumulate(x) = {np.multiply.accumulate(x)}")  # Накопленное произведение

print("\n5.4 Другие агрегирующие функции:")
matrix_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Матрица 3x3:\n{matrix_2d}")
print(f"Сумма по столбцам (axis=0): {np.sum(matrix_2d, axis=0)}")
print(f"Сумма по строкам (axis=1): {np.sum(matrix_2d, axis=1)}")
print(f"Общая сумма: {np.sum(matrix_2d)}")

# ========== 6. ВНЕШНИЕ ОПЕРАЦИИ (OUTER) ==========

print("\n" + "=" * 60)
print("6. ВНЕШНИЕ ОПЕРАЦИИ (OUTER)")
print("=" * 60)

x = np.arange(1, 6)
print(f"\nИсходный массив x = {x}")

print("\n6.1 Внешнее сложение:")
print("np.add.outer(x, x):")
print(np.add.outer(x, x))

print("\n6.2 Внешнее умножение:")
print("np.multiply.outer(x, x):")
print(np.multiply.outer(x, x))

print("\n6.3 Внешние операции с разными массивами:")
y = np.array([10, 20, 30])
print(f"x = {x}")
print(f"y = {y}")
print("Внешнее сложение x и y:")
print(np.add.outer(x, y))
print("Внешнее умножение x и y:")
print(np.multiply.outer(x, y))

# ========== 7. ОПЕРАЦИИ С ПАРАМЕТРОМ OUT ==========

print("\n" + "=" * 60)
print("7. ОПЕРАЦИИ С ПАРАМЕТРОМ OUT")
print("=" * 60)

print("\n7.1 Без использования out (создается новый массив):")
x = np.arange(5)
y = np.multiply(x, 10)
print(f"x = {x}")
print(f"y = np.multiply(x, 10) = {y}")

print("\n7.2 С использованием out (предотвращает создание нового массива):")
z = np.empty(5)  # Создаем пустой массив
np.multiply(x, 10, out=z)  # Результат записывается в z
print(f"z (после np.multiply с out) = {z}")

print("\n7.3 Использование out с частью массива:")
z = np.zeros(10)  # Массив из 10 нулей
print(f"Исходный z (10 нулей) = {z}")
np.multiply(x, 10, out=z[:5])  # Записываем результат в первые 5 элементов
print(f"z после записи в первые 5 элементов = {z}")

print("\n7.4 Использование out для экономии памяти:")
large_arr = np.arange(1000000)
result = np.empty_like(large_arr)
# Вместо: result = large_arr * 2 (создает новый массив)
np.multiply(large_arr, 2, out=result)  # Записывает в существующий массив
print(f"Умножение большого массива выполнено с использованием out")

# ========== 8. ИНДЕКСАЦИЯ И СРЕЗЫ ==========

print("\n" + "=" * 60)
print("8. ИНДЕКСАЦИЯ И СРЕЗЫ")
print("=" * 60)

arr = np.arange(10, 20)
print(f"\nИсходный массив: {arr}")

print("\n8.1 Базовая индексация:")
print(f"arr[0] = {arr[0]}")
print(f"arr[-1] = {arr[-1]}")
print(f"arr[3:7] = {arr[3:7]}")
print(f"arr[:5] = {arr[:5]}")
print(f"arr[5:] = {arr[5:]}")
print(f"arr[::2] = {arr[::2]}")  # Каждый второй элемент
print(f"arr[::-1] = {arr[::-1]}")  # Обратный порядок

print("\n8.2 Многомерная индексация:")
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Матрица 3x3:\n{matrix}")
print(f"matrix[0, 0] = {matrix[0, 0]}")
print(f"matrix[1, :] = {matrix[1, :]}")  # Вся вторая строка
print(f"matrix[:, 1] = {matrix[:, 1]}")  # Весь второй столбец
print(f"matrix[:2, :2] =\n{matrix[:2, :2]}")  # Подматрица 2x2

print("\n8.3 Булева индексация:")
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
mask = arr > 5
print(f"Массив: {arr}")
print(f"Маска (arr > 5): {mask}")
print(f"Элементы > 5: {arr[mask]}")
print(f"Элементы четные: {arr[arr % 2 == 0]}")

# ========== 9. РАБОТА СО СЛУЧАЙНЫМИ ЧИСЛАМИ ==========

print("\n" + "=" * 60)
print("9. РАБОТА СО СЛУЧАЙНЫМИ ЧИСЛАМИ")
print("=" * 60)

# Фиксируем seed для воспроизводимости
np.random.seed(42)

print("\n9.1 Равномерное распределение:")
uniform_arr = np.random.random(5)  # 5 чисел от 0 до 1
print(f"5 случайных чисел [0, 1): {uniform_arr}")

uniform_arr_100 = np.random.random(100)
print(f"Сумма 100 случайных чисел: {np.sum(uniform_arr_100):.4f}")
print(f"Среднее 100 случайных чисел: {np.mean(uniform_arr_100):.4f}")

print("\n9.2 Нормальное распределение:")
normal_arr = np.random.normal(0, 1, 10)  # mean=0, std=1, size=10
print(f"10 чисел из N(0,1): {normal_arr}")

print("\n9.3 Целочисленные случайные числа:")
int_arr = np.random.randint(0, 100, 10)  # 10 чисел от 0 до 99
print(f"10 случайных целых [0, 100): {int_arr}")

print("\n9.4 Выбор из массива:")
choices = np.array(['A', 'B', 'C', 'D'])
selected = np.random.choice(choices, 5)  # 5 выборов с возвращением
print(f"5 случайных выборов из {choices}: {selected}")

# ========== 10. ФОРМЫ И ПЕРЕФОРМИРОВАНИЕ МАССИВОВ ==========

print("\n" + "=" * 60)
print("10. ФОРМЫ И ПЕРЕФОРМИРОВАНИЕ МАССИВОВ")
print("=" * 60)

arr = np.arange(12)
print(f"\nИсходный массив (12 элементов): {arr}")

print("\n10.1 Изменение формы (reshape):")
reshaped = arr.reshape(3, 4)  # Преобразуем в матрицу 3x4
print(f"Матрица 3x4:\n{reshaped}")

print("\n10.2 Автоматическое определение размерности:")
auto_reshaped = arr.reshape(3, -1)  -1 означает "вычислить автоматически"
print(f"reshape(3, -1):\n{auto_reshaped}")

print("\n10.3 Сглаживание (flatten, ravel):")
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Матрица:\n{matrix}")
print(f"flatten(): {matrix.flatten()}")
print(f"ravel(): {matrix.ravel()}")

print("\n10.4 Транспонирование:")
print(f"Исходная матрица:\n{matrix}")
print(f"Транспонированная:\n{matrix.T}")

# ========== 11. ВЕКТОРИЗАЦИЯ И ПРОИЗВОДИТЕЛЬНОСТЬ ==========

print("\n" + "=" * 60)
print("11. ВЕКТОРИЗАЦИЯ И ПРОИЗВОДИТЕЛЬНОСТЬ")
print("=" * 60)

print("\n11.1 Сравнение производительности:")

# Создаем большой массив
large_array = np.random.random(1000000)

print("\nПример векторизованной операции (быстро):")
# Векторизованное сложение
result_vectorized = large_array + 10

print("\nПример невекторизованной операции (медленно):")
# Невекторизованное сложение через цикл
result_loop = np.empty_like(large_array)
for i in range(len(large_array)):
    result_loop[i] = large_array[i] + 10

print("Векторизованные операции выполняются на C-уровне и намного быстрее")

