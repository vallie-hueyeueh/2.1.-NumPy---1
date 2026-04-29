"""

ЛЕКЦИЯ 15

"""
# ============================================================================
# ТЕОРЕТИЧЕСКОЕ ВВЕДЕНИЕ
# ============================================================================
#
# ОБЫЧНОЕ ПРОГРАММИРОВАНИЕ:
#   Данные + Правила (алгоритм) → Ответы
#   Программист вручную пишет инструкции
#
# МАШИННОЕ ОБУЧЕНИЕ:
#   Данные + Ответы → Правила (модель)
#   Компьютер сам находит закономерности
#
# ВИДЫ МАШИННОГО ОБУЧЕНИЯ:
#
# 1) Обучение с учителем (Supervised Learning):
#    У нас есть пары "признаки → правильный ответ".
#    Задачи:
#      * Регрессия (Regression) – предсказать ЧИСЛО
#        Пример: по длине лепестка предсказать его ширину (непрерывная величина).
#      * Классификация (Classification) – предсказать КАТЕГОРИЮ (класс)
#        Пример: по размерам частей цветка определить вид ириса.
#
# 2) Обучение без учителя (Unsupervised Learning):
#    Правильных ответов нет, ищем скрытые структуры.
#    Задачи:
#      * Кластеризация (Clustering) – разбить объекты на похожие группы.
#      * Понижение размерности (Dimensionality Reduction) – сжать данные.
#    Методы: transform() – преобразование данных без использования меток.
#
# 3) Обучение с подкреплением (Reinforcement Learning):
#    Агент взаимодействует со средой и получает награду/штраф.
#
# СТАНДАРТНЫЕ ШАГИ РАБОТЫ С МОДЕЛЬЮ В SCIKIT-LEARN:
#    1. Выбирается класс модели (LinearRegression, LogisticRegression и т.д.)
#    2. Выбираются гиперпараметры модели (fit_intercept, max_depth, ...)
#    3. На основе данных создаётся матрица признаков (X) и целевой вектор (y).
#    4. Обучение модели методом fit(X, y).
#    5. Применение обученной модели:
#       5.1. Для supervised – predict(X_new)
#       5.2. Для unsupervised – transform(X) (и иногда predict)
#
# ТЕРМИНОЛОГИЯ:
#   Образец (sample) – одна строка данных (один цветок).
#   Признак (feature) – один столбец (например, длина чашелистика).
#   Целевая переменная (target, label) – то, что мы хотим предсказать.
#   Матрица признаков X – shape (n_samples, n_features)
#   Целевой вектор y – shape (n_samples,)
#   Модель – математическая функция с настраиваемыми параметрами.
#   Обучение – подгонка параметров модели под данные.
#   Гиперпараметры – настройки, которые задаются до обучения.

import seaborn as sns             
import matplotlib.pyplot as plt   
import numpy as np                
import pandas as pd                
# ----------------------------
# 1. ЗАГРУЗКА И ИЗУЧЕНИЕ ДАННЫХ
# ----------------------------
# датасет "Ирисы Фишера"
# В нём 150 цветков, по 50 каждого из трёх видов
# Признаки: sepal_length, sepal_width, petal_length, petal_width (в см)
# Целевая переменная: species (setosa, versicolor, virginica)
iris = sns.load_dataset("iris")

print("="*60)
print("Первые 5 строк датасета (head()):")
print(iris.head())
# DataFrame: строки – образцы, столбцы – признаки + target.

print("\nТип объекта iris:", type(iris))
# pandas.core.frame.DataFrame – таблица с подписями строк и столбцов.

print("Тип внутренних данных iris.values:", type(iris.values))
# numpy.ndarray – «голый» массив чисел и строк.

print("Форма массива iris.values.shape:", iris.values.shape)
# (150, 5) – 150 образцов, 5 столбцов (4 признака + 1 целевой).

print("Имена столбцов:\n", iris.columns)
# Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

print("Индексы строк (iris.index):", iris.index)
# RangeIndex(start=0, stop=150, step=1) – нумерация от 0 до 149.

# Попарный график рассеяния (scatter plot) для ВСЕХ признаков,
# точки раскрашены по виду ириса.
# sns.pairplot(iris, hue="species")
# plt.show()

# ----------------------------
# 1.1 ПОНЯТИЕ МАТРИЦЫ ПРИЗНАКОВ И ЦЕЛЕВОГО ВЕКТОРА
# ----------------------------
# X_iris – матрица признаков (только числа, без столбца species).
# Используем iris.drop('species', axis=1), чтобы убрать целевой столбец.
X_iris = iris.drop('species', axis=1)
print("\nПример матрицы признаков (первые 3 строки):")
print(X_iris.head(3))

# y_iris – целевой вектор (столбец species).
y_iris = iris['species']
print("\nПример целевого вектора (первые 10 значений):")
print(y_iris.head(10).to_list())

# внимание: для обучения с учителем X и y должны быть
# одинаковой длины по оси 0 (количество образцов).

# ============================================================================
# 2. РЕГРЕССИЯ. ЛИНЕЙНАЯ РЕГРЕССИЯ (Linear Regression)
#    ЗАДАЧА: по одному признаку (sepal length) предсказать другой (sepal width)
#    Только для цветов вида setosa (чтобы не смешивать разные формы).
# ============================================================================
# ШАГ 3: Подготовка X и y.
x = iris[iris["species"] == "setosa"].iloc[:, 0].to_numpy()  # 1-й столбец (sepal_length)
y = iris[iris["species"] == "setosa"].iloc[:, 1].to_numpy()  # 2-й столбец (sepal_width)
print("\nЛинейная регрессия:")
print("Первые 5 значений x (признак):", x[:5])
print("Первые 5 значений y (цель):   ", y[:5])

# ШАГ 1: Выбираем класс модели.
from sklearn.linear_model import LinearRegression

# ШАГ 2: Задаём гиперпараметры.
# fit_intercept=False → прямая проходит через начало координат (y = k*x).
model = LinearRegression(fit_intercept=False)

# ШАГ 4: Обучение (fit).
# В sklearn X должен быть двумерным: (n_samples, n_features).
# x сейчас одномерный (50,). Превращаем его в столбец через x[:, np.newaxis]
# (или x[:, None], что то же самое).
reg = model.fit(x[:, np.newaxis], y)  # reg и model указывают на один объект

print("Коэффициент линейной регрессии (coef_):", reg.coef_)
# Так как fit_intercept=False, intercept_ не вычисляется (равен 0).

# Визуализация исходных точек.
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.7, label='Данные (setosa)')
plt.xlabel('Длина чашелистика (sepal length), см')
plt.ylabel('Ширина чашелистика (sepal width), см')
plt.title('Линейная регрессия без свободного члена (y = k*x)')
plt.grid(True)

# ШАГ 5.1: Применяем модель – predict для плавной линии.
xfit = np.linspace(0, x.max(), 1000)          # 1000 равномерных точек от 0 до max(x)
yfit = model.predict(xfit[:, None])           # предсказанные значения y

plt.plot(xfit, yfit, "r", linewidth=2, label='Предсказание модели')
plt.legend()
plt.show()

# Дополнительная иллюстрация – можно вручную построить прямую по формуле:
# y = k*x + b, где k = reg.coef_, b = reg.intercept_.
# plt.plot(xfit, xfit * reg.coef_ + reg.intercept_, "k")
# (закомментировано, т.к. intercept_=0 при fit_intercept=False)

# ============================================================================
# 3. ПОЛИНОМИАЛЬНАЯ РЕГРЕССИЯ (Polynomial Regression)
#    Идея: добавить степени исходного признака, чтобы описать нелинейность.
# ============================================================================
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Шаг 1+2: создаём конвейер из двух шагов:
#   1) PolynomialFeatures(7) – генерирует признаки до 7-й степени включительно,
#      например [x, x^2, x^3, ..., x^7].
#   2) LinearRegression() – обычная линейная регрессия на расширенных признаках.
poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())

# Шаг 4: обучение на тех же x, y (setosa).
poly_model.fit(x[:, np.newaxis], y)

# Шаг 5.1: предсказание для плавной кривой.
xfit_poly = np.linspace(x.min(), x.max(), 1000)
yfit_poly = poly_model.predict(xfit_poly[:, None])

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.7, label='Данные (setosa)')
plt.plot(xfit_poly, yfit_poly, 'g-', linewidth=2, label='Полиномиальная регрессия (степень 7)')
plt.xlabel('Длина чашелистика, см')
plt.ylabel('Ширина чашелистика, см')
plt.title('Полиномиальная регрессия (степень 7)')
plt.legend()
plt.grid(True)
plt.show()

# ============================================================================
# 4. КЛАССИФИКАЦИЯ. ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ (Logistic Regression)
#    ЗАДАЧА: по длине чашелистика определить вид цветка (setosa или versicolor).
#    virginica исключаем, чтобы задача была бинарной (два класса).
# ============================================================================
# Сначала для наглядности визуализируем два класса на плоскости двух признаков.
setosa = iris[iris["species"] == "setosa"]
versicolor = iris[iris["species"] == "versicolor"]

x_0 = setosa.iloc[:, 0].to_numpy()      # sepal_length
y_0 = setosa.iloc[:, 1].to_numpy()      # sepal_width
x_1 = versicolor.iloc[:, 0].to_numpy()
y_1 = versicolor.iloc[:, 1].to_numpy()

plt.figure(figsize=(8, 6))
plt.scatter(x_0, y_0, color="red", alpha=0.5, label='setosa')
plt.scatter(x_1, y_1, color="green", alpha=0.5, label='versicolor')
plt.xlabel('Длина чашелистика, см')
plt.ylabel('Ширина чашелистика, см')
plt.title('Два класса в пространстве двух признаков')
plt.legend()
plt.grid(True)
plt.show()

# Для обучения логистической регрессии используем ОДИН признак (sepal length)
from sklearn.linear_model import LogisticRegression

# ШАГ 3: подготовка X и y.
X_clf = iris[iris["species"] != "virginica"].iloc[:, 0].to_numpy()  # 100 элементов
y_clf = iris[iris["species"] != "virginica"].iloc[:, 4]             # species (строка)
print("\nЛогистическая регрессия:")
print("X_clf shape:", X_clf.shape)   # (100,) – нужен столбец
print("y_clf shape:", y_clf.shape)   # (100,)
print("Классы в y_clf:", y_clf.unique())

# ШАГ 1+2: модель с параметрами по умолчанию (регуляризация L2)
logreg = LogisticRegression()

# ШАГ 4: обучение. X должен быть двумерным.
logreg.fit(X_clf[:, None], y_clf)

# ШАГ 5.1: predict_proba – возвращает вероятности для каждого класса.
# Порядок классов в logreg.classes_ (обычно алфавитный).
xfit_log = np.linspace(X_clf.min(), X_clf.max(), 1000)
y_proba = logreg.predict_proba(xfit_log[:, None])
print("Порядок классов в вероятностях:", logreg.classes_)

# График 1: совмещённая визуализация
# Искусственно разместим setosa на уровне y=1, versicolor на y=5,
# а вероятности масштабируем так: 1 + 4 * вероятность.
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
setosa_x = iris[iris["species"] == "setosa"].iloc[:, 0].to_numpy()
versicolor_x = iris[iris["species"] == "versicolor"].iloc[:, 0].to_numpy()
plt.scatter(setosa_x, np.full(50, 1), color="red", alpha=0.5, label='setosa')
plt.scatter(versicolor_x, np.full(50, 5), color="green", alpha=0.5, label='versicolor')
# y_proba[:, 0] – P(setosa), y_proba[:, 1] – P(versicolor)
plt.plot(xfit_log, 1 + 4 * y_proba[:, 0], 'red', linewidth=2, label='P(setosa) * 4 + 1')
plt.plot(xfit_log, 1 + 4 * y_proba[:, 1], 'green', linewidth=2, label='P(versicolor) * 4 + 1')
plt.xlabel('Длина чашелистика, см')
plt.ylabel('Искусственная координата')
plt.title('Логистическая регрессия (1 признак)')
plt.legend()
plt.grid(True)

# График 2: чистые вероятности.
plt.subplot(1, 2, 2)
plt.plot(xfit_log, y_proba[:, 0], 'red', linewidth=2, label='P(setosa)')
plt.plot(xfit_log, y_proba[:, 1], 'green', linewidth=2, label='P(versicolor)')
plt.xlabel('Длина чашелистика, см')
plt.ylabel('Вероятность')
plt.title('График вероятностей классов')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================================
# 5. ДЕРЕВЬЯ РЕШЕНИЙ (Decision Trees)
#    Классификация на основе ДВУХ признаков (sepal length и sepal width)
#    для двух классов (setosa, versicolor).
# ============================================================================
from sklearn.tree import DecisionTreeClassifier

# ШАГ 3: X – два столбца, y – числовые метки классов (1 и 2).
X_tree = iris[iris["species"] != "virginica"].iloc[:, 0:2].to_numpy()  # shape (100, 2)

# Целевой вектор создадим вручную: 50 единиц (setosa) и 50 двоек (versicolor).
y1 = np.full(50, 1)
y2 = np.full(50, 2)
y_tree = np.concatenate([y1, y2])  # можно было np.ravel([y1, y2]), результат тот же

print("\nДерево решений:")
print("Форма X_tree:", X_tree.shape)
print("Метки y_tree (первые 5 и с 50-й):", y_tree[:5], "...", y_tree[50:55])

# ШАГ 1+2: модель с ограничением максимальной глубины (чтобы избежать переобучения).
tree = DecisionTreeClassifier(max_depth=10)

# ШАГ 4: обучение.
tree.fit(X_tree, y_tree)

# Точность на обучающей выборке.
train_accuracy = tree.score(X_tree, y_tree)
print(f"Точность на обучении: {train_accuracy:.2f}")

# Визуализация решающей границы.
x_min, x_max = X_tree[:, 0].min() - 0.5, X_tree[:, 0].max() + 0.5
y_min, y_max = X_tree[:, 1].min() - 0.5, X_tree[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])  # np.c_ объединяет массивы по столбцам
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlGn)
plt.scatter(X_tree[y_tree == 1, 0], X_tree[y_tree == 1, 1],
            color='red', label='setosa (класс 1)')
plt.scatter(X_tree[y_tree == 2, 0], X_tree[y_tree == 2, 1],
            color='green', label='versicolor (класс 2)')
plt.xlabel('Длина чашелистика, см')
plt.ylabel('Ширина чашелистика, см')
plt.title('Дерево решений (max_depth=10) – решающая граница')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# ДОПОЛНИТЕЛЬНЫЙ ПРИМЕР: работа с np.c_
# ----------------------------
# np.c_ удобен для склеивания одномерных массивов в столбцы матрицы.
print("\nДемонстрация np.c_:")
print(np.c_[[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]])
# Результат:
# [[ 1 10]
#  [ 2 20]
#  [ 3 30]
#  [ 4 40]
#  [ 5 50]]