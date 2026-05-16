"""
==============================================================================
ЛЕКЦИЯ 16
==============================================================================
ОБЫЧНОЕ ПРОГРАММИРОВАНИЕ:
  Данные + Правила (алгоритм) → Ответы
  Программист вручную пишет инструкции

МАШИННОЕ ОБУЧЕНИЕ:
  Данные + Ответы → Правила (модель)
  Компьютер сам находит закономерности

ВИДЫ МАШИННОГО ОБУЧЕНИЯ:
1) Обучение с учителем (Supervised Learning):
   У нас есть пары "признаки → правильный ответ".
   * Регрессия (Regression) – предсказать ЧИСЛО (непрерывную величину).
   * Классификация (Classification) – предсказать КАТЕГОРИЮ (класс).

2) Обучение без учителя (Unsupervised Learning):
   Правильных ответов нет, ищем скрытые структуры.
   * Кластеризация (Clustering) – разбить объекты на группы.
   * Понижение размерности (Dimensionality Reduction) – сжать данные.

3) Обучение с подкреплением (Reinforcement Learning):
   Агент взаимодействует со средой и получает награду/штраф.

СТАНДАРТНЫЕ ШАГИ РАБОТЫ В SCIKIT-LEARN:
   1. Выбор класса модели (LinearRegression, SVC и т.д.)
   2. Выбор гиперпараметров (настроек до обучения)
   3. Подготовка матрицы признаков (X) и целевого вектора (y)
   4. Обучение модели: fit(X, y)
   5. Применение модели: predict(X_new)

АЛГОРИТМЫ КЛАССИФИКАЦИИ:
------------------------------------------------------------------------------
* Логистическая регрессия (Logistic Regression): 
  В основе лежит сигмоидная кривая, выдающая вероятность от 0 до 1. 
  Уравнение сигмоиды: $y = \frac{1}{1 + e^{-(mx+b)}}$

* Деревья решений (Decision Trees): 
  Непараметрический метод. Дробит пространство признаков, задавая 
  последовательные вопросы "да/нет" (например, с помощью коэффициента Джини).

* Метод опорных векторов (SVM): 
  Ищет линию (или гиперплоскость), разделяющую классы так, чтобы 
  зазор между крайними точками (опорными векторами) был максимальным. 
  При использовании ядра RBF переводит данные в многомерное пространство.

* Наивный Байес (Naive Bayes): 
  Генеративная модель на основе теоремы Байеса: 
  $P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$. 
  Хорошо работает с нормально распределенными (гауссовыми) данными.
==============================================================================
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Отключаем предупреждения sklearn для чистоты вывода
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. ЗАГРУЗКА И ИЗУЧЕНИЕ ДАННЫХ
# ============================================================================
print("--- 1. ЗАГРУЗКА ДАННЫХ ---")
iris = sns.load_dataset("iris")

print("Первые 5 строк датасета Ирисы Фишера:")
print(iris.head())
print(f"Форма данных: {iris.shape} (150 образцов, 4 признака + 1 целевая переменная)")

# Попарный график рассеяния 
# sns.pairplot(iris, hue="species")
# plt.show()


# ============================================================================
# 2. РЕГРЕССИЯ (ЛИНЕЙНАЯ И ПОЛИНОМИАЛЬНАЯ)
# Задача: по длине чашелистика (sepal_length) предсказать его ширину (sepal_width).
# Берем только вид 'setosa'.
# ============================================================================
print("\n--- 2. РЕГРЕССИЯ ---")
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

x_reg = iris[iris["species"] == "setosa"].iloc[:, 0].to_numpy() # sepal_length
y_reg = iris[iris["species"] == "setosa"].iloc[:, 1].to_numpy() # sepal_width

# Модель 1: Линейная регрессия без свободного члена (прямая через ноль)
model_lin = LinearRegression(fit_intercept=False)
model_lin.fit(x_reg[:, np.newaxis], y_reg)

# Модель 2: Полиномиальная регрессия (степень 7)
model_poly = make_pipeline(PolynomialFeatures(7), LinearRegression())
model_poly.fit(x_reg[:, np.newaxis], y_reg)

# Генерация точек для плавных линий
xfit_reg = np.linspace(0 if model_lin.fit_intercept==False else x_reg.min(), x_reg.max(), 1000)
yfit_lin = model_lin.predict(xfit_reg[:, np.newaxis])

xfit_poly = np.linspace(x_reg.min(), x_reg.max(), 1000)
yfit_poly = model_poly.predict(xfit_poly[:, np.newaxis])

plt.figure(figsize=(10, 5))
plt.scatter(x_reg, y_reg, alpha=0.7, label='Данные (setosa)')
plt.plot(xfit_reg, yfit_lin, "r-", linewidth=2, label='Линейная (без сдвига)')
plt.plot(xfit_poly, yfit_poly, "g-", linewidth=2, label='Полиномиальная (ст. 7)')
plt.xlabel('Длина чашелистика, см')
plt.ylabel('Ширина чашелистика, см')
plt.title('Регрессионные модели')
plt.legend()
plt.grid(True)
plt.show()


# ============================================================================
# 3. КЛАССИФИКАЦИЯ: ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ (1 ПРИЗНАК)
# Задача: по 1 признаку определить класс (setosa vs versicolor).
# Отрисовка сигмоиды вероятностей.
# ============================================================================
print("\n--- 3. ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ (ВЕРОЯТНОСТИ) ---")
from sklearn.linear_model import LogisticRegression

# Берем 100 образцов (без virginica), 1 признак (sepal_length)
X_log1 = iris[iris["species"] != "virginica"].iloc[:, 0].to_numpy()
y_log1 = iris[iris["species"] != "virginica"].iloc[:, 4]

logreg = LogisticRegression()
logreg.fit(X_log1[:, np.newaxis], y_log1)

xfit_log1 = np.linspace(X_log1.min(), X_log1.max(), 1000)
y_proba = logreg.predict_proba(xfit_log1[:, np.newaxis])

plt.figure(figsize=(8, 5))
plt.plot(xfit_log1, y_proba[:, 0], 'red', linewidth=2, label='P(setosa)')
plt.plot(xfit_log1, y_proba[:, 1], 'green', linewidth=2, label='P(versicolor)')
plt.scatter(X_log1[y_log1 == 'setosa'], np.zeros(50), color='red', alpha=0.5, label='Фактически setosa')
plt.scatter(X_log1[y_log1 == 'versicolor'], np.ones(50), color='green', alpha=0.5, label='Фактически versicolor')
plt.xlabel('Длина чашелистика, см')
plt.ylabel('Вероятность (Сигмоида)')
plt.title('Вероятности Логистической Регрессии')
plt.legend()
plt.grid(True)
plt.show()


# ============================================================================
# 4. ПОДГОТОВКА К 2D ВИЗУАЛИЗАЦИИ ГРАНИЦ КЛАССИФИКАТОРОВ
# ============================================================================
# Теперь используем  2 признака, чтобы рисовать области на плоскос
X_2d = iris.iloc[:100, [0, 1]].to_numpy()
# Конвертируем строки 'setosa'/'versicolor' в числа 1 и 2 для контурного графика
y_2d = np.where(iris.iloc[:100, 4] == 'setosa', 1, 2)

# Создание сетки (meshgrid) для фоновой раскраски
x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

def plot_decision_boundary(model, title):
    """Вспомогательная функция для обучения и отрисовки решающих границ 2D"""
    model.fit(X_2d, y_2d)
    
    # Предсказание для каждой точки координатной сетки
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlGn)
    
    plt.scatter(X_2d[y_2d == 1, 0], X_2d[y_2d == 1, 1], color='red', edgecolor='k', label='setosa')
    plt.scatter(X_2d[y_2d == 2, 0], X_2d[y_2d == 2, 1], color='green', edgecolor='k', label='versicolor')
    
    plt.xlabel('Длина чашелистика, см')
    plt.ylabel('Ширина чашелистика, см')
    plt.title(title)
    plt.legend()
    plt.show()


# ============================================================================
# 5. ДЕРЕВО РЕШЕНИЙ (Decision Tree)
# ============================================================================
print("\n--- 5. ДЕРЕВО РЕШЕНИЙ ---")
from sklearn.tree import DecisionTreeClassifier

# max_depth=3 ограничивает количество вопросов, делая границу ступенчатой, но менее "переобученной"
tree_clf = DecisionTreeClassifier(max_depth=3)
plot_decision_boundary(tree_clf, "Дерево решений (глубина 3)")


# ============================================================================
# 6. МЕТОД ОПОРНЫХ ВЕКТОРОВ (SVM)
# ============================================================================
print("\n--- 6. МЕТОД ОПОРНЫХ ВЕКТОРОВ (SVM) ---")
from sklearn.svm import SVC

# Линейное ядро (ищет прямую линию)
svm_lin = SVC(kernel='linear', C=1.0)
plot_decision_boundary(svm_lin, "SVM (Линейное ядро)")

# Нелинейное ядро (Радиальная базисная функция - RBF) - строит плавные "островки"
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
plot_decision_boundary(svm_rbf, "SVM (Ядро RBF - нелинейное)")


# ============================================================================
# 7. НАИВНЫЙ БАЙЕСОВСКИЙ КЛАССИФИКАТОР (Naive Bayes)
# ============================================================================
print("\n--- 7. НАИВНЫЙ БАЙЕС ---")
from sklearn.naive_bayes import GaussianNB

# Строит вероятностные гауссовы распределения (колоколы) для каждого класса
bayes_clf = GaussianNB()
plot_decision_boundary(bayes_clf, "Наивный Байес (GaussianNB)")


# ============================================================================
# 8. МЕТОД К-БЛИЖАЙШИХ СОСЕДЕЙ (KNN)
# ============================================================================
print("\n--- 8. МЕТОД K-БЛИЖАЙШИХ СОСЕДЕЙ ---")
from sklearn.neighbors import KNeighborsClassifier

# Смотрит на 5 ближайших соседей для классификации точки
knn_clf = KNeighborsClassifier(n_neighbors=5)
plot_decision_boundary(knn_clf, "K-Ближайших соседей (k=5)")