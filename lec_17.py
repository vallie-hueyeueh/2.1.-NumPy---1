"""
==============================================================================
ЛЕКЦИЯ 17 
==============================================================================
ТЕМЫ ЛЕКЦИИ:
  1. Деревья решений (Decision Trees) - углубленный разбор.
  2. Визуализация разделяющих поверхностей модели.
  3. Влияние гиперпараметра max_depth (глубина дерева) на переобучение.
  4. Подготовка к ансамблевым методам (Bagging) - как небольшое изменение 
     обучающей выборки радикально меняет структуру дерева.
  5. Ансамблевые методы (Ensembles): Бэггинг (Bagging).
  6. Случайный лес (Random Forest) - эволюция деревьев решений.
  7. Обучение без учителя: Метод главных компонент (PCA - Principal Component Analysis).
ОСНОВНЫЕ КОНЦЕПЦИИ:
  * Дерево решений — это алгоритм, который делит пространство признаков на
    прямоугольные области, задавая последовательные вопросы "больше/меньше".
  * Переобучение (Overfitting) — ситуация, когда алгоритм (особенно дерево)
    слишком сильно подстраивается под конкретные точки обучающей выборки, 
    запоминая даже шум. Модель получается сложной и плохо работает на новых данных.
  * Чтобы бороться с переобучением, используют ограничение глубины дерева 
    (max_depth) или объединяют много разных деревьев в "ансамбль" (Случайный лес).
    ТЕМЫ ЛЕКЦИИ:
  * Бэггинг (Bagging = Bootstrap Aggregating) — это техника, когда мы создаем 
    много одинаковых моделей (например, базовых деревьев), но каждую обучаем 
    на СЛУЧАЙНОМ кусочке данных. Затем их ответы усредняются (или идет голосование).
    Это резко снижает переобучение.
  * Случайный лес (Random Forest) — это продвинутый бэггинг над деревьями. 
    Он не только берет случайные данные для каждого дерева, но и случайные признаки.
  * PCA (Метод главных компонент) — позволяет найти оси (направления), вдоль 
    которых данные "вытянуты" сильнее всего (где дисперсия максимальна). 
    Полезно для сжатия данных и поиска скрытых закономерностей.
==============================================================================
"""
import seaborn as sns             
import matplotlib.pyplot as plt     
import numpy as np                 
import pandas as pd                 

# Импортируем модель "Дерево решений" для задачи классификации
from sklearn.tree import DecisionTreeClassifier

# Отключаем предупреждения pandas для чистоты вывода в консоль
import warnings
warnings.filterwarnings('ignore')

print("ШАГ 1: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")

# Загружаем классический набор данных "Ирисы Фишера"
# В нем 150 цветков, 3 вида (setosa, versicolor, virginica), по 50 каждого.
iris = sns.load_dataset("iris")

# --- ПРИМЕР: РУЧНАЯ КОДИРОВКА КАТЕГОРИЙ ---
# Алгоритмы машинного обучения не умеют читать текст ("setosa"). Им нужны числа.
# как перевести строки в числа с помощью конструкции match/case 

species_int = [] # Создаем пустой список для числовых меток

# Проходимся по каждой строке в данных
for row in iris.values:
    # row[4] - это пятая колонка (индексация с нуля), где лежит название вида
    match row[4]:
        case "setosa":
            species_int.append(1)     # Если setosa, добавляем 1
        case "versicolor":
            species_int.append(2)     # Если versicolor, добавляем 2
        case "virginica":
            species_int.append(3)     # Если virginica, добавляем 3

# Собираем новый датафрейм (таблицу), оставляя только два признака для наглядности (2D график)
# Берем длину чашелистика (sepal_length) и длину лепестка (petal_length)
data = iris[['sepal_length', 'petal_length']].copy()

# Добавляем новую колонку с числовыми метками
data['species'] = species_int


print("ШАГ 2: БАЗОВАЯ КЛАССИФИКАЦИЯ НА ДВУХ КЛАССАХ")
#работа дерева на двух классах (1 и 2)
# Фильтр данных: оставляем только те строки, где вид равен 1 (setosa) ИЛИ 2 (versicolor)
data_df = data[(data["species"] == 1) | (data["species"] == 2)]

# Разделяем данные на Признаки (X) и Ответы (y)
X = data_df[['sepal_length', 'petal_length']] # По этим данным модель будет учиться
y = data_df['species']                        # Это правильные ответы

# Выделим данные по каждому классу отдельно, чтобы раскрасить их на графике
data_of_setosa = data_df[data_df['species'] == 1]
data_of_versicolor = data_df[data_df['species'] == 2]

# Создаем и обучаем модель дерева решений (пока без ограничений гиперпараметров)
model = DecisionTreeClassifier()
model.fit(X, y) # Команда fit() заставляет модель найти закономерности


print("ШАГ 3: ВИЗУАЛИЗАЦИЯ РАЗДЕЛЯЮЩЕЙ ПОВЕРХНОСТИ")
# Чтобы нарисовать "фон" (зоны, как алгоритм делит плоскость), 
#нужно создать густую сетку точек (meshgrid) и попросить модель предсказать ответ для каждой.

# 1. Генерируем 100 равномерных точек от минимума до максимума по оси X (sepal_length)
x1_p = np.linspace(data_df['sepal_length'].min(), data_df['sepal_length'].max(), 100)
# 2. Генерируем 100 равномерных точек по оси Y (petal_length)
x2_p = np.linspace(data_df['petal_length'].min(), data_df['petal_length'].max(), 100)

# 3. Скрещиваем их, получая сетку 100x100 (10000 координат)
X1_p, X2_p = np.meshgrid(x1_p, x2_p)

# 4. Превращаем двумерную сетку в плоский список пар координат [X, Y], который "скушает" модель
# Функция ravel() вытягивает матрицу в одну длинную линию
X_p_array = np.vstack([X1_p.ravel(), X2_p.ravel()]).T

# Чтобы модель не ругалась на отсутствие имен колонок, обернем массив обратно в DataFrame
X_p = pd.DataFrame(X_p_array, columns=['sepal_length', 'petal_length'])

# 5. Делаем предсказание для каждой из 10000 точек фоновой сетки
y_p = model.predict(X_p)

# 6. Возвращаем предсказаниям форму сетки 100x100 для отрисовки
y_p_reshape = y_p.reshape(X1_p.shape)

# Рисуем график
plt.figure(figsize=(8, 6))
# contourf рисует закрашенные контуры (наш фон предсказаний)
# levels задают границы цветов (между 0, 1.5 и 2.5, чтобы отделить классы 1 и 2)
plt.contourf(X1_p, X2_p, y_p_reshape, alpha=0.3, levels=[0, 1.5, 2.5], cmap='Set2')

# Накладываем поверх фона реальные точки обучающей выборки
plt.scatter(data_of_setosa['sepal_length'], data_of_setosa['petal_length'], 
            color='blue', edgecolor='k', label='Setosa (1)')
plt.scatter(data_of_versicolor['sepal_length'], data_of_versicolor['petal_length'], 
            color='orange', edgecolor='k', label='Versicolor (2)')

plt.title("Дерево решений: Разделение двух классов")
plt.xlabel("Длина чашелистика (sepal_length)")
plt.ylabel("Длина лепестка (petal_length)")
plt.legend()
plt.show()


print("ШАГ 4: СИМУЛЯЦИЯ БЭГГИНГА (ВЛИЯНИЕ ВЫБОРКИ НА ДЕРЕВО)")
#суть проблемы Деревьев: 
# они ОЧЕНЬ чувствительны к малейшим изменениям в данных (высокая дисперсия).
# Чтобы это показать - делим данные одного класса на две разные половинки (А и В)
# и смотрим, как кардинально меняются решения алгоритма на разной глубине (max_depth).

# Выделяем все данные для класса 2 (versicolor) и 3 (virginica)
data_of_versicolor_all = data[data['species'] == 2]
data_of_virginica_all = data[data['species'] == 3]

# Искусственно разрезаем каждый класс пополам.
# iloc[:25] берем первые 25 цветков, iloc[25:] берем оставшиеся 25.
data_of_versicolor_A = data_of_versicolor_all.iloc[:25, :]
data_of_versicolor_B = data_of_versicolor_all.iloc[25:, :]

data_of_virginica_A = data_of_virginica_all.iloc[:25, :]
data_of_virginica_B = data_of_virginica_all.iloc[25:, :]

# Формируем два слегка отличающихся набора данных (Набор А и Набор B).
# ignore_index=True нужен, чтобы пересчитать индексы с нуля (иначе они ломаются).
data_df_A = pd.concat([data_of_virginica_A, data_of_versicolor_A], ignore_index=True)
data_df_B = pd.concat([data_of_virginica_B, data_of_versicolor_B], ignore_index=True)

# Задаем список глубин дерева, которые хотим проверить
max_depth_list = [1, 3, 5, 7]

# Создаем большую сетку графиков: 2 строки (для выборки А и B), 4 колонки (для разных max_depth)
fig, ax = plt.subplots(2, 4, figsize=(16, 8), sharex='col', sharey='row')
fig.suptitle("Сравнение Деревьев Решений на разных подвыборках (A и B) при разной максимальной глубине", fontsize=16)

# Внешний цикл: проходим по 4 вариантам глубины дерева
for j in range(4):
    md = max_depth_list[j] # Берем текущую глубину (1, 3, 5 или 7)
    
    # ---------------------------------------------------------
    # СТРОКА 0: Обучаем модель на Выборке "А"
    # ---------------------------------------------------------
    X_A = data_df_A[['sepal_length', 'petal_length']]
    y_A = data_df_A['species']
    
    model_A = DecisionTreeClassifier(max_depth=md, random_state=42)
    model_A.fit(X_A, y_A)
    
    # Сетка уже рассчитана ранее (X_p). Снова делаем предсказание для фона, но уже новой моделью
    y_p_A = model_A.predict(X_p)
    y_p_reshape_A = y_p_A.reshape(X1_p.shape)
    
    # Рисуем фон (выборка А)
    ax[0, j].contourf(X1_p, X2_p, y_p_reshape_A, alpha=0.3, levels=[0, 1.5, 2.5, 3.5], cmap='Paired')
    # Наносим точки выборки А
    ax[0, j].scatter(data_of_virginica_A['sepal_length'], data_of_virginica_A['petal_length'], color='green', edgecolor='k')
    ax[0, j].scatter(data_of_versicolor_A['sepal_length'], data_of_versicolor_A['petal_length'], color='orange', edgecolor='k')
    ax[0, j].set_title(f"Выборка А (Глубина {md})")

    # ---------------------------------------------------------
    # СТРОКА 1: Обучаем модель на Выборке "B"
    # ---------------------------------------------------------
    X_B = data_df_B[['sepal_length', 'petal_length']]
    y_B = data_df_B['species']
    
    model_B = DecisionTreeClassifier(max_depth=md, random_state=42)
    model_B.fit(X_B, y_B)
    
    y_p_B = model_B.predict(X_p)
    y_p_reshape_B = y_p_B.reshape(X1_p.shape)
    
    # Рисуем фон (выборка B)
    ax[1, j].contourf(X1_p, X2_p, y_p_reshape_B, alpha=0.3, levels=[0, 1.5, 2.5, 3.5], cmap='Paired')
    # Наносим точки выборки B
    ax[1, j].scatter(data_of_virginica_B['sepal_length'], data_of_virginica_B['petal_length'], color='green', edgecolor='k')
    ax[1, j].scatter(data_of_versicolor_B['sepal_length'], data_of_versicolor_B['petal_length'], color='orange', edgecolor='k')
    ax[1, j].set_title(f"Выборка B (Глубина {md})")

# Отрисовываем итоговую таблицу графиков
plt.tight_layout()
plt.show()

# ВЫВОД ИЗ ГРАФИКОВ:
# На маленькой глубине (max_depth=1) алгоритм недообучен, граница примитивная.
# На большой глубине (max_depth=5, 7) алгоритм переобучается: появляются странные "островки" и артефакты,
# которые пытаются идеально обогнуть каждую точку. 
# При этом на Выборке А и Выборке В эти островки абсолютно разные
# Это показывает нестабильность одиночных деревьев. Чтобы это исправить, в ML используют 
# "Случайный лес" (Random Forest), который усредняет результаты множества таких переобученных деревьев.

"""
часть 2

"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Импортируем модели для классификации
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

# Импортируем Метод главных компонент
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore') # Отключаем системные предупреждения

# ============================================================================
# ПОДГОТОВКА ДАННЫХ (Повторение из первой части для работоспособности кода)
# ============================================================================
iris = sns.load_dataset("iris")
species_int = []
for row in iris.values:
    match row[4]:
        case "setosa": species_int.append(1)
        case "versicolor": species_int.append(2)
        case "virginica": species_int.append(3)

data = iris[['sepal_length', 'petal_length']].copy()
data['species'] = species_int
# Для классификации берем только два класса (1 и 2)
data_df = data[(data["species"] == 1) | (data["species"] == 2)]
X = data_df[['sepal_length', 'petal_length']]
y = data_df['species']

# Создаем фон (сетку) для отрисовки графиков (100x100 точек)
x1_p = np.linspace(X['sepal_length'].min() - 0.5, X['sepal_length'].max() + 0.5, 100)
x2_p = np.linspace(X['petal_length'].min() - 0.5, X['petal_length'].max() + 0.5, 100)
X1_p, X2_p = np.meshgrid(x1_p, x2_p)
X_p = pd.DataFrame(np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=['sepal_length', 'petal_length'])


# ============================================================================
# ЧАСТЬ 1: АНСАМБЛИ (ОДИНОЧНОЕ ДЕРЕВО vs BAGGING vs СЛУЧАЙНЫЙ ЛЕС)
# ============================================================================
print("ОБУЧЕНИЕ АНСАМБЛЕЙ")

# 1. Одиночное дерево решений (Склонно к переобучению)
# Берем глубокое дерево (max_depth=6), чтобы оно максимально "подстроилось" под данные
model_tree = DecisionTreeClassifier(max_depth=6, random_state=42)
model_tree.fit(X, y)
y_pred_tree = model_tree.predict(X_p).reshape(X1_p.shape)

# 2. Бэггинг (BaggingClassifier)
# Мы берем 10 деревьев (n_estimators=10). 
# Каждое дерево увидит только 60% случайных данных (max_samples=0.6).
# Это заставит деревья быть РАЗНЫМИ. Итоговое решение принимается голосованием.
model_bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=6), # Базовая модель
    n_estimators=10,                               # Количество деревьев
    max_samples=0.6,                               # Доля выборки для каждого дерева
    random_state=1
)
model_bagging.fit(X, y)
y_pred_bagging = model_bagging.predict(X_p).reshape(X1_p.shape)

# 3. Случайный лес (RandomForestClassifier)
# Более оптимизированная и мощная версия Бэггинга. 
# Используется повсеместно в реальных задачах.
model_forest = RandomForestClassifier(
    n_estimators=10, 
    max_samples=0.6, 
    max_depth=3, # Ограничим глубину, чтобы лес был "проще" и обобщал лучше
    random_state=1
)
model_forest.fit(X, y)
y_pred_forest = model_forest.predict(X_p).reshape(X1_p.shape)


# --- ВИЗУАЛИЗАЦИЯ ТРЕХ МОДЕЛЕЙ ---
# Создаем картинку с 3 графиками в один ряд
fig, ax = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
fig.suptitle("Сравнение: Дерево решений -> Бэггинг -> Случайный лес", fontsize=16)

models_preds = [y_pred_tree, y_pred_bagging, y_pred_forest]
titles = ["1. Одиночное дерево (Переобучено)", "2. Бэггинг (Сглаживание)", "3. Случайный лес (Обобщение)"]

for i in range(3):
    # Рисуем разделяющий фон
    ax[i].contourf(X1_p, X2_p, models_preds[i], alpha=0.3, levels=[0, 1.5, 2.5], cmap='Paired')
    
    # Наносим точки двух классов
    ax[i].scatter(data_df[data_df['species']==1]['sepal_length'], 
                  data_df[data_df['species']==1]['petal_length'], label='Setosa')
    ax[i].scatter(data_df[data_df['species']==2]['sepal_length'], 
                  data_df[data_df['species']==2]['petal_length'], label='Versicolor')
    
    ax[i].set_title(titles[i])
    ax[i].set_xlabel("sepal_length")
    if i == 0: ax[i].set_ylabel("petal_length")

plt.legend()
plt.show()

# ВЫВОД ПО АНСАМБЛЯМ: Одиночное дерево строит сложные, угловатые границы (пытается 
# обойти каждую точку). Бэггинг и Случайный лес сглаживают эти границы, делая модель 
# более устойчивой к аномалиям (выбросам).


# ============================================================================
# ЧАСТЬ 2: МЕТОД ГЛАВНЫХ КОМПОНЕНТ (PCA)
# ============================================================================
print("\nМЕТОД ГЛАВНЫХ КОМПОНЕНТ (PCA)")
# Представить облако точек в виде эллипса (овала). PCA ищет:
# Главная компонента 1 - это ось, вдоль которой овал самый длинный.
# Главная компонента 2 - перпендикулярная ей ось (ширина овала).

# Возьмем данные только одного вида (setosa), чтобы посмотреть на форму их облака
data_setosa = data[data['species'] == 1][['sepal_length', 'petal_length']]

# Инициализируем PCA, просим найти 2 главные оси
pca = PCA(n_components=2)

# Обучаем модель ( мы передаем только X, правильных ответов "y" здесь нет 
# Это обучение без учителя, алгоритм просто изучает геометрию данных).
pca.fit(data_setosa)

# Извлекаем характеристики, которые нашел PCA:
# pca.mean_ - это центр (среднее арифметическое) облака точек.
# pca.components_ - вектора (направления) новых осей.
# pca.explained_variance_ - дисперсия (насколько сильно вытянуто облако) вдоль каждой оси.

print("Центр облака точек (x, y):", pca.mean_)
print("Направления осей (компоненты):\n", pca.components_)
print("Дисперсия по осям:\n", pca.explained_variance_)


# --- РИСУЕМ РЕЗУЛЬТАТ PCA ---
plt.figure(figsize=(8, 6))

# 1. Рисуем сами точки
plt.scatter(data_setosa['sepal_length'], data_setosa['petal_length'], alpha=0.6, label='Точки Setosa')

# 2. Рисуем центр облака точек (оранжевая точка)
mean_x, mean_y = pca.mean_[0], pca.mean_[1]
plt.scatter(mean_x, mean_y, color='orange', s=100, zorder=5, label='Центр масс (mean)')

# 3. Рисуем оси (Главные компоненты)
# Длина линии будет зависеть от того, насколько большая дисперсия (разброс) по этой оси.
# Для этого умножаем направление вектора на корень из дисперсии.
for length, vector in zip(pca.explained_variance_, pca.components_):
    # Корень из дисперсии - это стандартное отклонение (масштаб нашей линии)
    v = vector * np.sqrt(length) 
    
    # Рисуем линию от центра (mean_x, mean_y) до конца вектора
    plt.plot([mean_x, mean_x + v[0]], 
             [mean_y, mean_y + v[1]], 
             linewidth=3, color='black', label='Главная компонента')

# Убираем дубликаты из легенды (так как нарисовал две линии в цикле)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

# Делаем оси равного масштаба, чтобы перпендикулярные вектора визуально выглядели перпендикулярными
plt.axis('equal') 
plt.title("Метод Главных Компонент (PCA) на данных Setosa")
plt.xlabel("Длина чашелистика")
plt.ylabel("Длина лепестка")
plt.grid(True)
plt.show()

# ВЫВОД ПО PCA: Алгоритм нашел "скелет" данных. Самая длинная черная линия 
# показывает, что основное изменение формы цветков идет по диагонали (когда растет 
# длина чашелистика, растет и длина лепестка). Короткая линия показывает шум/отклонения.