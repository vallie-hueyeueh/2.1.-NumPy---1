import matplotlib.pyplot as plt
import numpy as np


plt.figure(figsize=(8, 5))
x = [1.25, 5.0, 10.0, 15.0, 20.0]
y = [1, 7, 3, 5, 11]

plt.plot(x, y, color='red', marker='o') 

plt.grid(True, linestyle=':', color='grey') 

plt.title("Простой график ")
plt.show()



plt.figure(figsize=(8, 5))
x_two = [1.25, 5.0, 10.0, 15.0, 20.0]
y1_two = [1, 7, 3, 5, 11]
y2_two = [4, 3, 1, 8, 12]
plt.plot(x_two, y1_two, color='red', marker='o', linestyle='-', label='line 1')
plt.plot(x_two, y2_two, color='green', marker='o', linestyle='-.', label='line 1')
plt.legend()
plt.title("Две линии разного стиля")
plt.show()



fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(2, 1, 1) 
ax1.plot([1, 2, 3, 4, 5], [1, 7, 6, 3, 5])
ax2 = fig.add_subplot(2, 2, 3)
ax2.plot([1, 2, 3, 4, 5], [9, 4, 2, 4, 9])
ax3 = fig.add_subplot(2, 2, 4)
ax3.plot([1, 2, 3, 4, 5], [-7, -4, 2, -4, -7])
plt.suptitle("Subplots")
plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 5))
x_parabola = np.linspace(-5, 5, 100)
y_parabola = x_parabola ** 2
plt.plot(x_parabola, y_parabola)
plt.annotate('min', xy=(0, 0), xytext=(0, 10),
             arrowprops=dict(facecolor='green', shrink=0.05),
             ha='center', fontsize=12)
plt.title("3")
plt.show()



plt.figure(figsize=(8, 5))
data = np.random.randint(0, 11, size=(7, 7))
heatmap = plt.pcolormesh(data, cmap='viridis')
plt.colorbar(heatmap)
plt.title("Тепловая карта")
plt.show()


plt.figure(figsize=(8, 5))
x_sin = np.linspace(0, 5, 100)
y_sin = np.cos(x_sin * np.pi) 
plt.plot(x_sin, y_sin, color='red')
plt.fill_between(x_sin, y_sin, 0)
plt.title("fill_between")
plt.show()


plt.figure(figsize=(8, 5))
x = np.linspace(0, 5, 500)
y = np.cos(x * np.pi)
y[y < -0.5] = np.nan
plt.plot(x, y, color='tab:blue', linewidth=3)
plt.ylim(-1.0, 1.0)
plt.title("6")
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
x_step = [0, 1, 2, 3, 4, 5, 6]
y_step = [0, 1, 2, 3, 4, 5, 6]

axes[0].step(x_step, y_step, where='pre', color='green', marker='o')
axes[0].grid(True)
axes[0].set_title("where='pre'")

axes[1].step(x_step, y_step, where='mid', color='green', marker='o')
axes[1].grid(True)
axes[1].set_title("where='mid'")

axes[2].step(x_step, y_step, where='post', color='green', marker='o')
axes[2].grid(True)
axes[2].set_title("where='post'")

plt.suptitle("Ступенчатые графики")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
x_stack = np.arange(0, 11, 1)
y1_stack = np.array([0, 2, 3.5, 4.8, 5, 5, 4.8, 3.5, 2, 0, 0])
y2_stack = np.array([0, 4, 6,   8,   10, 9.5, 8,   6,   4, 0, 0])
y3_stack = np.array([0, 3, 5,   7,   9,  12,  14,  16,  18, 21, 20])
plt.stackplot(x_stack, y1_stack, y2_stack, y3_stack, labels=['y1', 'y2', 'y3'])
plt.legend(loc='upper left')
plt.title("Stackplot")
plt.show()

labels_pie = ['BMV', 'Toyota', 'Ford', 'Jaguar', 'AUDI']
sizes_pie = [35, 12, 18, 20, 15]
colors_pie = ['#2ca02c', '#ff7f0e', '#1f77b4', '#9467bd', '#d62728']
explode_pie = (0.1, 0, 0, 0, 0)
plt.figure(figsize=(6, 6))
plt.pie(sizes_pie, explode=explode_pie, labels=labels_pie, colors=colors_pie, startangle=140)
plt.title("Круговая диаграмма")
plt.show()


plt.figure(figsize=(6, 6))
plt.pie(sizes_pie, labels=labels_pie, colors=colors_pie, startangle=140)
centre_circle = plt.Circle((0,0), 0.50, fc='white')
fig_donut = plt.gcf()
fig_donut.gca().add_artist(centre_circle)
plt.title("Кольцевая диаграмма")
plt.show()



/////////////////////////////////////////////////////



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
iris = sns.load_dataset("iris")

print("PCA")
from sklearn.decomposition import PCA
iris_2_classes = iris[iris['species'] != 'virginica']
X_pca = iris_2_classes.iloc[:, 0:4].values
y_pca = iris_2_classes['species'].values
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_pca)
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[y_pca == 'setosa', 0], X_reduced[y_pca == 'setosa', 1], 
            color='red', label='setosa')
plt.scatter(X_reduced[y_pca == 'versicolor', 0], X_reduced[y_pca == 'versicolor', 1], 
            color='green', label='versicolor')
plt.title('Метод главных компонент (PCA) для 2 сортов')
plt.xlabel('Главная компонента 1')
plt.ylabel('Главная компонента 2')
plt.legend()
plt.grid(True)
plt.show()

print("\nRandomForestClassifier")
from sklearn.ensemble import RandomForestClassifier
X_rf = iris_2_classes.iloc[:, [0, 1]].values
y_rf = np.where(iris_2_classes['species'] == 'setosa', 1, 2)
forest = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
forest.fit(X_rf, y_rf)
x_min = X_rf[:, 0].min() - 0.5
x_max = X_rf[:, 0].max() + 0.5
y_min = X_rf[:, 1].min() - 0.5
y_max = X_rf[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = forest.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlGn)
plt.scatter(X_rf[y_rf == 1, 0], X_rf[y_rf == 1, 1], color='red', edgecolor='black', label='setosa')
plt.scatter(X_rf[y_rf == 2, 0], X_rf[y_rf == 2, 1], color='green', edgecolor='black', label='versicolor')
plt.title('Random Forest')
plt.xlabel('Длина чашелистика')
plt.ylabel('Ширина чашелистика')
plt.legend()
plt.show()

print("\nKMeans")
from sklearn.cluster import KMeans
X_kmeans = iris.iloc[:, [0, 1]].values
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_kmeans)
centers = kmeans.cluster_centers_
plt.figure(figsize=(8, 6))
plt.scatter(X_kmeans[:, 0], X_kmeans[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.7, label='Цветки')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Центры кластеров')
plt.title('Кластеризация методом K-средних')
plt.xlabel('Длина чашелистика')
plt.ylabel('Ширина чашелистика')
plt.legend()
plt.grid(True)
plt.show()

