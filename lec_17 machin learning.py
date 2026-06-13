
"""
================================================================================================
МАШИННОЕ ОБУЧЕНИЕ И НЕЙРОННЫЕ СЕТИ 
------------------------------------------------------------------------------------------------
РАЗДЕЛ 1: ТИПЫ НЕЙРОННЫХ СЕТЕЙ И ИХ АРХИТЕКТУРА
------------------------------------------------------------------------------------------------
  основные виды нейронных сетей:
1. Сверточные нейронные сети (CNN - Convolutional Neural Networks): 
   Используются для задач компьютерного зрения (классификация изображений, поиск объектов).
2. Рекуррентные нейронные сети (RNN): 
   Применяются для обработки последовательностей: распознавание рукописного текста, 
   обработка естественного языка (NLP), анализ временных рядов.
3. Генеративно-состязательные сети (GAN): 
   Используются для создания нового контента (генерация фотореалистичных лиц, музыки).
4. Многослойный перцептрон (MLP): 
   Базовая полносвязная архитектура. Подходит для классических табличных данных.

Архитектура MLP состоит из Узлов (Нейронов), объединенных в Слои:
* Входной слой (Input layer).
* Скрытые слои (Hidden layers) - их количество определяет "Глубину" (Depth) сети. 
  Количество нейронов в слое называется "Шириной" (Width).
* Выходной слой (Output layer).
В полносвязных сетях каждый нейрон одного слоя соединен со всеми нейронами следующего.

------------------------------------------------------------------------------------------------
РАЗДЕЛ 2: ПРЕДСТАВЛЕНИЕ ДАННЫХ И ТЕНЗОРЫ
------------------------------------------------------------------------------------------------
Для компьютера любые данные — будь то текст, картинки или звук — представляют собой 
многомерные массивы чисел (Тензоры). 
* Скаляр (0D): Одиночное число.
* Вектор (1D): Одномерный массив (например, аудиосигнал или вектор признаков).
* Матрица (2D): Двумерная таблица (черно-белое изображение).
* 3D Тензор: Цветное фото (Высота × Ширина × 3 цветовых канала: Красный, Зеленый, Синий).
* 4D Тензор: Пакет (Batch) изображений. Нейросети обрабатывают данные пачками для скорости.

------------------------------------------------------------------------------------------------
РАЗДЕЛ 3: АНАТОМИЯ НЕЙРОНА И ФУНКЦИИ АКТИВАЦИИ
------------------------------------------------------------------------------------------------
Каждый искусственный нейрон (перцептрон) вычисляет взвешенную сумму входящих сигналов:
Y = Функция_Активации( Σ (Вход * Вес) + Смещение )

* Вес (Weight, w): Сила связи. В процессе обучения сеть автоматически подбирает эти параметры.
* Смещение (Bias, b): Гибкость, порог активации.
* Функция активации (Activation): Нелинейный фильтр, позволяющий сети решать сложные задачи.
  - ReLU (max(0, x)): Золотой стандарт, но имеет проблему "Мертвых нейронов" при минусовых входах.
  - Leaky ReLU: Спасает отрицательные сигналы.
  - Softmax: Используется на выходе для превращения чисел в вероятности (сумма = 1.0).

------------------------------------------------------------------------------------------------
РАЗДЕЛ 4: ОБУЧЕНИЕ СЕТИ И ФРЕЙМВОРКИ
------------------------------------------------------------------------------------------------
Как сеть "умнеет"?
1. Прямой проход (Forward Pass): Сеть делает предсказание.
2. Функция потерь (Loss Function): Оценивает масштаб ошибки.
3. Обратное распространение (Backpropagation): Вычисляются градиенты (производные).
4. Оптимизатор (Optimizer): Метод (например, Adam), обновляющий веса на основе градиентов 
   для минимизации ошибки.

Для разработки в индустрии используются фреймворки: TensorFlow / Keras (Google) и PyTorch (Meta). 
Для обмена моделями между фреймворками создан стандарт ONNX.

------------------------------------------------------------------------------------------------
РАЗДЕЛ 5: ТРАНСФЕРНОЕ ОБУЧЕНИЕ И АУГМЕНТАЦИЯ
------------------------------------------------------------------------------------------------
Обучать сеть с нуля сложно. В индустрии берут готовую сеть (например, MobileNet), 
замораживают ее "глаза" (сверточные слои), отрезают старый выходной слой и пришивают новый. 
Обучается только эта малая часть под новую задачу, что занимает минуты вместо дней.

Аугментация (Data Augmentation):
Программное искажение фотографий (повороты, масштабирование). Искусственно увеличивает 
базу данных, защищая модель от переобучения (зубрежки).

================================================================================================
ПРАКТИЧЕСКАЯ ЧАСТЬ
================================================================================================
"""

import os
import math
import warnings
import numpy as np
import urllib.request
import ssl
from io import BytesIO

# Настройки для чистой консоли и предотвращения ошибок сертификатов при скачивании
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
warnings.filterwarnings('ignore')
ssl._create_default_https_context = ssl._create_unverified_context

def print_section(title: str):
    print(f"\n{'='*95}\n 🔹 {title.upper()}\n{'='*95}")

# Вспомогательная функция для загрузки изображений по URL (заменяет локальные файлы)
def load_img_from_url(url, target_size=(224, 224)):
    from PIL import Image
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    img = Image.open(BytesIO(urllib.request.urlopen(req).read()))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img.resize(target_size)

# ========================================================================================
# 1: БАЗОВЫЕ ОПЕРАЦИИ
# ========================================================================================
def module_1_numpy_basics():
    print_section("1: РАБОТА С ТЕНЗОРАМИ В NUMPY")
    print("Рассмотрим базовый инструмент подготовки данных — библиотеку NumPy.\n")
    
    # Вектор (1D Тензор)
    a = np.array([1, 2, 3, 4, 5])
    print(f"Вектор 'a': {a}")
    print(f"Срез a[1:3] (со второго элемента по третий): {a[1:3]}")
    print(f"Срез a[:2] (от начала до второго элемента): {a[:2]}")
    print(f"Срез a[2:] (со второго элемента до конца): {a[2:]}")
    print(f"Индекс a[-1] (последний элемент): {a[-1]}")
    
    # Матрица (2D Тензор)
    array2 = np.array([
        [1, 2, 3, 4], 
        [5, 6, 7, 8], 
        [9, 10, 11, 12]
    ])
    print("\nМатрица 'array2':\n", array2)
    print("\nСрез array2[:, :2] (Все строки, но только первые два столбца):\n", array2[:, :2])
    print("\nСрез array2[1:3, :2] (Строки с 1 по 2, первые два столбца):\n", array2[1:3, :2])
    print(f"Индекс array2[-1, -1] (Нижний правый угол): {array2[-1, -1]}")
    
    # Трехмерный массив (3D Тензор) - формат цветных изображений
    array3 = np.array([ [[1,2,3], [4,5,6]], [[1,2,3], [4,5,6]] ])
    print(f"\nРазмерность 3D тензора (Глубина, Высота, Ширина): {array3.shape}")

# ========================================================================================
# 2: СИМУЛЯЦИЯ MLP И УЯЗВИМОСТЬ ФУНКЦИИ RELU
# ========================================================================================
def module_2_mlp_simulation():
    print_section("2: СИМУЛЯЦИЯ ВНУТРЕННОСТЕЙ СЕТИ И 'МЕРТВЫЙ' RELU")
    print("Имитация уже обученной сети для сложения чисел (2 входа -> 3 скрытых узла -> Выход).")
    
    # Зафиксированные веса и смещения
    w1, w2, w3 = 1.0264927, 0.9856401, -0.04909311
    w4, w5, w6 = 0.88846944, 0.05428, 0.794296
    w7, w8 = 1.1687347, 0.80088929
    
    b0, b1, b2, b3 = -0.00078612, -0.06866002, -0.00055442, -0.00008929

    def relu(x): return max(0, x)
    def leaky_relu(x, alpha=0.1): return x if x > 0 else x * alpha

    def predict_sim(x1, x2, act_fn="relu"):
        # Суммы на скрытом слое
        h1 = (x1 * w1) + (x2 * w2) + b0
        h2 = (x1 * w3) + (x2 * w4) + b1
        h3 = (x1 * w5) + (x2 * w6) + b2
        
        # Активация и сборка
        if act_fn == "relu":
            return (relu(h1)*w7) + (relu(h2)*w8) + (relu(h3)*w6) + b3
        return (leaky_relu(h1)*w7) + (leaky_relu(h2)*w8) + (leaky_relu(h3)*w6) + b3

    print("\n Тест 1 (Обычные данные): predict(2, 2) =", predict_sim(2, 2, "relu"))
    print(" Тест 2 (Дробные): predict(1.5, 1.5) =", predict_sim(1.5, 1.5, "relu"))
    print("\n Тест 3 (Аномальные данные): predict(-10, -15) =", predict_sim(-10, -15, "relu"))
    print(">> КРИТИЧЕСКАЯ ОШИБКА: Отрицательные суммы обнулились функцией ReLU. Нейроны 'умерли'.")
    
    print("\n Тест 4 (Авторский фикс): Используем Leaky ReLU (-10, -15)")
    print("Ответ с Leaky ReLU:", predict_sim(-10, -15, "leaky_relu"))

# ========================================================================================
# 3: ОБУЧЕНИЕ НЕЙРОНА "С НУЛЯ" (ЛИНЕЙНАЯ РЕГРЕССИЯ)
# ========================================================================================
def module_3_linear_regression():
    print_section("3: КАК СЕТЬ 'УМНЕЕТ' (ДОПОЛНЕНИЕ)")
    print("Дополнительный пример: Нейросеть из 1 узла выводит формулу Цельсий -> Фаренгейт.")
    try:
        import tensorflow as tf
    except ImportError: return

    # Данные для обучения (Вход X и Истина Y)
    celsius = np.array([-40, -10,  0,  8, 15, 22,  38], dtype=float)
    fahrenheit = np.array([-40,  14, 32, 46, 59, 72, 100], dtype=float)

    # 1 слой с 1 нейроном
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

    print("\nСеть учится, подбирая Вес (W) и Смещение (b)...")
    model.fit(celsius, fahrenheit, epochs=500, verbose=0)

    # Боевое предсказание
    pred = model.predict(np.array([100.0]), verbose=0)[0][0]
    print(f"Ответ сети на вопрос 'Сколько Фаренгейтов в 100°C?': {pred:.2f}°F (Идеал: 212.00)")
    
    weights = model.layers[0].get_weights()
    print(f"Подобранный Вес (W): {weights[0][0][0]:.2f} (Формула: 1.8)")
    print(f"Подобранное Смещение (b): {weights[1][0]:.2f} (Формула: 32.0)")

# ========================================================================================
# 4: АНАЛИЗ ТЕНЗОРОВ ИЗОБРАЖЕНИЙ И RESNET-50
# ========================================================================================
def module_4_tensor_analysis_resnet():
    print_section("4: АНАЛИЗ ПИКСЕЛЕЙ И ИНФЕРЕНС ЧЕРЕЗ RESNET-50")
    print("Детальный разбор процессов нормализации.\n")

    try:
        from tensorflow.keras.preprocessing import image as keras_image
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
    except ImportError: return

    cat_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"
    img_pil = load_img_from_url(cat_url, target_size=(224, 224))
    
    img_array = keras_image.img_to_array(img_pil)
    
    print("--- СЫРОЙ МАССИВ ДАННЫХ ИЗОБРАЖЕНИЯ ---")
    print(f"print(img_array.shape): {img_array.shape}")
    print(f"print(img_array[100, 100]) [RGB центрального пикселя]: {img_array[100, 100]}")
    print(f"print(np.min(img_array)): {np.min(img_array)}")
    print(f"print(np.max(img_array)): {np.max(img_array)}")

    # Формирование пакета
    img_batch = np.expand_dims(img_array, axis=0) 
    # Нормализация
    img_preprocessed = preprocess_input(img_batch)
    
    print("\n--- ДАННЫЕ ПОСЛЕ НОРМАЛИЗАЦИИ (preprocess_input) ---")
    print("Предобученные сети требуют математического центрирования пикселей")
    print(f"print(img_preprocessed.shape): {img_preprocessed.shape}")
    print(f"print(img_preprocessed[0, 100, 100]): {img_preprocessed[0, 100, 100]}")
    print(f"print(np.min(img_preprocessed)): {np.min(img_preprocessed)}")
    print(f"print(np.max(img_preprocessed)): {np.max(img_preprocessed)}")
    
    print("\n--- ИНФЕРЕНС (Распознавание сетью ResNet50) ---")
    model = ResNet50(weights='imagenet')
    preds = model.predict(img_preprocessed, verbose=0)
    
    print("Результаты функции decode_predictions:")
    for i, (_, label, prob) in enumerate(decode_predictions(preds, top=3)[0]):
        print(f"  {label.upper()} : {prob * 100:.2f}%")

# ========================================================================================
# 5: МЕХАНИКА СВЕРТКИ В КОМПЬЮТЕРНОМ ЗРЕНИИ
# ========================================================================================
def module_5_manual_convolution():
    print_section("5: КАК 'ВИДЯТ' СВЕРТОЧНЫЕ СЕТИ (ДОПОЛНЕНИЕ)")
    print("Дополнительный пример: Ручное применение фильтра Собеля к матрице изображения.\n")
    
    # 5x5 картинка. Вертикальная линия (10) на черном фоне (0)
    image = np.array([
        [0, 0, 10, 0, 0],
        [0, 0, 10, 0, 0],
        [0, 0, 10, 0, 0],
        [0, 0, 10, 0, 0],
        [0, 0, 10, 0, 0]
    ])
    
    # Фильтр Собеля (3x3) для поиска вертикальных границ
    sobel_filter = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    feature_map = np.zeros((3, 3))
    # Матричное скольжение (Convolution)
    for i in range(3):
        for j in range(3):
            patch = image[i:i+3, j:j+3]
            feature_map[i, j] = np.sum(patch * sobel_filter)

    print("Оригинальное изображение (5x5):\n", image)
    print("\nКарта признаков (Feature Map) после прохода фильтра:\n", feature_map)
    print(">> внимание: Ядро выдало сильный сигнал строго в местах перепада яркости!")

# ========================================================================================
# 6: ПОЛНЫЙ ПАЙПЛАЙН ТРАНСФЕРНОГО ОБУЧЕНИЯ 
# ========================================================================================
def module_6_full_transfer_pipeline():
    print_section("6: АУГМЕНТАЦИЯ И ТРАНСФЕРНОЕ ОБУЧЕНИЕ")
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
        from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
        from tensorflow.keras.models import Model, load_model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.preprocessing import image as keras_image
    except ImportError: return

    # Загрузка мини-датасета
    print("[1] Загрузка датасета 'Dogs vs Cats'...")
    url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    zip_p = tf.keras.utils.get_file('cats_and_dogs.zip', origin=url, extract=True)
    base_dir = os.path.join(os.path.dirname(zip_p), 'cats_and_dogs_filtered')
    
    TRAIN_DIR, VAL_DIR = os.path.join(base_dir, 'train'), os.path.join(base_dir, 'validation')
    IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, NUM_CLASSES = 224, 224, 64, 2
    TRAIN_SAMPLES = 500       # выборка по 500 шт
    VALIDATION_SAMPLES = 500

    print("[2] Инициализация ImageDataGenerator (Аугментация)...")
    # Точные параметры (повороты на 20 градусов, зум, отзеркаливание)
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20, 
        width_shift_range=0.2, 
        height_shift_range=0.2,
        zoom_range=0.2, 
    )
    # Валидация не искажается
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, class_mode='categorical'
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, class_mode='categorical'
    )

    print("\n[3] Сборка Гибридной Модели (MobileNet + Custom Head)...")
    # include_top=False — важнейший параметр, отрезающий старую классификацию
    model = MobileNet(include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    
    # Заморозка весов
    for layer in model.layers:
        layer.trainable = False 

    # Точный код сборки кастомной модели
    input_tensor = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    custom_model = model(input_tensor)
    custom_model = GlobalAveragePooling2D()(custom_model)
    custom_model = Dropout(0.5)(custom_model)
    prediction = Dense(NUM_CLASSES, activation='softmax')(custom_model)
    
    target_model = Model(inputs=input_tensor, outputs=prediction)
    target_model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])

    print("\n[4] Обучение (model.fit)...")
    # math.ceil округляет шаги вверх, чтобы охватить все данные батча
    num_steps = math.ceil(float(TRAIN_SAMPLES) / BATCH_SIZE)
    val_steps = math.ceil(float(VALIDATION_SAMPLES) / BATCH_SIZE)

    target_model.fit(
        train_gen, steps_per_epoch=num_steps, epochs=1, # 1 эпоха для скорости скрипта
        validation_data=val_gen, validation_steps=val_steps, verbose=1
    )

    print("\n[5] Сохранение модели на диск...")
    model_path = "our_model.h5"
    target_model.save(model_path)

    print("\n[6] Загрузка модели и Инференс (Сырой выход Softmax)...")
    loaded_model = load_model(model_path)

    # Фото luna.jpg (собака)
    luna_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c0/Golden_Retriever_with_tennis_ball.jpg/320px-Golden_Retriever_with_tennis_ball.jpg"
    img_luna = load_img_from_url(luna_url)
    
    img_array = keras_image.img_to_array(img_luna)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    
    # Сырой предикт - вывод
    raw_pred = loaded_model.predict(img_preprocessed, verbose=0)
    print("print(prediction) ->", raw_pred)

    if os.path.exists(model_path):
        os.remove(model_path)

# ========================================================================================
if __name__ == "__main__":
    print("\n" + "*"*95)
    print(" ЗАПУСК: РАБОТА")
    print("*"*95)
    
    module_1_numpy_basics()
    module_2_mlp_simulation()
    module_3_linear_regression()
    module_4_tensor_analysis_resnet()
    module_5_manual_convolution()
    module_6_full_transfer_pipeline()
    
    print_section("конец")
