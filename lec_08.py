"""
Лекция 8: работа со строками и временными рядами в pandas

Содержание:
1. Строковые методы через .str (капитализация, длина, извлечение, замена, split)
2. Индикаторные переменные из строк с разделителями
3. Регулярные выражения для извлечения подстрок
4. Основы numpy.datetime64 и numpy.timedelta64
5. Pandas Timestamp, Timedelta, DatetimeIndex, PeriodIndex
6. Создание диапазонов дат и времени (date_range, period_range, timedelta_range)
7. Арифметика с датами и срезы по индексу
8. Рабочие примеры анализа данных рецептов (замена отсутствующего JSON)
9. Дополнительно: обработка пропусков, частоты, преобразование периодов
"""

import numpy as np
import pandas as pd

# =============================================================================
# 1. СТРОКОВЫЕ ОПЕРАЦИИ В PANDAS (Vectorized String Methods)
# =============================================================================

# Исходные данные: список строк
data = ["one one", "TWO TWO", "THREE THREE", "fouR fouR", np.nan]  # добавим пропуск
print("Исходные данные:", data)

# Создаём Series
names = pd.Series(data)
print("Объект Series:")
print(names)

# -----------------------------------------------------------------------------
# 1.1 Базовые методы .str
# -----------------------------------------------------------------------------

# .str.capitalize() - первая буква заглавная, остальные строчные
print("\n.str.capitalize():")
print(names.str.capitalize())   # NaN остаётся NaN

# .str.lower() / .str.upper()
print("\n.str.lower():")
print(names.str.lower())

print("\n.str.upper():")
print(names.str.upper())

# .str.len() - длина строки (для NaN возвращает NaN)
print("\n.str.len() - количество символов:")
print(names.str.len())

# .str.strip() - удаляет пробелы в начале и конце
print("\n.str.strip() (добавим пробелы вручную):")
names_with_spaces = pd.Series(["  text  ", "  another  "])
print(names_with_spaces.str.strip())

# .str.replace() - замена подстроки
print("\n.str.replace('TWO', '2'):")
print(names.str.replace("TWO", "2", regex=False))  # regex=False для точной замены

# .str.split() - разбивает строку по разделителю
print("\n.str.split() по пробелу (возвращает списки):")
print(names.str.split())

# Можно выбрать элемент списка через .str.get()
print("\nПервое слово после split:")
print(names.str.split().str.get(0))

# .str.cat() - объединение строк (требует осторожности с NaN)
print("\nОбъединение строк через запятую (игнорируем NaN):")
print(names.str.cat(sep=', ', na_rep='<EMPTY>'))

# .str.startswith() / .str.endswith()
print("\nНачинается с 'T':")
print(names.str.startswith('T', na=False))  # na=False считает NaN как False

# .str.isnumeric() - проверка, состоит ли строка только из цифр
print("\nПроверка на цифры:")
print(pd.Series(['123', '12a', '  1']).str.isnumeric())

# -----------------------------------------------------------------------------
# 1.2 Обработка пропущенных значений (NaN)
# -----------------------------------------------------------------------------
print("\nРабота с NaN в строковых методах:")
s_with_nan = pd.Series(['apple', np.nan, 'banana'])
print(s_with_nan)
print(".str.len() ->", s_with_nan.str.len())        # NaN остаётся NaN
print(".str.contains('a'):", s_with_nan.str.contains('a', na=False))  # na=False заменяет NaN на False

# -----------------------------------------------------------------------------
# 1.3 Регулярные выражения с .str.extract()
# -----------------------------------------------------------------------------

# Простое извлечение первой последовательности букв/цифр
regex = r"([a-z0-9]+)"   # одна группа
extracted = names.str.extract(regex, expand=False)   # expand=False -> Series
print("\n.extract(одна группа, expand=False):")
print(extracted)

# Извлечение двух групп (первые два слова)
regex_two = r"([a-z0-9]+)\s+([a-z0-9]+)"   # слово, пробелы, слово
extracted_df = names.str.extract(regex_two, expand=True)  # expand=True -> DataFrame
print("\n.extract(две группы, expand=True):")
print(extracted_df)

# .str.extractall() - если несколько совпадений в строке
text = pd.Series(['a1 b2 c3', 'd4 e5'])
print("\n.extractall() для поиска всех чисел:")
print(text.str.extractall(r'([a-z][0-9])'))

# -----------------------------------------------------------------------------
# 1.4 Индикаторные переменные из строк с разделителями
# -----------------------------------------------------------------------------

# Пример с колонкой 'info', содержащей категории, разделённые '|'
data_info = ['one1 one2 one3', 'two999 ddd ', 'three three', 'four four']
df_info = pd.DataFrame({
    'name': data_info,
    'info': ['A|B', 'B|C', 'A|B|C', 'D']
})
print("\nИсходный DataFrame:")
print(df_info)

# .str.get_dummies() создаёт one-hot encoding
dummies = df_info['info'].str.get_dummies('|')
print("\nРезультат .str.get_dummies('|'):")
print(dummies)

# Можно объединить с исходным DataFrame
df_info = pd.concat([df_info, dummies], axis=1)
print("\nОбъединённый DataFrame:")
print(df_info)

# =============================================================================
# 2. РАБОТА С ДАТАМИ И ВРЕМЕНЕМ
# =============================================================================
# -----------------------------------------------------------------------------
# 2.1 Основы numpy.datetime64 и numpy.timedelta64
# -----------------------------------------------------------------------------

# Создание даты с указанием точности (здесь наносекунды)
d_np = np.datetime64("2026-03-04", "ns")
print("numpy.datetime64 (ns):", d_np)
print("Прибавляем 1 (наносекунда):", d_np + 1)

# Разные единицы измерения
d_day = np.datetime64("2026-03-04", "D")
print("С точностью до дня:", d_day)
print("+1 день (автоматически преобразуется):", d_day + 1)

# Временные дельты
t_day = np.timedelta64(1, "D")
print("\ntimedelta64 1 день:", t_day)
print("+1 (увеличивает значение в тех же единицах):", t_day + 1)

t_month = np.timedelta64(1, "M")
print("timedelta64 1 месяц:", t_month)
print("+1:", t_month + 1)

# Сложение разных единиц может быть запрещено (зависит от контекста)
# Следующая строка вызовет TypeError в общем случае:
# print(t_day + t_month)   # TypeError: Cannot get common metadata for ...

# Но если сложить datetime64 с timedelta64, то numpy приведёт к общему знаменателю
print("datetime64 + timedelta64:", np.datetime64("2026-03-04") + np.timedelta64(10, 'D'))

# -----------------------------------------------------------------------------
# 2.2 Pandas: Timestamp, Timedelta, DatetimeIndex
# -----------------------------------------------------------------------------

# pd.to_datetime() преобразует множество форматов
ts = pd.to_datetime("4th of March, 2026")
print("\npandas Timestamp из текста:", ts)
print("Тип:", type(ts))
print("День недели (англ.):", ts.strftime("%A"))

# Можно передавать список строк -> DatetimeIndex
dt_idx = pd.to_datetime(["2026-03-04", "2026-03-05", "2026-03-06"])
print("\nDatetimeIndex из списка:")
print(dt_idx)
print("Тип:", type(dt_idx))

# pd.to_timedelta() создаёт Timedelta или TimedeltaIndex
td = pd.to_timedelta(1, unit='D')
print("\npandas Timedelta 1 день:", td)

# Арифметика с Timestamp и Timedelta
print("ts + td =", ts + td)

# Создание последовательности дат через сложение TimedeltaIndex
# pd.to_timedelta(np.arange(12), "D") создаёт массив длительностей
delta_array = pd.to_timedelta(np.arange(12), "D")
d2 = ts + delta_array
print("\nts + pd.to_timedelta(np.arange(12), 'D'):")
print(d2)
print("Тип результата:", type(d2))   # DatetimeIndex

# Используем этот индекс в Series
index = d2
data_series = pd.Series(np.arange(12), index=index)
print("\nРяд с индексом из дат:")
print(data_series)

# Срез по датам (частичное индексирование)
print("\nСрез '2026-03-06':'2026-03-10':")
print(data_series["2026-03-06":"2026-03-10"])

# Срез по месяцу (вернёт все даты апреля 2026, если они есть)
print("\nСрез '2026-04' (весь апрель):")
print(data_series["2026-04"])   # пусто, т.к. данных за апрель нет

# -----------------------------------------------------------------------------
# 2.3 Диапазоны дат и времени (date_range, period_range, timedelta_range)
# -----------------------------------------------------------------------------

# pd.date_range — равномерный временной ряд
dr = pd.date_range("2026-01-01", periods=10, freq="h")   # ежечасно
print("\npd.date_range(start='2026-01-01', periods=10, freq='h'):")
print(dr)
print("Первый элемент + 2 * freq:", dr[0] + 2 * dr.freq)

# Частоты: 'D' - день, 'B' - рабочий день, 'W' - неделя, 'M' - конец месяца, 'MS' - начало месяца
dr_business = pd.date_range("2026-01-01", periods=10, freq="B")
print("\nРабочие дни (freq='B'):")
print(dr_business)

dr_month_start = pd.date_range("2026-01-01", periods=5, freq="MS")
print("\nНачало месяцев (freq='MS'):")
print(dr_month_start)

# pd.period_range — периоды (интервалы)
pr = pd.period_range("2026-01-01", periods=10, freq="M")
print("\npd.period_range с частотой 'M' (месяцы):")
print(pr)

# Периоды поддерживают арифметику
print("pr[0] + 1 =", pr[0] + 1)
print("pr[0] - pr[1] =", pr[0] - pr[1])   # разность в единицах частоты

# pd.timedelta_range — диапазон длительностей
tdr = pd.timedelta_range(0, periods=10, freq="h")
print("\npd.timedelta_range(start=0, periods=10, freq='h'):")
print(tdr)

# Составные частоты, например, 2 часа 15 минут (поддерживается с pandas 1.3.0)
try:
    tdr2 = pd.timedelta_range(0, periods=10, freq='2h15min')
    print("\npd.timedelta_range с частотой '2h15min':")
    print(tdr2)
except ValueError as e:
    print("\nСоставная частота не поддерживается в этой версии pandas:", e)

# -----------------------------------------------------------------------------
# 2.4 Периоды (Period) и их атрибуты
# -----------------------------------------------------------------------------

p1 = pd.Period("2026Q1")   # первый квартал 2026
print("\nPeriod '2026Q1':", p1)
print("Месяц:", p1.month)          # для квартала месяц начала периода (1)
print("День:", p1.day)             # день начала периода (1)
print("Квартал:", p1.quarter)
print("Год:", p1.year)

p2 = pd.Period("2026-03-04", freq='D')
print("\nPeriod с дневной частотой:", p2)
print("День недели:", p2.dayofweek)   # 0 = понедельник, 6 = воскресенье

# -----------------------------------------------------------------------------
# 2.5 Преобразование между DatetimeIndex и PeriodIndex
# -----------------------------------------------------------------------------

# DatetimeIndex -> PeriodIndex
dt_idx = pd.to_datetime(["2026-03-04", "2026-03-05"])
p_idx = dt_idx.to_period("D")
print("\nDatetimeIndex -> PeriodIndex (freq='D'):")
print(p_idx)

# PeriodIndex -> DatetimeIndex (начало периода)
dt_from_p = p_idx.to_timestamp(how='start')
print("\nPeriodIndex -> DatetimeIndex (начало периода):")
print(dt_from_p)

# -----------------------------------------------------------------------------
# 2.6 Дополнительно: обработка ошибок при парсинге дат
# -----------------------------------------------------------------------------

# Параметр errors='coerce' превращает непонятные строки в NaT
dates_mixed = pd.to_datetime(['2026-03-04', 'not a date', '2026-03-06'], errors='coerce')
print("\nПарсинг с errors='coerce':")
print(dates_mixed)

# -----------------------------------------------------------------------------
# 2.7 Работа с часовыми поясами (tz)
# -----------------------------------------------------------------------------

# Локализация временной метки
ts_utc = pd.Timestamp("2026-03-04 12:00").tz_localize('UTC')
print("\nTimestamp с часовым поясом UTC:", ts_utc)

# Конвертация в другой пояс
ts_moscow = ts_utc.tz_convert('Europe/Moscow')
print("То же время в Москве:", ts_moscow)

# =============================================================================
# 3. ПРИМЕР С РЕЦЕПТАМИ (вместо загрузки из JSON используем свой датасет)
# =============================================================================

# Создаём небольшой DataFrame, имитирующий загруженные данные
recipes = pd.DataFrame({
    'name': [
        'Pancakes',
        'Omelette',
        'Salad cream',
        'Chicken Tikka Masala',
        'Beef Nachos with Chili',
        'Avocado Toast'
    ],
    'description': [
        'Breakfast dish with syrup',
        'Quick breakfast with eggs',
        'Lunch with cream and salt',
        'Indian dinner, spicy',
        'Snack with pepper and cream',
        'Healthy breakfast with avocado'
    ],
    'ingredients': [
        'flour, milk, eggs, salt',
        'eggs, salt, pepper, cream',
        'lettuce, tomatoes, cream, salt',
        'chicken, yogurt, spices, salt, pepper',
        'beef, tortilla, cheese, cream, salt, pepper',
        'bread, avocado, salt, pepper'
    ]
})

print("Исходный датасет recipes:")
print(recipes)

# -----------------------------------------------------------------------------
# 3.1 Поиск подстрок в тексте (.str.contains)
# -----------------------------------------------------------------------------

# Подсчёт рецептов со словом 'Breakfast' в описании (учёт регистра)
breakfast_count = recipes.description.str.contains("Breakfast").sum()
print(f"\nРецептов со словом 'Breakfast': {breakfast_count}")

# Без учёта регистра с помощью flags=re.IGNORECASE или предварительного lower()
breakfast_lower_count = recipes.description.str.lower().str.contains("breakfast").sum()
print(f"Рецептов со словом 'breakfast' (без учёта регистра): {breakfast_lower_count}")

# -----------------------------------------------------------------------------
# 3.2 Индикаторные переменные для специй
# -----------------------------------------------------------------------------

spices = ["salt", "pepper", "cream"]

# Создаём булевы колонки для каждой специи
for spice in spices:
    recipes[spice] = recipes.ingredients.str.contains(spice)

print("\nDataFrame с колонками специй:")
print(recipes[['name'] + spices])

# Выбираем рецепты, где есть ВСЕ три специи
all_three = recipes[spices].all(axis=1)
print("\nРецепты, содержащие salt, pepper и cream:")
print(recipes.loc[all_three, 'name'])

# Альтернатива через query
indicator_df = recipes[spices]   # берём только булевы колонки
selected = indicator_df.query("salt & pepper & cream")
print("\nТе же рецепты через query:")
print(recipes.loc[selected.index, 'name'])

# -----------------------------------------------------------------------------
# 3.3 Статистика по длине списка ингредиентов
# -----------------------------------------------------------------------------

# Длина строки ingredients (количество символов)
recipes['ingredients_len'] = recipes.ingredients.str.len()
print("\nДобавлена колонка 'ingredients_len':")
print(recipes[['name', 'ingredients_len']])

# Описательная статистика
print("\nСтатистика по длине ингредиентов:")
print(recipes['ingredients_len'].describe())

# Рецепт с самой длинной строкой ингредиентов
max_len_idx = recipes['ingredients_len'].idxmax()
print(f"\nРецепт с самой длинной строкой ингредиентов: {recipes.loc[max_len_idx, 'name']}")
print(f"Длина строки: {recipes.loc[max_len_idx, 'ingredients_len']}")

# -----------------------------------------------------------------------------
# 3.4 Доступ к отдельной записи
# -----------------------------------------------------------------------------

print("\nПервая запись (Series):")
print(recipes.iloc[0])

# =============================================================================
# 4. ДОПОЛНИТЕЛЬНЫЕ ПРИМЕРЫ С ВРЕМЕННЫМИ РЯДАМИ
# =============================================================================
# -----------------------------------------------------------------------------
# 4.1 Арифметика с индексами и смещениями
# -----------------------------------------------------------------------------

idx = pd.date_range("2026-03-01", periods=5, freq="D")
print("Индекс:", idx)
print("Сдвиг на 2 дня:", idx + pd.Timedelta(days=2))

# Разность двух индексов даёт TimedeltaIndex
diff = idx[1:] - idx[:-1]
print("\nРазность соседних дат:", diff)

# -----------------------------------------------------------------------------
# 4.2 Преобразование частоты (resample) для временных рядов
# -----------------------------------------------------------------------------

# Создадим почасовые данные за несколько дней
rng = pd.date_range("2026-03-01", periods=72, freq="h")
ts_data = pd.Series(np.random.randn(len(rng)), index=rng)
print("\nПочасовые данные (первые 5):")
print(ts_data.head())

# Пересэмплируем до среднего за день
daily_mean = ts_data.resample('D').mean()
print("\nСреднее за день (resample('D').mean()):")
print(daily_mean)

# -----------------------------------------------------------------------------
# 4.3 Работа с пропусками в датах
# -----------------------------------------------------------------------------

# Создадим даты с пропуском
dates_with_gap = pd.date_range("2026-03-01", periods=5, freq="D").delete(2)  # удалили 3-ю дату
print("\nДаты с пропуском (индекс):", dates_with_gap)
values = [10, 20, 30, 40]
series_gap = pd.Series(values, index=dates_with_gap)
print("Ряд с пропущенной датой:")
print(series_gap)

# Восстановление непрерывного индекса с интерполяцией
series_filled = series_gap.resample('D').asfreq().interpolate()
print("\nПосле resample и интерполяции:")
print(series_filled)
