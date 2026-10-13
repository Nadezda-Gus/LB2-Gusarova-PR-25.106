import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Загружаем датасет из библиотеки seaborn
df = sns.load_dataset('titanic')
# Выводим общую информацию о датасете
print("Информация о датасете:")
df.info()
# Выводим первые 5 строк
print("\nПервые 5 строк датасета:")
print(df.head())
# Статистика по числовым признакам
print("\nСтатистика по числовым признакам:")
print(df.describe())


print("\nКоличество пропусков в каждом столбце:")
print(df.isnull().sum())
# Заполнение 'age' медианой
median_age = df['age'].median()
df['age'].fillna(median_age, inplace=True)
# Заполнение 'embarked' модой
mode_embarked = df['embarked'].mode()[0]
df['embarked'].fillna(mode_embarked, inplace=True)
# Удаление столбцов
df.drop(['deck', 'embark_town'], axis=1, inplace=True)
# Проверяем, что пропусков не осталось (кроме 'age' и 'embarked')
print("\nКоличество пропусков после обработки:")
print(df.isnull().sum())

# Применение One-Hot Encoding к 'sex' и 'embarked'
df_processed = pd.get_dummies(df, columns=['sex', 'embarked'],
drop_first=True)
# Удалим и другие некатегориальные, но не числовые столбцы для упрощения
# Или те, что не несут пользы для гипотетической модели
df_processed.drop(['pclass', 'who', 'adult_male', 'class', 'alive',
'alone'], axis=1, inplace=True)
print("\nДатасет после One-Hot Encoding:")
print(df_processed.head())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Выбираем столбцы для масштабирования
cols_to_scale = ['age', 'fare']
df_processed[cols_to_scale] = scaler.fit_transform(df_processed[cols_to_scale])
print("\nДатасет после масштабирования 'age' и 'fare':")
print(df_processed.head())

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.boxplot(y=df_processed['fare'])
plt.title('Boxplot для Fare (до обработки выбросов)')
# Расчет IQR для 'fare'
Q1 = df_processed['fare'].quantile(0.25)
Q3 = df_processed['fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Ограничение выбросов (Capping)
df_processed['fare'] = np.where(df_processed['fare'] > upper_bound,
                                upper_bound, df_processed['fare'])
df_processed['fare'] = np.where(df_processed['fare'] < lower_bound,
lower_bound, df_processed['fare'])
plt.subplot(1, 2, 2)
sns.boxplot(y=df_processed['fare'])
plt.title('Boxplot для Fare (после обработки выбросов)')
plt.tight_layout()
plt.show()
print("\nФинальный вид обработанного датасета:")
print(df_processed.head())

print("\n=== ЗАДАНИЕ 2 ===")
# 1. Загрузка датасета
df2 = sns.load_dataset('titanic')
# 2. Заполнение age средним значением
mean_age = df2['age'].mean()
df2['age'].fillna(mean_age, inplace=True)
# Заполнение embarked модой
mode_embarked = df2['embarked'].mode()[0]
df2['embarked'].fillna(mode_embarked, inplace=True)
# Удаление столбцов
df2.drop(['deck', 'embark_town'], axis=1, inplace=True)
print("Статистика до масштабирования:")
print(df2[['age', 'fare']].describe())
# 3. Label Encoding для sex
le = LabelEncoder()
df2['sex_encoded'] = le.fit_transform(df2['sex'])
# One-Hot Encoding для embarked
df2 = pd.get_dummies(df2, columns=['embarked'], drop_first=True)
# Удаление лишних столбцов
df2.drop(['pclass', 'who', 'adult_male', 'class', 'alive', 'alone', 'sex'],
axis=1, inplace=True)
# 4. Масштабирование с помощью нормализации
scaler = MinMaxScaler()
cols_to_scale = ['age', 'fare']
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])
print("\nСтатистика после масштабирования:")
print(df2[['age', 'fare']].describe())
print("\nДатасет после обработки:")
print(df2.head())
print("\nАнализ применения Label Encoding:")
print("Label Encoding для признака sex может быть некорректным, потомучто:")
print("1. Модель может ошибочно интерпретировать коды как порядковые (0 <1)")
print("2. Это может привести к неверным весовым коэффициентам в модели")
print("3. Для бинарных признаков без естественного порядка лучшеиспользовать One-Hot Encoding")

print("\n=== ЗАДАНИЕ 3 ===")
# 1. Загрузка датасета
df3 = sns.load_dataset('titanic')
print(f"Размер датасета до удаления пропусков в age: {df3.shape}")
# 2. Удаление строк с пропусками в age
initial_size = df3.shape[0]
df3 = df3.dropna(subset=['age'])
final_size = df3.shape[0]
print(f"Размер датасета после удаления пропусков в age: {df3.shape}")
print(f"Удалено строк: {initial_size - final_size}")
# Заполнение embarked модой
mode_embarked = df3['embarked'].mode()[0]
df3['embarked'].fillna(mode_embarked, inplace=True)
# Удаление столбцов
df3.drop(['deck', 'embark_town'], axis=1, inplace=True)
# 3. One-Hot Encoding для sex и embarked
df3 = pd.get_dummies(df3, columns=['sex', 'embarked'], drop_first=True)
df3.drop(['pclass', 'who', 'adult_male', 'class', 'alive', 'alone'], axis=1,
inplace=True)
# 4. Стандартизация для fare
scaler = StandardScaler()
df3['fare'] = scaler.fit_transform(df3[['fare']])
print("\nДатасет после кодирования и стандартизации:")
print(df3.head())
# 5. Удаление выбросов в fare по методу IQR
Q1 = df3['fare'].quantile(0.25)
Q3 = df3['fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(f"\nГраницы выбросов: нижняя = {lower_bound:.2f}, верхняя = {upper_bound:.2f}")
before_outliers = df3.shape[0]
df3 = df3[(df3['fare'] >= lower_bound) & (df3['fare'] <= upper_bound)]
after_outliers = df3.shape[0]
print(f"Удалено выбросов: {before_outliers - after_outliers}")
print("\nСравнение подходов к обработке пропусков:")
print(f"При заполнении медианой (Задание 1): сохраняется {df1.shape[0]}строк")
print(f"При удалении строк (Задание 3): сохраняется {df3.shape[0]} строк")
print(f"Потеря данных при удалении: {((initial_size - final_size) /initial_size * 100):.1f}%")
print("\nВывод: Удаление строк с пропусками привело к большей потереданных,")
print("поскольку мы теряем всю информацию из этих строк. Заполнениемедианой")
print("сохраняет все наблюдения, но может внести некоторую погрешность.")