import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv(r"sales.csv", low_memory=False)
#Veri seti hakkında bilgilerin alınması
"""print(df.shape)
print (df.info())
print(df.iloc[0])"""
#Gerekli olmayacak değişkenlerin silinmesi
df.drop("tax", axis=1, inplace=True)
df.drop("unit_price", axis=1, inplace=True)
df.drop("quantity", axis=1, inplace=True)
#print(df.info())
#Aykırı değerleri görme
sns.boxplot(x='reward_points', data=df)
plt.title('Distribution of Reward Points')
plt.show()
# Aykırı değerleri ortalama ile değiştirme
# IQR hesaplama
mean = df['reward_points'].mean()
Q1 = df['reward_points'].quantile(0.25)
Q3 = df['reward_points'].quantile(0.75)
IQR = Q3 - Q1

# Aykırı değerleri belirleme
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Aykırı değerleri ortalama ile değiştirme
df['reward_points'] = np.where((df['reward_points'] < lower_bound) | (df['reward_points'] > upper_bound), mean, df['reward_points'])

# Kategorik değişkenleri sayısal verilere dönüştürme
label_encoder = LabelEncoder()
categorical_columns = ['branch', 'city', 'customer_type', 'gender', 'product_name', 'product_category']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])
    """print(f"{col} sütunu:")
    print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
    print("\n")
print(df.info())"""
#print(df.iloc[0])
#Normalleştirme
scaler = MinMaxScaler()
df[['reward_points']] = scaler.fit_transform(df[['reward_points']])
#print(df.iloc[0])
# Veri görselleştirme

# Korelasyon hesabı
correlation_matrix = df.corr()
"""
# Korelasyon matrisini görselleştirme
plt.figure(figsize=(12, 10))
plt.xticks(rotation=45, ha='right')  # X eksenindeki etiketleri 45 derece döndür
plt.yticks(rotation=45)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')

plt.title('Veri Korelasyon Matrisi')
plt.show()
"""
# Dağılım grafikleri
"""
sns.scatterplot(x=df['reward_points'], y=df['gender'])
plt.title('reward_points vs gender')
plt.xlabel('reward_points')
plt.ylabel('gender')
plt.show()
"""

"""
sns.countplot(x='city', data=df)
plt.title('Sales Count by City')
plt.xlabel('Sales Count')
plt.ylabel('City')
plt.show()
"""
#REGRESYON ANALİZİ

X = df.drop("total_price", axis=1)
y = df["total_price"]

# Eğitim ve test verisine ayırma ( %80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Modelin başarısını değerlendirme
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R²: {r2}")"""

