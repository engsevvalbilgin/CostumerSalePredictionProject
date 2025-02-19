# Sales Data Analysis and Regression Model

This project demonstrates how to perform **data cleaning**, **preprocessing**, and **linear regression analysis** on a sales dataset. The goal is to predict the total price (`total_price`) of sales transactions based on various factors such as `reward_points`, `customer_type`, `product_name`, and others.

## Prerequisites

Make sure you have the following libraries installed:
- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn

You can install the required libraries using `pip`:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

## Steps Involved

### 1. **Loading the Dataset**
The dataset (`sales.csv`) is loaded into a pandas DataFrame. The dataset contains information about sales transactions, including variables like `reward_points`, `branch`, `city`, `product_name`, and `total_price`.

```python
df = pd.read_csv(r"sales.csv", low_memory=False)
```

### 2. **Data Preprocessing**
- Unnecessary columns (`tax`, `unit_price`, `quantity`) are dropped from the dataset.
  
```python
df.drop("tax", axis=1, inplace=True)
df.drop("unit_price", axis=1, inplace=True)
df.drop("quantity", axis=1, inplace=True)
```

### 3. **Outlier Detection and Handling**
A boxplot is used to visualize the distribution of `reward_points` and detect potential outliers. The Interquartile Range (IQR) is calculated, and any outliers in `reward_points` are replaced with the mean value of the column.

```python
sns.boxplot(x='reward_points', data=df)
plt.title('Distribution of Reward Points')
plt.show()

mean = df['reward_points'].mean()
Q1 = df['reward_points'].quantile(0.25)
Q3 = df['reward_points'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df['reward_points'] = np.where((df['reward_points'] < lower_bound) | (df['reward_points'] > upper_bound), mean, df['reward_points'])
```

### 4. **Categorical Data Encoding**
The categorical variables (`branch`, `city`, `customer_type`, `gender`, `product_name`, `product_category`) are converted to numerical values using **Label Encoding**.

```python
label_encoder = LabelEncoder()
categorical_columns = ['branch', 'city', 'customer_type', 'gender', 'product_name', 'product_category']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])
```

### 5. **Normalization**
The `reward_points` feature is normalized using **MinMaxScaler** to scale the values between 0 and 1.

```python
scaler = MinMaxScaler()
df[['reward_points']] = scaler.fit_transform(df[['reward_points']])
```

### 6. **Data Visualization**
Various visualizations are created, such as:
- **Correlation Heatmap**: Displays the correlation between features in the dataset.
- **Scatter Plots and Count Plots**: Visualize relationships between features like `reward_points` vs. `gender` and the distribution of sales by `city`.

```python
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
```

### 7. **Regression Analysis**
The **Linear Regression** model is used to predict `total_price`. The dataset is split into training (80%) and testing (20%) sets. The model is trained on the training data and then evaluated on the test data using performance metrics like **Mean Squared Error (MSE)** and **R²**.

```python
X = df.drop("total_price", axis=1)
y = df["total_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R²: {r2}")
```

### Sample Output

```
Mean Squared Error: 1234.56
R²: 0.89
```

## Conclusion

This project demonstrates the complete workflow of preparing a dataset for regression analysis, handling outliers, encoding categorical data, normalizing features, and training a linear regression model to predict a target variable. The model's performance is evaluated using common regression metrics, making it suitable for forecasting and further refinement.

This analysis can be extended to more complex datasets or to other predictive modeling tasks in the future.
