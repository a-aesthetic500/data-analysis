import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

print(df.head())
print(df.describe())

df.hist(figsize=(12, 10), bins=30)
plt.suptitle("Histograms of Features")
plt.show()

for col in housing.feature_names:
    plt.figure(figsize=(5, 4))
    sns.scatterplot(x=df[col], y=df['MedHouseVal'])
    plt.title(f"{col} vs MedHouseVal")
    plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print(pd.DataFrame({'Feature': X_train.columns, 'Coefficient': model.coef_}))

vif = pd.DataFrame()
vif["feature"] = X.columns
vif["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X.shape[1])]
print(vif)

X_refined = df[['MedInc', 'HouseAge', 'AveRooms', 'Population', 'AveOccup', 'Latitude']]
y_log = np.log(df['MedHouseVal'])

X_train, X_test, y_train, y_test = train_test_split(X_refined, y_log, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
residuals = y_train - y_train_pred

sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution")
plt.show()

plt.scatter(y_train_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted log(MedHouseVal)")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.show()

print("Intercept:", model.intercept_)
print(pd.DataFrame({'Feature': X_refined.columns, 'Coefficient': model.coef_}))

vif = pd.DataFrame()
vif["feature"] = X_refined.columns
vif["VIF"] = [variance_inflation_factor(X_refined.values, i) for i in range(X_refined.shape[1])]
print("VIF\n", vif)

X_train, X_test, y_train, y_test = train_test_split(X_refined, y_log, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_log = model.predict(X_test)
print("Test R² (log target):", r2_score(y_test, y_pred_log))
print("Test RMSE (log target):", np.sqrt(mean_squared_error(y_test, y_pred_log)))

y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X_refined, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Test R² (original target):", r2_score(y_test, y_pred))
print("Test RMSE (original target):", np.sqrt(mean_squared_error(y_test, y_pred)))