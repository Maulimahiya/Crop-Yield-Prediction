#Dataset load karna
import pandas as pd

file_path = (r"C:\Users\mauli\Downloads\Crop_Yield_Prediction\crop_yield.csv\crop_yield.csv")

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

#Dataset ko samajhna aur missing values check karna
print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

#Data cleaning
print("\nCleaning Data...")

df = df.drop_duplicates()

for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Rows After Cleaning:")
print(df.duplicated().sum())

print("\nCleaned Dataset Shape:")
print(df.shape)

#Crop data ke graphs banao
import matplotlib.pyplot as plt
import seaborn as sns

print("\nGenerating EDA Graphs...")

print(df.describe())

plt.figure(figsize=(8, 5))
sns.heatmap(df.select_dtypes(include="number").corr(),
            annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

for col in df.select_dtypes(include="number").columns[:3]:
    plt.figure(figsize=(7, 4))
    sns.histplot(df[col], kde=True)
    plt.title("Distribution of " + col)
    plt.tight_layout()
    plt.show()


#Dataset ke columns check karo
print("\nAll Dataset Columns:")
for col in df.columns:
    print(col)

print("\nData Types:")
print(df.dtypes)


#Features aur target select karo
from sklearn.model_selection import train_test_split

X = df.drop(columns=["Yield", "Production"])
y = df["Yield"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("Target column:", y.name)
print("Features prepared successfully!")

#Simple Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

X_simple = df[["Annual_Rainfall"]]
y_simple = df["Yield"]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=42
)

model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train_s)

y_pred_s = model_simple.predict(X_test_s)

print("\nSimple Linear Regression Results")
print("MAE:", mean_absolute_error(y_test_s, y_pred_s))
print("MSE:", mean_squared_error(y_test_s, y_pred_s))
print("RMSE:", np.sqrt(mean_squared_error(y_test_s, y_pred_s)))
print("R2 Score:", r2_score(y_test_s, y_pred_s))


#Multiple Linear Regression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X_multi = df.drop(columns=["Yield", "Production"])
y_multi = df["Yield"]

X_multi = pd.get_dummies(X_multi, drop_first=True)

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)

y_pred_m = model_multi.predict(X_test_m)

print("\nMultiple Linear Regression Results")
print("MAE:", mean_absolute_error(y_test_m, y_pred_m))
print("MSE:", mean_squared_error(y_test_m, y_pred_m))
print("RMSE:", np.sqrt(mean_squared_error(y_test_m, y_pred_m)))
print("R2 Score:", r2_score(y_test_m, y_pred_m))


#Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

X_poly = df[["Crop_Year", "Area", "Annual_Rainfall",
             "Fertilizer", "Pesticide"]]
y_poly = df["Yield"]

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_poly, y_poly, test_size=0.2, random_state=42
)

model_poly = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)

model_poly.fit(X_train_p, y_train_p)
y_pred_p = model_poly.predict(X_test_p)

print("\nPolynomial Regression Results")
print("MAE:", mean_absolute_error(y_test_p, y_pred_p))
print("MSE:", mean_squared_error(y_test_p, y_pred_p))
print("RMSE:", np.sqrt(mean_squared_error(y_test_p, y_pred_p)))
print("R2 Score:", r2_score(y_test_p, y_pred_p))



#K-Nearest Neighbors (KNN) Regressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X_knn = df.drop(columns=["Yield", "Production"])
X_knn = pd.get_dummies(X_knn, drop_first=True)
y_knn = df["Yield"]

X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(
    X_knn, y_knn, test_size=0.2, random_state=42
)

model_knn = make_pipeline(
    StandardScaler(),
    KNeighborsRegressor(n_neighbors=5)
)

model_knn.fit(X_train_k, y_train_k)
y_pred_k = model_knn.predict(X_test_k)

print("\nKNN Regression Results")
print("MAE:", mean_absolute_error(y_test_k, y_pred_k))
print("MSE:", mean_squared_error(y_test_k, y_pred_k))
print("RMSE:", np.sqrt(mean_squared_error(y_test_k, y_pred_k)))
print("R2 Score:", r2_score(y_test_k, y_pred_k))




#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

X_rf = df.drop(columns=["Yield", "Production"])
X_rf = pd.get_dummies(X_rf, drop_first=True)
y_rf = df["Yield"]

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_rf, y_rf, test_size=0.2, random_state=42
)

model_rf = RandomForestRegressor(
    n_estimators=100, random_state=42, n_jobs=-1
)

model_rf.fit(X_train_rf, y_train_rf)
y_pred_rf = model_rf.predict(X_test_rf)

print("\nRandom Forest Regression Results")
print("MAE:", mean_absolute_error(y_test_rf, y_pred_rf))
print("MSE:", mean_squared_error(y_test_rf, y_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test_rf, y_pred_rf)))
print("R2 Score:", r2_score(y_test_rf, y_pred_rf))



#Model Comparison
results = {
    "Algorithm": [
        "Simple Linear Regression",
        "Multiple Linear Regression",
        "Polynomial Regression",
        "KNN Regression",
        "Random Forest Regression"
    ],
    "MAE": [
        mean_absolute_error(y_test_s, y_pred_s),
        mean_absolute_error(y_test_m, y_pred_m),
        mean_absolute_error(y_test_p, y_pred_p),
        mean_absolute_error(y_test_k, y_pred_k),
        mean_absolute_error(y_test_rf, y_pred_rf)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test_s, y_pred_s)),
        np.sqrt(mean_squared_error(y_test_m, y_pred_m)),
        np.sqrt(mean_squared_error(y_test_p, y_pred_p)),
        np.sqrt(mean_squared_error(y_test_k, y_pred_k)),
        np.sqrt(mean_squared_error(y_test_rf, y_pred_rf))
    ],
    "R2 Score": [
        r2_score(y_test_s, y_pred_s),
        r2_score(y_test_m, y_pred_m),
        r2_score(y_test_p, y_pred_p),
        r2_score(y_test_k, y_pred_k),
        r2_score(y_test_rf, y_pred_rf)
    ]
}

comparison = pd.DataFrame(results)
print("\nMODEL COMPARISON")
print(comparison.round(4).to_string(index=False))

print("\nBest model based on R2 Score:")
print(comparison.loc[comparison["R2 Score"].idxmax(), "Algorithm"])



#Best model ka prediction graph
plt.figure(figsize=(8, 5))

plt.scatter(y_test_rf, y_pred_rf, alpha=0.5)
plt.xlabel("Actual Crop Yield")
plt.ylabel("Predicted Crop Yield")
plt.title("Random Forest: Actual vs Predicted Yield")

plt.tight_layout()
plt.show()




#Results ko CSV file mein save karo
comparison.to_csv("model_comparison.csv", index=False)

print("\nModel comparison saved successfully!")
print("File: model_comparison.csv")



