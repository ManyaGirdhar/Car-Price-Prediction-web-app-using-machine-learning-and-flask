import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('automobile.csv')

df.head(4)

df

df.columns

df.dtypes

df.isna().any()
df = df.dropna(axis=1)

df.size

df.isna().any()

df.dtypes

df['make'].fillna('', inplace=True)  # Replace NaN with an empty string
df['make'] = df['make'].astype(str)
print(df['make'].unique())

df['fuel-type'].fillna('', inplace=True)
df['fuel-type'] = df['fuel-type'].astype(str)
df['fuel-type'].unique()

make_encoder = LabelEncoder()
fuel_encoder = LabelEncoder()

df['make'] = make_encoder.fit_transform(df['make'])
df['fuel-type'] = fuel_encoder.fit_transform(df['fuel-type'])

df['make']

print(make_encoder.classes_)
print(fuel_encoder.classes_)

print("Make Encoder : ")
points = [i for i in range(len(make_encoder.classes_))]
for i in zip(points ,make_encoder.inverse_transform(points)):
    print(i)

print("Fuel Encoder : ")
points = [i for i in range(len(fuel_encoder.classes_))]
for i in zip(points ,fuel_encoder.inverse_transform(points)):
    print(i)

df.isna().any()

df.size

df.dtypes

df = df.drop(df.select_dtypes(['object']), axis=1)

df.columns

from sklearn.model_selection import train_test_split

X = df.drop('price', axis=1)
y = df['price']
# Split the data into training and testing sets (80% training, 20% testing in this example)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.feature_selection import SelectKBest, f_regression
# Use SelectKBest with the f_regression scoring function
k_best_selector = SelectKBest(score_func=f_regression, k=6)  # Select the top 6 features
X_train_k_best = k_best_selector.fit_transform(X_train, y_train)
X_test_k_best = k_best_selector.transform(X_test)

# Display the selected features
selected_features = X.columns[k_best_selector.get_support()]

selected_features

X_train_k_best = pd.DataFrame(X_train_k_best, columns= selected_features)
X_test_k_best = pd.DataFrame(X_test_k_best, columns= selected_features)

X_test['fuel-type']

X_train_k_best

X_train_k_best['make'] = X_train['make'].values
X_test_k_best['make'] = X_test['make'].values

X_train_k_best['fuel-type'] = X_train['fuel-type'].values
X_test_k_best['fuel-type'] = X_test['fuel-type'].values

X_train_k_best

X_test_k_best

avg_values = X_train_k_best.mean(axis=0)
print("Average values of selected features:")
for feature, avg_value in zip(selected_features, avg_values):
    print(f"{feature}: {avg_value}")

X_train_k_best

X_test_k_best.isna().any()

X_train_k_best.shape

X_test_k_best.shape

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.linear_model import HuberRegressor, PassiveAggressiveRegressor, BayesianRidge
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import NuSVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Elastic Net': ElasticNet(),
    'Decision Tree Regression': DecisionTreeRegressor(),
    'Random Forest Regression': RandomForestRegressor(),
    'Gradient Boosting Regression': GradientBoostingRegressor(),
    'K-Nearest Neighbors Regression': KNeighborsRegressor(),
    'XGBoost Regression': XGBRegressor(),
    'AdaBoost Regression': AdaBoostRegressor(),
    'Bagging Regression': BaggingRegressor(),
    'Extra Trees Regression': ExtraTreesRegressor(),
    'Huber Regression': HuberRegressor(),
    'Passive Aggressive Regression': PassiveAggressiveRegressor(),
    'Bayesian Ridge Regression': BayesianRidge(),
    'MLP Regression': MLPRegressor(),
    'Gaussian Process Regression': GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel())
}
results = []
# Train and evaluate each model
for model_name, model in models.items():
    # Train the model
    model.fit(X_train_k_best, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test_k_best)

    # Calculate RMSE
    rmse = sqrt(mean_squared_error(y_test, y_pred))

    # Calculate R2 score
    r2 = r2_score(y_test, y_pred)

    # Append results to the list
    results.append({
        'Model': model_name,
        'RMSE': rmse,
        'R2 Score': r2
    })

from tabulate import tabulate
results_df = pd.DataFrame(results)

# Print the results table
print(tabulate(results_df, headers='keys', tablefmt='fancy_grid', showindex=False))

# print(X_train_k_best.columns)

model = GradientBoostingRegressor()
model.fit(X_train_k_best, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test_k_best)

# Calculate RMSE
rmse = sqrt(mean_squared_error(y_test, y_pred))

# Calculate R2 score
r2 = r2_score(y_test, y_pred)

print("rmse : " , rmse , "r2 score : " , r2)

import pickle
# Specify the file path where you want to save the pickle file
pickle_file_path = 'model.pkl'

# Open the file in binary write mode and save the data
with open(pickle_file_path, 'wb') as file:
    pickle.dump(model, file)


# fitting the size of the plot
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 8))
# plotting the graphs
plt.plot([i for i in range(len(y_test))],y_test, color = 'green',label="actual values")
plt.plot([i for i in range(len(y_test))],y_pred, color='red', label="Predicted values")
# showing the plotting
plt.legend()
plt.show()

model

vals = [
    168.8,
    64.1,
    2548,
    130,
    21,
    27,
    0 , 1
]

input_data = pd.DataFrame([vals], columns=['length', 'width', 'curb-weight', 'engine-size', 'city-mpg', 'highway-mpg' , 'make' , 'fuel-type'])

input_data

predicted_value = model.predict(
    input_data
)
# predicted_value = sc.inverse_transform(predicted_value.reshape(-1, 1))
predicted_value

