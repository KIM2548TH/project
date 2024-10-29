import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os  # To create folder if not exists
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# # Function to create folder if not exists
# def create_folder_if_not_exists(folder_path):
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#         print(f"Folder '{folder_path}' created.")
#     else:
#         print(f"Folder '{folder_path}' already exists.")

# Function to perform polynomial regression
def polynomial_regression(data, degree, y_column, model_name, features_name):
    """
    Perform Polynomial Regression on the given dataset and save the model and features.
    :param data: Dataset for regression
    :param degree: Degree of the polynomial
    :param y_column: Column index of the dependent variable
    :param model_name: Name of the file to save the model
    :param features_name: Name of the file to save the polynomial features
    """
    X = data.iloc[:23, 0].values.reshape(-1, 1)
    Y = data.iloc[:23, y_column].dropna().values
    
    # Polynomial feature transformation
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # Fitting polynomial regression model
    lin_poly = LinearRegression()
    lin_poly.fit(X_poly, Y)

    # Calculate RMSE and R^2 score
    rmse = np.sqrt(mean_squared_error(Y, lin_poly.predict(poly.fit_transform(X))))
    r2 = r2_score(Y, lin_poly.predict(poly.fit_transform(X)))
    print(f"RMSE: {rmse}")
    print(f"R^2: {r2}")

    # Print coefficients and intercept
    print("Coefficients:", lin_poly.coef_)
    print("Intercept:", lin_poly.intercept_)

    # Display the regression formula
    res = f"f(x) = {lin_poly.intercept_}"
    for i, r in enumerate(lin_poly.coef_):
        res += f" + {r:.2f}*x^{i}"
    print("Regression formula:", res)

    # Visualising the Polynomial Regression results
    plt.scatter(X, Y, color="blue")
    plt.plot(X, lin_poly.predict(poly.fit_transform(X)), color="red")
    plt.title("Polynomial Regression")
    plt.xlabel("Position level")
    plt.ylabel("Value")
    plt.show()

    # Create folder for models if not exists
    folder_path = os.path.dirname(model_name)
    # create_folder_if_not_exists(folder_path)

    # Save the model and features
    joblib.dump(lin_poly, model_name)
    joblib.dump(poly, features_name)
    print(f"Model saved as {model_name}")
    print(f"Polynomial Features saved as {features_name}")

# Load the dataset
data_url = '/home/kimbiaw/project/funlab/AImodels/Data/province.csv'
data = pd.read_csv(data_url)

# Perform Polynomial Regression for Food
polynomial_regression(data, degree=3, y_column=10,
                      model_name="/home/kimbiaw/project/funlab/AImodels/sungkha/model/polynomial_food_regression_model_sungkha.pkl",
                      features_name="/home/kimbiaw/project/funlab/AImodels/sungkha/model/polynomial_food_features_sungkha.pkl")

# Perform Polynomial Regression for House
polynomial_regression(data, degree=3, y_column=11,
                      model_name="/home/kimbiaw/project/funlab/AImodels/sungkha/model/polynomial_house_regression_model.pkl",
                      features_name="/home/kimbiaw/project/funlab/AImodels/sungkha/model/polynomial_house_features.pkl")

# Perform Polynomial Regression for Living
polynomial_regression(data, degree=3, y_column=9,
                      model_name="/home/kimbiaw/project/funlab/AImodels/sungkha/model/polynomial_living_regression_model_living.pkl",
                      features_name="/home/kimbiaw/project/funlab/AImodels/sungkha/model/polynomial_living_features_living.pkl")

# Perform Polynomial Regression for Walk
polynomial_regression(data, degree=3, y_column=12,
                      model_name="/home/kimbiaw/project/funlab/AImodels/sungkha/model/polynomial_regression_model_walk.pkl",
                      features_name="/home/kimbiaw/project/funlab/AImodels/sungkha/model/polynomial_features_walk.pkl")
