# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import operator
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset
url = "/home/kimbiaw/project/funlab/AImodels/Data/province.csv"
datas = pd.read_csv(url)

print(datas)
X = datas.iloc[:23, 0].values.reshape(-1, 1)
X_pre = datas.iloc[::, 0].values.reshape(-1, 1)
Y = datas.iloc[:23, 11].dropna().values

print(X, Y)
# plt.scatter(X, Y, s=10)
# plt.show()

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

poly.fit(X_poly, Y)
lin_poly = LinearRegression()
lin_poly.fit(X_poly, Y)


rmse = np.sqrt(mean_squared_error(Y, lin_poly.predict(poly.fit_transform(X))))
r2 = r2_score(Y, lin_poly.predict(poly.fit_transform(X)))
print(rmse)
print(r2)

print(lin_poly.coef_)
print(lin_poly.intercept_)

res = "f(x) = " + str(lin_poly.intercept_)

for i, r in enumerate(lin_poly.coef_):
    res = res + " + {}*x^{} ".format("%.2f" % r, i)

print(res)

# Visualising the Polynomial Regression results
plt.scatter(X, Y, color="blue")

plt.plot(X, lin_poly.predict(poly.fit_transform(X)), color="red")
plt.title("Polynomial Regression")
plt.xlabel("Postiton level")
plt.ylabel("Salary")

plt.show()
joblib.dump(lin_poly, "/home/kimbiaw/project/funlab/AImodels/bangkok/model/polynomial_house_regression_model_bangkok.pkl")
print("โมเดล Polynomial Regression ถูกบันทึกเรียบร้อยแล้ว!")

# บันทึก PolynomialFeatures
# joblib.dump(poly, "/home/kimbiaw/project/funlab/AImodels/bangkok/model/polynomial_house_features_bangkok.pkl")
# print("พหุนาม PolynomialFeatures ถูกบันทึกเรียบร้อยแล้ว!")
