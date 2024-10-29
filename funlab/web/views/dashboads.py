import datetime
import mongoengine as me



from flask import (
    Blueprint,
    render_template,
    url_for,
    redirect,
    request,
    session,
    current_app,
    send_file,
    abort,
)
from flask_login import login_user, logout_user, login_required, current_user

module = Blueprint("dashboard", __name__, url_prefix="/dashboard")

from flask_login import login_user, logout_user, login_required, current_user

from funlab.models import users
module = Blueprint("dashboard", __name__, url_prefix="/dashboard")


@module.route("/",methods=["GET","POST"])
def index():

    year = request.form.get('year')
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import operator
    import joblib

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import PolynomialFeatures
    poly = joblib.load("/home/kimbiaw/project/funlab/funlab/web/views/utils/polynomial_features.pkl")

    ##cuntry
    model_house = joblib.load("/home/kimbiaw/project/funlab/funlab/web/views/utils/polynomial_house_regression_model_country.pkl")
    model_food = joblib.load("/home/kimbiaw/project/funlab/funlab/web/views/utils/polynomial_food_regression_model_country.pkl")
    model_walk = joblib.load("/home/kimbiaw/project/funlab/funlab/web/views/utils/polynomial_walk_regression_model_country.pkl")
    model_living = joblib.load("/home/kimbiaw/project/funlab/funlab/web/views/utils/polynomial_living_regression_model_country.pkl")
    ##cuntry
    
    ##bangkok
    model_house_bangkok = joblib.load("/home/kimbiaw/project/funlab/funlab/web/views/utils/polynomial_house_regression_model_bangkok.pkl")
    model_food_bangkok = joblib.load("/home/kimbiaw/project/funlab/funlab/web/views/utils/polynomial_food_regression_model_bangkok.pkl")
    model_walk_bangkok = joblib.load("/home/kimbiaw/project/funlab/funlab/web/views/utils/polynomial_walk_regression_model_bangkok.pkl")
    model_living_bangkok = joblib.load("/home/kimbiaw/project/funlab/funlab/web/views/utils/polynomial_living_regression_model_bangkok.pkl")
    ##bangkok

    ##sungkha
    model_house_sungkha = joblib.load("/home/kimbiaw/project/funlab/funlab/web/views/utils/polynomial_house_regression_model_sungkha.pkl")
    model_food_sungkha = joblib.load("/home/kimbiaw/project/funlab/funlab/web/views/utils/polynomial_food_regression_model_sungkha.pkl")
    model_walk_sungkha = joblib.load("/home/kimbiaw/project/funlab/funlab/web/views/utils/polynomial_walk_regression_model_sungkha.pkl")
    model_living_sungkha = joblib.load("/home/kimbiaw/project/funlab/funlab/web/views/utils/polynomial_living_regression_model_sungkha.pkl")
    ##sungkha


    ##nakhon
    model_house_nakhon = joblib.load("/home/kimbiaw/project/funlab/funlab/web/views/utils/polynomial_house_regression_model_nakhon.pkl")
    model_food_nakhon = joblib.load("/home/kimbiaw/project/funlab/funlab/web/views/utils/polynomial_food_regression_model_nakhon.pkl")
    model_walk_nakhon = joblib.load("/home/kimbiaw/project/funlab/funlab/web/views/utils/polynomial_walk_regression_model_nakhon.pkl")
    model_living_nakhon = joblib.load("/home/kimbiaw/project/funlab/funlab/web/views/utils/polynomial_living_regression_model_nakhon.pkl")
    ##nakhon


    # 1. Load the dataset
    url = "/home/kimbiaw/project/funlab/funlab/web/views/utils/22222(22222) .csv"
    url2 = "/home/kimbiaw/project/funlab/AImodels/Data/province.csv"
    datas = pd.read_csv(url)
    data_province = pd.read_csv(url2)

    # 2. Define the independent (X) and dependent (Y) variables
    X = datas.iloc[:23, 0].values.reshape(-1, 1)
    X_pre = datas.iloc[:28, 0].values.reshape(-1, 1)  # Using first 23 rows for X (Date)

    #country
    Y_food = datas.iloc[:23, 2].dropna().values
    Y_house = datas.iloc[:23, 3].dropna().values
    Y_walk = datas.iloc[:23, 4].dropna().values  # Corresponding Y values for house
    Y_living = datas.iloc[:23, 1].dropna().values
    # 3. Transform X using the saved PolynomialFeatures
    X_poly = poly.transform(X_pre)  # แปลง X ให้เป็นรูปแบบ Polynomial
    # 4. ทำนายค่าจากโมเดลที่บันทึกไว้
    Y_pred_food = model_food.predict(X_poly)
    Y_pred_house = model_house.predict(X_poly)
    Y_pred_walk = model_walk.predict(X_poly)
    Y_pred_living = model_living.predict(X_poly)
    #country

    #bangkok
    Y_food_bangkok = data_province.iloc[:23, 2].dropna().values
    Y_house_bangkok = data_province.iloc[:23, 3].dropna().values
    Y_walk_bangkok = data_province.iloc[:23, 4].dropna().values  # Corresponding Y values for house
    Y_living_bangkok = data_province.iloc[:23, 1].dropna().values
    # 3. Transform X using the saved PolynomialFeatures
    X_poly = poly.transform(X_pre)  # แปลง X ให้เป็นรูปแบบ Polynomial
    # 4. ทำนายค่าจากโมเดลที่บันทึกไว้
    Y_pred_food_bangkok = model_food_bangkok.predict(X_poly)
    Y_pred_house_bangkok = model_house_bangkok.predict(X_poly)
    Y_pred_walk_bangkok = model_walk_bangkok.predict(X_poly)
    Y_pred_living_bangkok = model_living_bangkok.predict(X_poly)
    #bangkok

    #sungkha
    Y_food_sungkha = data_province.iloc[:23, 6].dropna().values
    Y_house_sungkha = data_province.iloc[:23, 7].dropna().values
    Y_walk_sungkha = data_province.iloc[:23, 8].dropna().values  # Corresponding Y values for house
    Y_living_sungkha = data_province.iloc[:23, 5].dropna().values
    # 3. Transform X using the saved PolynomialFeatures
    X_poly = poly.transform(X_pre)  # แปลง X ให้เป็นรูปแบบ Polynomial
    # 4. ทำนายค่าจากโมเดลที่บันทึกไว้
    Y_pred_food_sungkha = model_food_sungkha .predict(X_poly)
    Y_pred_house_sungkha = model_house_sungkha .predict(X_poly)
    Y_pred_walk_sungkha = model_walk_sungkha .predict(X_poly)
    Y_pred_living_sungkha = model_living_sungkha .predict(X_poly)
    #sungkha

    #nakhon
    Y_food_nakhon = data_province.iloc[:23, 10].dropna().values
    Y_house_nakhon = data_province.iloc[:23, 11].dropna().values
    Y_walk_nakhon = data_province.iloc[:23, 12].dropna().values  # Corresponding Y values for house
    Y_living_nakhon = data_province.iloc[:23, 9].dropna().values
    # 3. Transform X using the saved PolynomialFeatures
    X_poly = poly.transform(X_pre)  # แปลง X ให้เป็นรูปแบบ Polynomial
    # 4. ทำนายค่าจากโมเดลที่บันทึกไว้
    Y_pred_food_nakhon = model_food_nakhon.predict(X_poly)
    Y_pred_house_nakhon = model_house_nakhon.predict(X_poly)
    Y_pred_walk_nakhon = model_walk_nakhon.predict(X_poly)
    Y_pred_living_nakhon = model_living_nakhon.predict(X_poly)
    #nakhon

    # 5. แสดงกราฟ
    # plt.figure(figsize=(10, 6))

    # # Scatter plot of actual data (ดาต้าจริง)
    # plt.scatter(X, Y_food, color="green", label="Actual Data")
    # plt.scatter(X, Y_house, color="orange", label="Actual Data")
    # plt.scatter(X, Y_walk, color="blue", label="Actual Data")

    # # เส้นเชื่อมระหว่างดาต้าจริง
    # plt.plot(X, Y_food, color="green", label="Connected Actual Data", linestyle="dotted")
    # plt.plot(X, Y_house, color="orange", label="Connected Actual Data", linestyle="dotted")
    # plt.plot(X, Y_walk, color="blue", label="Connected Actual Data", linestyle="dotted")

    # # Plot the polynomial regression curve with dashed line (เส้นประ)
    # plt.plot(X_pre, Y_pred_food, color="green", label="Predictions", linestyle="-")
    # plt.plot(X_pre, Y_pred_house, color="orange", label="Predictions", linestyle="-")
    # plt.plot(X_pre, Y_pred_walk, color="red", label="Predictions", linestyle="-")

    # # Add labels and title
    # plt.title("Polynomial Regression (Dashed Line for Prediction)")
    # plt.xlabel("Date")
    # plt.ylabel("House")
    # plt.legend()
    # plt.grid(True)
    list_data_year = []
    list_pre_year = []

    #thai
    list_real_thai_food =[]
    list_real_thai_house =[]
    list_real_thai_walk =[]
    list_real_thai_living = []

    list_pre_thai_food =[]
    list_pre_thai_house =[]
    list_pre_thai_walk =[]
    list_pre_thai_living = []
    #thai

    #bangkok
    list_real_bangkok_food =[]
    list_real_bangkok_house =[]
    list_real_bangkok_walk =[]
    list_real_bangkok_living = []

    list_pre_bangkok_food =[]
    list_pre_bangkok_house =[]
    list_pre_bangkok_walk =[]
    list_pre_bangkok_living = []
    #bangkok

    #sungkha
    list_real_sungkha_food =[]
    list_real_sungkha_house =[]
    list_real_sungkha_walk =[]
    list_real_sungkha_living = []

    list_pre_sungkha_food =[]
    list_pre_sungkha_house =[]
    list_pre_sungkha_walk =[]
    list_pre_sungkha_living = []
    #sungkha

    #nakhon
    list_real_nakhon_food =[]
    list_real_nakhon_house =[]
    list_real_nakhon_walk =[]
    list_real_nakhon_living = []

    list_pre_nakhon_food =[]
    list_pre_nakhon_house =[]
    list_pre_nakhon_walk =[]
    list_pre_nakhon_living = []
    #nakhon


    ##year data
    for i in X:
        list_data_year.append(float(i))
    for i in X_pre:
        list_pre_year.append(float(i))

    ##data real plot country
    for i in Y_food:
        list_real_thai_food.append(float(i))
    for i in Y_house:
        list_real_thai_house.append(float(i))
    for i in Y_walk:
        list_real_thai_walk.append(float(i))
    for i in Y_living:
        list_real_thai_living.append(float(i))
    ##data real plot country
    ##data pre plot country
    for i in Y_pred_food:
        list_pre_thai_food.append(float(i))
    for i in Y_pred_house:
        list_pre_thai_house.append(float(i))
    for i in Y_pred_walk:
        list_pre_thai_walk.append(float(i))
    for i in Y_pred_living:
        list_pre_thai_living.append(float(i))
    ##data pre plot country

    ##data real plot bangkok
    for i in Y_food_bangkok:
        list_real_bangkok_food.append(float(i))
    for i in Y_house_bangkok:
        list_real_bangkok_house.append(float(i))
    for i in Y_walk_bangkok:
        list_real_bangkok_walk.append(float(i))
    for i in Y_living_bangkok:
        list_real_bangkok_living.append(float(i))
    ##data real plot bangkok
    ##data pre plot bangkok
    for i in Y_pred_food_bangkok:
        list_pre_bangkok_food.append(float(i))
    for i in Y_pred_house_bangkok:
        list_pre_bangkok_house.append(float(i))
    for i in Y_pred_walk_bangkok:
        list_pre_bangkok_walk.append(float(i))
    for i in Y_pred_living_bangkok:
        list_pre_bangkok_living.append(float(i))
    ##data pre plot bangkok

    ##data real plot sungkha
    for i in Y_food_sungkha:
        list_real_sungkha_food.append(float(i))
    for i in Y_house_sungkha:
        list_real_sungkha_house.append(float(i))
    for i in Y_walk_sungkha:
        list_real_sungkha_walk.append(float(i))
    for i in Y_living_sungkha:
        list_real_sungkha_living.append(float(i))
    ##data real plot sungkha
    ##data pre plot sungkha
    for i in Y_pred_food_sungkha:
        list_pre_sungkha_food.append(float(i))
    for i in Y_pred_house_sungkha:
        list_pre_sungkha_house.append(float(i))
    for i in Y_pred_walk_sungkha:
        list_pre_sungkha_walk.append(float(i))
    for i in Y_pred_living_sungkha:
        list_pre_sungkha_living.append(float(i))
    ##data pre plot sungkha

    ##data real plot nakhon
    for i in Y_food_nakhon:
        list_real_nakhon_food.append(float(i))
    for i in Y_house_nakhon:
        list_real_nakhon_house.append(float(i))
    for i in Y_walk_nakhon:
        list_real_nakhon_walk.append(float(i))
    for i in Y_living_nakhon:
        list_real_nakhon_living.append(float(i))
    ##data real plot nakhon
    ##data pre plot nakhon
    for i in Y_pred_food_nakhon:
        list_pre_nakhon_food.append(float(i))
    for i in Y_pred_house_nakhon:
        list_pre_nakhon_house.append(float(i))
    for i in Y_pred_walk_nakhon:
        list_pre_nakhon_walk.append(float(i))
    for i in Y_pred_living_nakhon:
        list_pre_nakhon_living.append(float(i))
    ##data pre plot nakhon
    # print(list_pre_nakhon_living)
    import json
    # print(10000)
    # สร้างข้อมูลที่ต้องการจะส่งออก
    if year:
        year = float(year)
        list_pre_year = [(str(int(year)))]  # Convert to integer first, then to string, and finally to a list of characters

    data = {
        "list_pre_year": list_pre_year,
        "list_real_thai_food": list_real_thai_food,
        "list_real_thai_house": list_real_thai_house,
        "list_real_thai_walk": list_real_thai_walk,
        "list_real_thai_living": list_real_thai_living,
        "list_pre_thai_food": list_pre_thai_food,
        "list_pre_thai_house": list_pre_thai_house,
        "list_pre_thai_walk": list_pre_thai_walk,
        "list_pre_thai_living": list_pre_thai_living,
        "list_real_bangkok_food": list_real_bangkok_food,
        "list_real_bangkok_house": list_real_bangkok_house,
        "list_real_bangkok_walk": list_real_bangkok_walk,
        "list_real_bangkok_walk": list_real_bangkok_living,
        "list_pre_bangkok_food": list_pre_bangkok_food,
        "list_pre_bangkok_house": list_pre_bangkok_house,
        "list_pre_bangkok_walk": list_pre_bangkok_walk,
        "list_pre_bangkok_living": list_pre_bangkok_living,
        "list_real_sungkha_food": list_real_sungkha_food,
        "list_real_sungkha_house": list_real_sungkha_house,
        "list_real_sungkha_walk": list_real_sungkha_walk,
        "list_real_sungkha_walk": list_real_sungkha_living,
        "list_pre_sungkha_food": list_pre_sungkha_food,
        "list_pre_sungkha_house": list_pre_sungkha_house,
        "list_pre_sungkha_walk": list_pre_sungkha_walk,
        "list_pre_sungkha_living": list_pre_sungkha_living,
        "list_real_nakhon_food": list_real_nakhon_food,
        "list_real_nakhon_house": list_real_nakhon_house,
        "list_real_nakhon_walk": list_real_nakhon_walk,
        "list_real_nakhon_walk": list_real_nakhon_living,
        "list_pre_nakhon_food": list_pre_nakhon_food,
        "list_pre_nakhon_house": list_pre_nakhon_house,
        "list_pre_nakhon_walk": list_pre_nakhon_walk,
        "list_pre_nakhon_living": list_pre_nakhon_living,
    }
    # data = list(filter(lambda d: d.list_pre_year == year1, data))
    # print(data)

    return render_template("dashboards/dashboards.html", data=data ,year=year)

