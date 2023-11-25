import pandas as p
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import joblib

dataset = p.read_csv("yield_df.csv")
factors = dataset[["average_rain_fall_mm_per_year", "avg_temp"]]
target = dataset["hg/ha_yield"]
X_train, X_test, Y_train, Y_test = train_test_split(
    factors, target, test_size=0.2, random_state=42
)


regressor = LinearRegression()
regressor.fit(X_train, Y_train)

joblib.dump(regressor, "crop_yield_predictor.pkl")
