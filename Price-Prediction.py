
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns


file_path = r'C:\Users\balav\Downloads\archive\real_estate_dataset.csv'
df = pd.read_csv(file_path)


features = ['Square_Feet', 'Num_Bedrooms', 'Num_Bathrooms',
            'Num_Floors', 'Year_Built', 'Has_Garden',
            'Has_Pool', 'Garage_Size', 'Location_Score', 'Distance_to_Center']
X = df[features]
y = df['Price']


X = X.fillna(X.median())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")
