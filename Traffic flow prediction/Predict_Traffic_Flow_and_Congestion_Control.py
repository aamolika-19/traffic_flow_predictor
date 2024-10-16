import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv('traffic_volume_data.csv')

# Convert date_time to datetime
df['date_time'] = pd.to_datetime(df['date_time'])

# Extract additional time features
df['hour'] = df['date_time'].dt.hour
df['month_day'] = df['date_time'].dt.day
df['weekday'] = df['date_time'].dt.weekday
df['month'] = df['date_time'].dt.month
df['year'] = df['date_time'].dt.year

# Define features and target
features = ['is_holiday', 'air_pollution_index', 'humidity', 'wind_speed', 'wind_direction', 'visibility_in_miles', 'dew_point', 'temperature', 'rain_p_h', 'snow_p_h', 'clouds_all', 'weather_type', 'hour', 'month_day', 'weekday', 'month', 'year', 'last_1_hour_traffic', 'last_2_hour_traffic', 'last_3_hour_traffic', 'last_4_hour_traffic', 'last_5_hour_traffic', 'last_6_hour_traffic']
target = 'traffic_volume'

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Preprocessing pipelines
numeric_features = ['air_pollution_index', 'humidity', 'wind_speed', 'wind_direction', 'visibility_in_miles', 'dew_point', 'temperature', 'rain_p_h', 'snow_p_h', 'clouds_all', 'last_1_hour_traffic', 'last_2_hour_traffic', 'last_3_hour_traffic', 'last_4_hour_traffic', 'last_5_hour_traffic', 'last_6_hour_traffic']
categorical_features = ['is_holiday', 'weather_type', 'hour', 'month_day', 'weekday', 'month', 'year']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Model pipeline
from sklearn.ensemble import RandomForestRegressor

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

# Train the model
model.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predict
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")
