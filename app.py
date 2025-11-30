#importing all needed libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report

from prophet import Prophet

# Streamlit App
st.set_page_config(page_title="OLA Ride Prediction App", layout="wide")

st.title("🚕 OLA Ride Prediction & Forecasting Dashboard")

# LOAD DATA and reading dataset
file_path = "C://Users//Harish//OneDrive//Desktop//ML Project//ola.csv"
df = pd.read_csv(file_path)

#sidebar to see raw data
st.sidebar.header("Dataset Information")
if st.sidebar.checkbox("Show Raw Data",key="raw"):
    st.write(df.head())
    st.write(df.tail())


#Clearing data and preprocessing
df = df.drop_duplicates()
df = df.fillna(0)
df = df.round(0)

#Converting the column to datetime objects
df['datetime'] = pd.to_datetime(df['datetime'])

# Extract year, month, and day
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day

# Extract time components
df['hour'] = df['datetime'].dt.hour
df['minute'] = df['datetime'].dt.minute
df['second'] = df['datetime'].dt.second

st.sidebar.header("Dataset Information After cleaning")
if st.sidebar.checkbox("Show Cleaned Data",key="filter"):
    st.write(df.head())
    st.write(df.tail())
    
# Create the search bar
search_query = st.text_input("Search by location or weather_name", value="")

# Filter the DataFrame based on the search query
if search_query:
    filtered_df = df[
        df['location'].str.contains(search_query, case=False) |
        df['weather_name'].str.contains(search_query, case=False)
    ]
else:
    filtered_df = df      #Whole dataset will visible

# Display the results
if not filtered_df.empty:
    st.dataframe(filtered_df)
else:
    st.write("No results found.")

st.header("📊 Exploratory Data Analysis (EDA)")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Temperature Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['temp'], kde=True, ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Count Boxplot")
    fig, ax = plt.subplots()
    sns.boxplot(y=df['count'], ax=ax)
    st.pyplot(fig)

st.subheader("Humidity vs Count Scatterplot")
fig, ax = plt.subplots()
sns.scatterplot(x='humidity', y='count', data=df, ax=ax)
st.pyplot(fig)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# ---------------------------------------------
# RANDOM FOREST REGRESSION
# ---------------------------------------------
st.header("🌲 Random Forest Regression")

X = df.drop(['temp', 'datetime', 'count'], axis=1)
y = df['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

y_pred = rf_reg.predict(X_test)

st.subheader("Regression Metrics:")
st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.2f}")
st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.2f}")
st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

st.subheader("Feature Importance")
importance = pd.Series(rf_reg.feature_importances_, index=X.columns)
fig, ax = plt.subplots()
importance.sort_values().plot(kind='barh', color='green', ax=ax)
st.pyplot(fig)

# ---------------------------------------------
# RANDOM FOREST CLASSIFICATION
# ---------------------------------------------
st.header("🔍 Random Forest Classification (High Demand Prediction)")

y_class = (df['count'] > df['count'].mean()).astype(int)
X_class = df.drop(['count', 'datetime'], axis=1)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

clf = RandomForestClassifier()
clf.fit(X_train_c, y_train_c)
y_pred_c = clf.predict(X_test_c)

st.subheader("Classification Report")
st.text(classification_report(y_test_c, y_pred_c))

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test_c, y_pred_c)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# ---------------------------------------------
# PROPHET FORECASTING
# ---------------------------------------------
st.header("📈 Time Series Forecasting using Prophet")

prophet_df = df.rename(columns={'datetime': 'ds', 'count': 'y'})[['ds', 'y']]
model = Prophet()
model.fit(prophet_df)

forecast_type = st.selectbox("Select Forecast Type", ["Daily (30 days)", "Hourly (48 hours)"])

if forecast_type == "Daily (30 days)":
    future = model.make_future_dataframe(periods=30, freq='D')
else:
    future = model.make_future_dataframe(periods=48, freq='H')

forecast = model.predict(future)

st.subheader("Forecast Plot")
fig1 = model.plot(forecast)
st.pyplot(fig1)

st.subheader("Forecast Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

st.success("✅ Forecasting Completed!")

