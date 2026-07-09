# 🚕 OLA Ride Prediction & Forecasting Dashboard

A Machine Learning dashboard built with **Streamlit** that analyzes OLA ride booking data, performs exploratory data analysis (EDA), predicts ride demand using **Random Forest**, and forecasts future bookings using **Facebook Prophet**.

---

## 📌 Features

### 📊 Data Analysis
- View raw dataset
- Data cleaning and preprocessing
- Search rides by:
  - Location
  - Weather condition
- Automatic feature extraction from datetime
  - Year
  - Month
  - Day
  - Hour
  - Minute
  - Second

---

### 📈 Exploratory Data Analysis (EDA)

Visualizations include:

- Temperature Distribution
- Booking Count Boxplot
- Humidity vs Booking Count Scatter Plot
- Correlation Heatmap

---

### 🌲 Random Forest Regression

Predict ride booking counts using:

- Random Forest Regressor

Performance Metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

Also displays:

- Feature Importance Graph

---

### 🔍 Random Forest Classification

Predict whether booking demand is **High** or **Low**.

Outputs:

- Classification Report
- Confusion Matrix

---

### 📈 Time Series Forecasting

Uses **Facebook Prophet** to forecast future bookings.

Supports:

- Daily Forecast (30 Days)
- Hourly Forecast (48 Hours)

Displays:

- Forecast Plot
- Trend Components
- Seasonality

---

### 🔮 Future Booking Prediction

Users can select any future date and obtain:

- Predicted bookings
- Lower confidence interval
- Upper confidence interval

---

# 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Prophet

---

# 📂 Project Structure

```
ML Project/
│
├── app.py
├── ola.csv
├── requirements.txt
├── README.md
└── screenshots/
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/your-username/OLA-Ride-Prediction.git

cd OLA-Ride-Prediction
```

---

## 2. Create Virtual Environment (Recommended)

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux/Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run the Application

```bash
streamlit run app.py
```

---

# 📦 Requirements

```text
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
prophet
```

Or install manually

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn prophet
```

---

# 📊 Machine Learning Models

## Regression

- Random Forest Regressor

Predicts:

- Ride Booking Count

---

## Classification

- Random Forest Classifier

Predicts:

- High Demand
- Low Demand

---

## Forecasting

- Facebook Prophet

Forecasts:

- Daily booking trend
- Hourly booking trend
- Future booking estimation

---

# 📷 Dashboard Preview

You can add screenshots here.

```
screenshots/
    dashboard.png
    regression.png
    forecasting.png
```

Example:

```markdown
![Dashboard](screenshots/dashboard.png)
```

---

# 📈 Dataset

The dataset contains ride booking information including:

- Datetime
- Temperature
- Humidity
- Weather
- Location
- Ride Count

---

# 🚀 Future Improvements

- Deploy on Streamlit Cloud
- User Authentication
- Live OLA API Integration
- Interactive Map
- XGBoost & LightGBM Models
- Hyperparameter Tuning
- Deep Learning Forecasting (LSTM)
- Download Prediction Reports
- Dark Mode UI

---

#  How to Run

1. Clone the Repository

   git clone https://github.com/your-username/OLA-Ride-Prediction.git
   cd OLA-Ride-Prediction
2. Create a Virtual Environment (Recommended)
   python -m venv venv

   Activate it : venv\Scripts\activate

3.Install all requirements (Required libraries)

    Create a file named requirements.txt with the following contents:

   1.streamlit
   2.pandas
   3.numpy
   4.matplotlib
   5.seaborn
   6.scikit-learn
   7.prophet

   Then install everything with:

   pip install -r requirements.txt


   Your project folder should look like:

   OLA-Ride-Prediction/
│
├── app.py
├── ola.csv
├── requirements.txt
├── README.md
└── screenshots/

6. Run the Streamlit App

    streamlit run app.py

   If the command isn't recognized on Windows, use:

   python -m streamlit run app.py

   
   


# 👨‍💻 Author

**Harish Singh Saud**

GitHub: https://github.com/singhsaudharish

LinkedIn: https://linkedin.com/in/harishsingh-saud

---

# ⭐ If you like this project

Please give it a ⭐ on GitHub!

It motivates me to build more Machine Learning projects.

---

# 📜 License

This project is licensed under the MIT License.
