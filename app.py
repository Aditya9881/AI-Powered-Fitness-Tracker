import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Customizing the Sidebar
st.sidebar.image("https://marketplace.canva.com/EAFxdcos7WU/1/0/1600w/canva-dark-blue-and-brown-illustrative-fitness-gym-logo-oqe3ybeEcQQ.jpg", use_container_width=True)
st.sidebar.markdown(
    "**Track your fitness and calorie burn with real-time predictions!**"
)

# Title of the web app
st.title("AI-Powered Fitness Tracker")

st.write(
    "This AI model predicts **calories burned** based on your body parameters."
    " Adjust the sliders below to see your predicted calories burned!"
)

# Sidebar for user input
st.sidebar.header("Enter Your Parameters:")

def user_input_features():
    age = st.sidebar.slider("Age (years):", 10, 100, 25)
    bmi = st.sidebar.slider("BMI:", 15, 40, 22)
    duration = st.sidebar.slider("Duration (minutes):", 0, 60, 20)
    heart_rate = st.sidebar.slider("Heart Rate (bpm):", 60, 180, 90)
    body_temp = st.sidebar.slider("Body Temperature (°C):", 35, 42, 37)
    gender = st.sidebar.radio("Gender:", ["Male", "Female"])
    
    # Convert gender to numerical
    gender_male = 1 if gender == "Male" else 0
    gender_female = 1 if gender == "Female" else 0
    
    input_data = pd.DataFrame({
        "Age": [age],
        "BMI": [bmi],
        "Duration": [duration],
        "Heart_Rate": [heart_rate],
        "Body_Temp": [body_temp],
        "Gender_Male": [gender_male],
        "Gender_Female": [gender_female]
    })
    
    return input_data

user_data = user_input_features()

# Display user input
st.write("## Your Input Parameters:")
st.dataframe(user_data)

# Progress Bar - Simulating Data Processing
st.write("🔄 **Processing your data...**")
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
    time.sleep(0.01)

# Load and preprocess dataset
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

# Merge datasets
df = exercise.merge(calories, on="User_ID")
df.drop(columns=["User_ID"], inplace=True)

# Compute BMI for dataset
df["BMI"] = round(df["Weight"] / ((df["Height"] / 100) ** 2), 2)

# Feature selection
df = df[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]

# One-hot encoding for gender
df = pd.get_dummies(df, drop_first=True)

# Train/Test Split
X = df.drop("Calories", axis=1)
y = df["Calories"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train model
model = RandomForestRegressor(n_estimators=1000, max_depth=6, max_features=3, random_state=42)
model.fit(X_train, y_train)

# Ensure columns match
user_data = user_data.reindex(columns=X_train.columns, fill_value=0)

# Make predictions
predicted_calories = model.predict(user_data)[0]
calories_per_minute = round(predicted_calories / user_data["Duration"].values[0], 2) if user_data["Duration"].values[0] != 0 else 0

# Display predictions
st.success(" **Prediction Complete!**")

st.write("##  Calories Burned Prediction:")
st.metric(label="Predicted Calories Burned", value=f"{round(predicted_calories, 2)} kcal")

st.write(f" **Calories Burned Per Minute:** {calories_per_minute} kcal/min")

# Find similar results
similar_data = df[(df["Calories"] >= predicted_calories - 10) & (df["Calories"] <= predicted_calories + 10)]
st.write("## Similar Cases in Dataset:")
st.dataframe(similar_data.sample(min(5, len(similar_data))))

# Extra Insights
st.write("---")
st.write("## Health Insights Based on Your Data")

def percentile_calc(column_name, user_value):
    return round((sum(df[column_name] < user_value) / len(df)) * 100, 2)

st.write(f" You are older than **{percentile_calc('Age', user_data['Age'].values[0])}%** of people in the dataset.")
st.write(f" Your exercise duration is **longer** than **{percentile_calc('Duration', user_data['Duration'].values[0])}%** of other users.")
st.write(f" Your heart rate is higher than **{percentile_calc('Heart_Rate', user_data['Heart_Rate'].values[0])}%** of other users.")
st.write(f" Your body temperature is higher than **{percentile_calc('Body_Temp', user_data['Body_Temp'].values[0])}%** of people in the dataset.")

# Plot Data for Visualization
st.write("---")
st.write("##  Data Visualization")

fig, ax = plt.subplots()
sns.histplot(df["Calories"], bins=30, kde=True, ax=ax)
ax.axvline(predicted_calories, color='r', linestyle='--', label="Your Prediction")
ax.set_title("Distribution of Calories Burned")
ax.set_xlabel("Calories Burned")
ax.set_ylabel("Count")
ax.legend()
st.pyplot(fig)

st.write("###  Stay fit and keep tracking your fitness goals! ")

