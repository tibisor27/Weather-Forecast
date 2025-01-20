# Weather Forecast Application

A weather forecast application that predicts the temperature and humidity for the next 5 hours based on historical data and using **Machine Learning** models. This project uses **Django** for the backend, **HTML**, **CSS**, and **JavaScript** for the frontend, and **Random Forest** and **Linear Regression** for the prediction model, trained using **scikit-learn** and **PyTorch**.

## Description

This application provides users with weather forecasts for the next 5 hours for a chosen location, predicting temperature and humidity based on **Machine Learning** algorithms. The user can enter a city, and the application will display the current weather data along with predictions for the upcoming hours.

The **Random Forest** and **Linear Regression** algorithms are used to predict weather conditions based on a historical dataset. The current weather data is fetched using the **OpenWeatherMap API**, while models are trained using historical data from a CSV file.

## Purpose

The main goal of this project was to learn about prediction models in AI, specifically models that predict weather patterns based on historical data. I chose to work on this project in order to explore and implement different AI and machine learning techniques, such as Random Forest and Linear Regression, and to understand how to predict continuous variables like temperature and humidity.

## Technologies Used

- **Backend**: Django (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning Model**: Random Forest, Linear Regression (scikit-learn, PyTorch)
- **API Used**: OpenWeatherMap (for fetching current weather data)
- **Data Handling**: pandas, numpy
- **Data Scaling**: StandardScaler (scikit-learn)
- **Model Training**: scikit-learn, PyTorch

![WebSite](weatherProject/forecast/static/img/test.png)

