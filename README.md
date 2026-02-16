# Divvy BikeShare Demand Predictor

## Overview
This GitHub repository contains a machine learning project focused on predicting the total daily demand for Divvy bikes in Chicago. Utilizing time series forecasting techniques, the project implements two neural network models: a standard Long Short-Term Memory (LSTM) network and a Bidirectional LSTM (BiLSTM) network. These models analyze historical bike rental data to forecast future demand, helping optimize bike availability and urban planning.

The dataset is sourced from Divvy's public trip data, incorporating features like weather conditions, holidays, and temporal patterns to improve prediction accuracy.

## Key Features
- **Data Preprocessing**: Cleaning and feature engineering of Divvy trip data, including aggregation to hourly totals and integration of external factors (e.g., temperature, precipitation).
- **Models Implemented**:
  - **LSTM**: A recurrent neural network designed for sequential data, capturing long-term dependencies in bike demand patterns.
  - **Bidirectional LSTM**: An enhanced version that processes data in both forward and backward directions, potentially improving context understanding for better forecasts.
- **Evaluation Metric**: Models are assessed using Mean Squared Error (MSE).
- **Visualization**: Plots for actual vs. predicted demand using Matplotlib.

## Technologies Used
- Python 3.12.7
- Pytorch for model building
- Pandas and NumPy for data manipulation
- Scikit-learn for preprocessing and evaluation
- Jupyter Notebook for exploratory analysis

## Results
- **LSTM Performance**: Achieved an MAE of 69.1964 rentals on test data.
- **BiLSTM Performance**: Improved to an MSE of 66.4501, possibly showing better handling of seasonal trends.
