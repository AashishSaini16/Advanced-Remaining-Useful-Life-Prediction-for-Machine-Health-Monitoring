# Advanced Remaining Useful Life(RUL) Prediction for Machine Health Monitoring

This repository contains a Jupyter notebook for developing a stacking ensemble model to predict the Remaining Useful Life (RUL) of industrial machines using sensor data from the AI4I 2020 Predictive Maintenance Dataset. The project leverages XGBoost, RandomForest, LightGBM, SHAP for interpretability, and an interactive widget for real-time predictions, demonstrating preprocessing, model training, evaluation, and deployment-ready insights. From a business perspective, this system can help manufacturers reduce unplanned downtime (estimated at $50B+ annually industry-wide), optimize maintenance schedules, extend asset life, and enhance operational efficiency, potentially cutting maintenance costs by 10-20% through proactive interventions.

### Dataset Source
The dataset is sourced from the [UCI Machine Learning Repository: AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset). It includes sensor readings (e.g., air temperature, process temperature, rotational speed, torque, tool wear) and failure indicators for machine types (L, M, H).

### Key Features
* Data Preprocessing: Handles categorical encoding of machine types (L/M/H) and removes non-predictive columns (UID, Product ID).
* Feature Engineering: Introduces physics-informed features like temperature delta (process temp - air temp) and power (torque × rotational speed) to enhance model performance.
* Model Ensemble: Combines XGBoost, RandomForest, and LightGBM via a stacking regressor with an XGBoost meta-learner, optimized for RUL prediction.
* Interpretability: Uses SHAP for feature importance and individual prediction explanations, with visualizations like summary and dependence plots.
* Interactive Tool: Provides an interactive widget for real-time RUL predictions based on user-input sensor values.

### Technologies Used
* Python: Core programming language.
* Pandas & NumPy: Data manipulation and numerical computations.
* Scikit-learn: Model training, evaluation, and stacking.
* XGBoost & LightGBM: Gradient boosting models with GPU support for efficiency.
* Optuna: Hyperparameter optimization (pre-tuned parameters included).
* Seaborn & Matplotlib: Data visualization for EDA and model insights.
* SHAP: Model interpretability and feature importance analysis.
* ipywidgets: Interactive interface for real-time RUL predictions.
* Joblib: Model persistence for saving and loading.

### Project Workflow
* Data Loading and EDA: Loads the AI4I 2020 dataset directly from UCI. Performs exploratory data analysis (EDA) with histograms and, correlation matrix.
![Histograms of Numerical Features](https://github.com/AashishSaini16/Advanced-Remaining-Useful-Life-Prediction-for-Machine-Health-Monitoring/blob/main/Histograms%20of%20Numerical%20Features.PNG)
![Correlation Matrix](https://github.com/AashishSaini16/Advanced-Remaining-Useful-Life-Prediction-for-Machine-Health-Monitoring/blob/main/Correlation%20Matrix.PNG)

* Data Preprocessing and Feature Engineering: Drops non-predictive columns, encodes 'Type' column, creates derived features (Temp_delta, Power_W), and defines RUL based on tool wear threshold and failure status.

* Model Training: Trains base models (XGBoost with monotonic constraints, RandomForest, LightGBM) and combines them using a stacking regressor with 5-fold CV.
![Feature Importances from XGBoost](https://github.com/AashishSaini16/Advanced-Remaining-Useful-Life-Prediction-for-Machine-Health-Monitoring/blob/main/Feature%20Importances%20from%20XGBoost.PNG)

* Model Evaluation: Evaluates the stacking model on a test set using MAE (2.00), RMSE (8.38), R² (0.9849), and MAPE (0.0205, excluding zero RUL cases). Includes ablation study showing ensemble improvement over standalone XGBoost.
![Actual vs Predicted RUL](https://github.com/AashishSaini16/Advanced-Remaining-Useful-Life-Prediction-for-Machine-Health-Monitoring/blob/main/Actual%20vs%20Predicted%20RUL.PNG)
![Residual Plot](https://github.com/AashishSaini16/Advanced-Remaining-Useful-Life-Prediction-for-Machine-Health-Monitoring/blob/main/Residual%20Plot.PNG)

* SHAP Analysis for Interpretability: Uses SHAP to analyze feature contributions, with summary plots.
![SHAP Summary Plot](https://github.com/AashishSaini16/Advanced-Remaining-Useful-Life-Prediction-for-Machine-Health-Monitoring/blob/main/SHAP%20Summary%20Plot.PNG)

* Interactive RUL Predictor: Implements an interactive widget with sliders for sensor inputs, input validation, and real-time predictions including derived features.
![Interactive Widget Demo](https://github.com/AashishSaini16/Advanced-Remaining-Useful-Life-Prediction-for-Machine-Health-Monitoring/blob/main/Interactive%20Widget%20Demo.PNG)
interpretability. It provides actionable insights for maintenance scheduling and an interactive tool for real-time predictions, suitable for deployment in industrial settings.
