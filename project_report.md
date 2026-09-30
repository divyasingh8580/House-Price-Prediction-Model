# House Price Prediction — Project Report

## 1. Title
House Price Prediction Using Machine Learning

## 2. Problem Statement
The objective is to develop a machine learning regression model that predicts the median value of a house from demographic, geographic and housing-related features.

## 3. Objective
- Understand a real-world regression problem.
- Perform data preprocessing.
- Explore relationships between features and house prices.
- Train and compare multiple regression algorithms.
- Select the best model using evaluation metrics.
- Save the trained model.
- Deploy the model through a Flask web application.

## 4. Dataset
California Housing dataset.

Input features:
1. MedInc
2. HouseAge
3. AveRooms
4. AveBedrms
5. Population
6. AveOccup
7. Latitude
8. Longitude

Target:
- MedHouseVal

## 5. Machine Learning Type
Supervised Learning → Regression.

## 6. Algorithms
- Linear Regression
- Decision Tree Regression
- Random Forest Regression

## 7. Preprocessing
The project uses:
- Median imputation for missing values.
- StandardScaler for numerical feature scaling.
- A scikit-learn Pipeline to keep preprocessing and prediction together.

## 8. Evaluation Metrics

### MAE
Mean Absolute Error measures the average absolute difference between actual and predicted values.

### RMSE
Root Mean Squared Error penalizes larger errors more strongly.

### R² Score
R² measures how much of the variation in the target is explained by the model.

## 9. Model Selection
The script compares all three models and selects the model with the lowest RMSE on the test set.

## 10. Deployment
The trained pipeline is saved using Joblib. Flask loads the saved model and provides a browser-based form for prediction.

## 11. Future Improvements
- Hyperparameter tuning using GridSearchCV/RandomizedSearchCV.
- Cross-validation.
- Better UI.
- Docker deployment.
- Cloud deployment.
- Monitoring and model versioning.
- More advanced ensemble models.

## 12. Conclusion
This project demonstrates a complete machine learning workflow from data loading and preprocessing to model training, evaluation, saving, and web deployment.
