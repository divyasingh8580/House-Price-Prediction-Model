# House Price Prediction — End-to-End Machine Learning Project

This project predicts the median house value using the California Housing dataset.

## Project flow
1. Load dataset
2. Exploratory Data Analysis (EDA)
3. Split train/test data
4. Handle missing values
5. Feature scaling
6. Train multiple regression models
7. Compare models
8. Evaluate using MAE, RMSE and R²
9. Save the best model
10. Build a Flask web application
11. Run the project locally
12. Upload the project to GitHub

## Dataset
The project uses `sklearn.datasets.fetch_california_housing`, so you do not need to manually download a CSV. The first run downloads the dataset and caches it locally.

## Features
- MedInc: median income
- HouseAge: median house age
- AveRooms: average rooms
- AveBedrms: average bedrooms
- Population: population
- AveOccup: average occupants
- Latitude
- Longitude

## Target
`MedHouseVal` — median house value in units of $100,000.

For example, a prediction of `3.50` means approximately `$350,000`.

## Installation

```bash
python -m venv venv
```

Windows PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```

Windows CMD:
```cmd
venv\Scripts\activate
```

Install packages:
```bash
pip install -r requirements.txt
```

## Train the model

```bash
python train.py
```

This creates:
- `model/house_price_model.pkl`
- `model/feature_names.pkl`

## Run the Flask app

```bash
python app.py
```

Open:
`http://127.0.0.1:5000`

## Important
The target is in units of $100,000. The web app converts the model output into an approximate dollar value.

## GitHub

Do not upload the virtual environment. The included `.gitignore` prevents that.

```bash
git init
git add .
git commit -m "House price prediction ML project"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```
