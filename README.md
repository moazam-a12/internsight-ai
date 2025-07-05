# Intern Performance Prediction Model

This project predicts intern performance using regression models trained on **task completion time** and **feedback**. The model outputs a **continuous score (0–1)**, then classifies interns as:

- `Excel` (score ≥ 0.5)  
- `Struggle` (score < 0.5)

---

## Objective

- Predict a performance score:  
  `0` = poor, `1` = excellent
- Classify interns into:
  - `Excel` (score ≥ 0.5)  
  - `Struggle` (score < 0.5)

**Models Used**:
- Random Forest Regressor  
- XGBoost Regressor

---

## Dataset

A synthetic dataset (12,000 records) simulates realistic intern performance.

| Feature                    | Description                                                  |
|----------------------------|--------------------------------------------------------------|
| `Task_Completion_Time_Hrs` | Time taken to complete tasks (1.0 – 16.5 hrs)                |
| `Feedback_Rating`          | Intern's self-rating (1.0 – 5.0 scale)                       |
| `Attendance_Percentage`    | Not used in modeling (added for realism)                     |
| `Performance`              | Target score (0–1), calculated using normalized features     |

---

## Preprocessing (`utils/preprocess.py`)

- Loads the dataset and selects relevant features (`Task_Completion_Time_Hrs`, `Feedback_Rating`).
- Splits the data into training (80%) and testing (20%) sets.
- Standardizes the features using `StandardScaler` to ensure they are on the same scale.
- Saves the fitted scaler as `scaler.pkl` for use during prediction to avoid data leakage.

---

## Model Training (`models/train_model.py`)

- Trains two regression models: **Random Forest Regressor** and **XGBoost Regressor**.
- **Random Forest** is configured with `n_estimators=50` and `max_depth=5`.
- **XGBoost** is configured with `n_estimators=50` and `max_depth=3`.
- Both models are trained on the scaled training data and saved for future use.

---

## Evaluation Metrics

The table below shows the evaluation metrics for both models:

| Metric               | Random Forest | XGBoost     |
|----------------------|---------------|-------------|
| R² Score             | 0.9829        | 0.9971      |
| Mean Absolute Error  | 0.0216        | 0.0085      |
| Mean Squared Error   | 0.0007        | 0.0001      |
| Threshold Accuracy   | 97.25%        | 98.42%      |

- **R² Score** measures how well the model explains the variance in the data.
- **Mean Absolute Error (MAE)** calculates the average magnitude of errors in predictions.
- **Mean Squared Error (MSE)** penalizes larger errors more heavily.
- **Threshold Accuracy** evaluates the models' ability to classify interns as `Excel` or `Struggle` based on a score threshold of 0.5.

---

## Author

**Moazam**

> “Built with logic, tested with precision, and delivered with ✨chaotic brilliance✨.”
