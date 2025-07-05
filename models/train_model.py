from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
import joblib

def train_random_forest(X_train, y_train, X_test, y_test):
    print("Training with Random Forest...")
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "trained_random_forest_model.pkl")
    print("Random Forest model saved as 'trained_random_forest_model.pkl'")

    # Evaluate
    y_pred = model.predict(X_test)
    thresholded_pred = [1 if p >= 0.5 else 0 for p in y_pred]
    thresholded_y_test = [1 if y >= 0.5 else 0 for y in y_test]
    metrics = {
        "R2 Score": r2_score(y_test, y_pred),
        "Mean Absolute Error": mean_absolute_error(y_test, y_pred),
        "Mean Squared Error": mean_squared_error(y_test, y_pred),
        "Thresholded Accuracy": accuracy_score(thresholded_y_test, thresholded_pred)
    }

    return model, metrics

def train_xgboost(X_train, y_train, X_test, y_test):
    print("Training with XGBoost...")
    model = XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "trained_xgboost_model.pkl")
    print("XGBoost model saved as 'trained_xgboost_model.pkl'")

    # Evaluate
    y_pred = model.predict(X_test)
    thresholded_pred = [1 if p >= 0.5 else 0 for p in y_pred]
    thresholded_y_test = [1 if y >= 0.5 else 0 for y in y_test]
    metrics = {
        "R2 Score": r2_score(y_test, y_pred),
        "Mean Absolute Error": mean_absolute_error(y_test, y_pred),
        "Mean Squared Error": mean_squared_error(y_test, y_pred),
        "Thresholded Accuracy": accuracy_score(thresholded_y_test, thresholded_pred)
    }

    return model, metrics