import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_preprocess_data(filepath, target_column, test_size=0.2, random_state=42):
    df = pd.read_csv(filepath)

    # Features (only Task_Completion_Time_Hrs and Feedback_Rating)
    features = df[['Task_Completion_Time_Hrs', 'Feedback_Rating']]
    target = df[target_column]  # Performance is now numerical (0 to 1)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, "scaler.pkl")
    print("Scaler saved as 'scaler.pkl'")

    return X_train_scaled, X_test_scaled, y_train, y_test