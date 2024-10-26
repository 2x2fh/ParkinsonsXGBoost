import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier, XGBModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    data = pd.read_csv("parkinsons.data")
    X = data.drop(columns=['status', 'name'])
    y = data['status']

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    param_dist = {
        'n_estimators': np.arange(100, 1000, 100),  # Increased range
        'learning_rate': np.linspace(0.01, 0.3, 20),  # More values
        'max_depth': np.arange(3, 15, 2),  # Wider range
        'subsample': np.linspace(0.5, 1.0, 6),  # Adjusted range
        'colsample_bytree': np.linspace(0.5, 1.0, 6),
        'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],  # Additional value
        'min_child_weight': [1, 2, 3, 4, 5],
        'reg_alpha': [0, 0.01, 0.1, 1, 10],  # More values
        'reg_lambda': [0, 0.01, 0.1, 1, 10]
    }

    model = XGBClassifier(eval_metric='logloss', scale_pos_weight=1)

    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                       n_iter=100, cv=5, n_jobs=-1, verbose=2, scoring='accuracy', random_state=42)

    random_search.fit(X_train, y_train)

    print("Best Parameters:", random_search.best_params_)
    print("Best Cross-Validation Accuracy:", random_search.best_score_)

    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on Test Set: {accuracy * 100:.2f}%")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Classification Report:")
    print(classification_report(y_test, y_pred))


