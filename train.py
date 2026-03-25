import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Set MLflow tracking URI from environment
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

with mlflow.start_run() as run:
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    # Save run_id to file
    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)

    print("Accuracy:", acc)
