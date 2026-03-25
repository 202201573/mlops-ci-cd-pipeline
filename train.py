import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("file:./mlruns")

data = pd.read_csv("data.csv")

X = data[["feature1", "feature2"]]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

with mlflow.start_run() as run:
    mlflow.log_metric("accuracy", accuracy)
    
    # These lines must be indented to stay inside the 'with' block
    print("Accuracy:", accuracy)
    print("Run ID:", run.info.run_id)

    # Save Run ID
    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)
