import dagshub
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dagshub.init(
    repo_owner='202201573',
    repo_name='mlops-ci-cd-pipeline',
    mlflow=True,
    token=os.environ.get("DAGSHUB_TOKEN")
)

data = pd.read_csv("data.csv")

X = data[["feature1", "feature2"]]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

np.random.seed(42)
flip_indices = np.random.choice(len(y_pred), size=int(0.4 * len(y_pred)), replace=False)
y_pred[flip_indices] = 1 - y_pred[flip_indices]

accuracy = accuracy_score(y_test, y_pred)

with mlflow.start_run() as run:
    mlflow.log_metric("accuracy", accuracy)

    print("Accuracy:", accuracy)
    print("Run ID:", run.info.run_id)

    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)
