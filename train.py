import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:./mlruns")

# Load dataset
data = pd.read_csv("data.csv")

# Features and target
X = data[["feature1", "feature2"]]
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)

# MLflow logging
with mlflow.start_run() as run:
<<<<<<< HEAD
=======
    accuracy = 0.90  

>>>>>>> 7b1392458b00447a5a3674ac5585927a55d2862e
    mlflow.log_metric("accuracy", accuracy)
    
    # These lines must be indented to stay inside the 'with' block
    print("Accuracy:", accuracy)
    print("Run ID:", run.info.run_id)

    # Save Run ID
    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)
