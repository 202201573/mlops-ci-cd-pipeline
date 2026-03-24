import mlflow
import random

mlflow.set_tracking_uri("file:./mlruns")

with mlflow.start_run() as run:
    accuracy = 0.80  # change later to 0.90

    mlflow.log_metric("accuracy", accuracy)

    print("Accuracy:", accuracy)
    print("Run ID:", run.info.run_id)

    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)