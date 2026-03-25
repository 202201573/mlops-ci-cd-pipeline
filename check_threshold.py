import mlflow
import os
import sys

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)

accuracy = run.data.metrics.get("accuracy", 0)

print(f"Model Accuracy: {accuracy}")

if accuracy < 0.85:
    print("❌ Accuracy below threshold!")
    sys.exit(1)
else:
    print("✅ Model passed threshold!")
