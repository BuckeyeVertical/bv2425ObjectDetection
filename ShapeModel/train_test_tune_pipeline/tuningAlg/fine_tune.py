#this will be an algorithm that fine tunes a YOLO model on synth data
# https://docs.ultralytics.com/integrations/ray-tune/#custom-search-space-example 
# https://docs.ultralytics.com/guides/hyperparameter-tuning/ 

import wandb
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ray import tune

wandb.init(project="YOLO-Tuning", entity="your-entity")

# Load YOLO model
model = YOLO("yolo11n.pt")

# Tune hyperparameters
result_grid = model.tune(
    data="coco8.yaml",
    space={},
    epochs=50,
    use_ray=True)

experiment_path = f"{storage_path}/{exp_name}"
print(f"Loading results from {experiment_path}...")

restored_tuner = tune.Tuner.restore(experiment_path, trainable=train_mnist)
result_grid = restored_tuner.get_results()

for i, result in enumerate(result_grid):
    plt.plot(
        result.metrics_dataframe["training_iteration"],
        result.metrics_dataframe["mean_accuracy"],
        label=f"Trial {i}",
    )

plt.xlabel("Training Iterations")
plt.ylabel("Mean Accuracy")
plt.legend()
plt.show()