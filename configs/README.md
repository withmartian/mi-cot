# Configs

Configuration files for experiments, models, and datasets in JSON/YAML format.

## File Structure

- **model_configs/** - Architecture and training configurations
- **dataset_configs/** - Dataset-specific parameters
- **experiment_configs/** - Experiment hyperparameters

## Usage

Configuration files are loaded in experiment scripts:

```python
import json
with open('configs/experiment_configs/my_experiment.json') as f:
    config = json.load(f)
```

## Example Structure

```json
{
  "model": "rpc",
  "dataset": "math_reasoning",
  "batch_size": 32,
  "learning_rate": 0.001,
  "epochs": 10
}
```
