# Checkpoints

Model checkpoints saved during training and final trained models.

## Structure

- **{model_name}/_{epoch}_/ - Checkpoints for specific models
- **final/_{model_name}/_ - Final trained models

## Usage

```python
import torch
checkpoint = torch.load('checkpoints/final/my_model.pt')
model.load_state_dict(checkpoint['state_dict'])
```

## Note

Add this directory to `.gitignore` - checkpoints are large and regenerated.
