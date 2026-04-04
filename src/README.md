# Source Library

Core library code for the MI-CoT project. Shared utilities, models, and data processing.

## Structure

### `utils/`
General utilities:
- `helpers.py` - Helper functions
- `metrics.py` - Metric computations
- `plotting.py` - Visualization utilities

### `models/`
Reusable model architectures:
- Base model classes
- Pretrained model wrappers

### `data/`
Data processing and loading:
- Data loaders
- Preprocessing utilities
- Dataset wrappers

## Usage

```python
from src.utils import helpers
from src.models import MyModel
from src.data import DataLoader
```

## Development

Keep library code here separate from experiments for reusability across scripts.
