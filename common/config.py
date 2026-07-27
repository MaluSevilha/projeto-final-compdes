from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CIFAR10_DIR = DATA_DIR / "cifar-10-batches-py"
CIFAR100_DIR = DATA_DIR / "cifar-100-python"
