import yaml
from pathlib import Path

def load_config(name: str = "basic_config") -> dict:
    path = Path(__file__).parents[2] / "configs" / f"{name}.yml"
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config