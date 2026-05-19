from pathlib import Path

import yaml

def load_config():
    """Load configuration from YAML file"""
    current_file = Path(__file__)
    project_root = current_file.parents[2]
    config_path = project_root / 'config' / 'config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
