import yaml
import os

# Load and parse the YAML configuration file
def load_settings():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "..", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Global configuration object, imported by other modules
CONFIG = load_settings()