import yaml

# Load and parse the YAML configuration file
def load_settings():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

# Global configuration object, imported by other modules
CONFIG = load_settings()