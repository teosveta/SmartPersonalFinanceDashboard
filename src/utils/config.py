import os
import yaml
from pathlib import Path


class Config:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.data_dir / "models"
        self.config_dir = self.project_root / "config"

        # Create directories if they don't exist
        for directory in [self.data_dir, self.models_dir,
                          self.data_dir / "raw", self.data_dir / "processed"]:
            directory.mkdir(parents=True, exist_ok=True)

    def load_config(self, config_name):
        config_path = self.config_dir / f"{config_name}.yaml"
        if config_path.exists():
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        return {}

    @property
    def database_url(self):
        return f"sqlite:///{self.data_dir}/finance_dashboard.db"


config = Config()