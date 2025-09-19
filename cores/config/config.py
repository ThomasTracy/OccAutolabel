import os
import sys
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional



class Config:
    def __init__(self, config_path: str, env_prefix: str = "APP_"):
        self.config_path = Path(config_path)
        self.env_prefix = env_prefix
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"File does not exist: {self.config_path}")
        
        if self.config_path.suffix.lower() not in ['.yaml', '.yml']:
            raise ValueError("only support yaml and yml")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        config = self._parse_config(config)
        
        # 环境变量覆盖
        # self._override_with_env_vars(config)
        return config
    
    def _parse_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, dict):
                    self._parse_config(value)
                elif isinstance(value, list):
                    config[key] = np.array(value)
                else:
                    config[key] = value
        return config
    
    def _override_with_env_vars(self, config: Dict[str, Any], prefix: str = ""):
        """用环境变量覆盖配置"""
        current_prefix = f"{self.env_prefix}{prefix}"
        
        for key, value in config.items():
            env_key = f"{current_prefix}{key.upper()}"
            
            if isinstance(value, dict):
                self._override_with_env_vars(value, f"{prefix}{key.upper()}_")
            elif env_key in os.environ:
                # 基本类型转换
                env_value = os.environ[env_key]
                if isinstance(value, bool):
                    config[key] = env_value.lower() in ('true', '1', 'yes')
                elif isinstance(value, int):
                    config[key] = int(env_value)
                elif isinstance(value, float):
                    config[key] = float(env_value)
                else:
                    config[key] = env_value
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self._config[key]