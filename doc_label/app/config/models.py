from functools import lru_cache, partial
from pathlib import Path

import platformdirs
from pydantic import AnyUrl, Field
from pydantic_settings import BaseSettings
from pydantic_settings.main import SettingsConfigDict

from .settings import APP_NAME


class BaseAppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.dev"),
        env_file_encoding="utf-8",
        extra="allow",
    )


class AppSettings(BaseAppSettings):
    app_base_url: AnyUrl = "http://localhost:8000"


class AppDirectories(BaseAppSettings):
    cache_dir: Path = Field(default_factory=partial(platformdirs.user_cache_path, appname=APP_NAME))
    data_dir: Path = Field(default_factory=partial(platformdirs.user_data_path, appname=APP_NAME))


class DatasetSettings(BaseAppSettings):
    image_render_dpi: int = Field(default=200, ge=1)
