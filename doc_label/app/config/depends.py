from functools import lru_cache

from .models import AppDirectories, AppSettings, DatasetSettings


@lru_cache()
def get_app_settings():
    return AppSettings()


@lru_cache()
def get_app_dirs():
    return AppDirectories()


@lru_cache()
def get_dataset_settings():
    return DatasetSettings()
