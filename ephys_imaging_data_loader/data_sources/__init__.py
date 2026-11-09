"""
data_sources
============

Where recordings come from. Every source turns (root name, relative path)
into a file on local disk that the file readers can open.

- ``LocalFolderDataSource``  — files on this computer
- ``BoxCloudDataSource``     — files on Box, downloaded once into a cache
- ``LocalThenBoxDataSource`` — "auto": local copy if it exists, else Box
- ``open_data_source(mode)`` — builds one of the above from the settings file
"""

from .base_data_source import DataSource, FetchedFile, normalize_relative_path
from .box_cloud_data_source import BoxCloudDataSource
from .box_download_cache import BoxDownloadCache
from .data_source_settings import DataSourceSettings, load_settings, open_data_source
from .local_folder_data_source import LocalFolderDataSource
from .local_then_box_data_source import LocalThenBoxDataSource

__all__ = [
    "DataSource",
    "FetchedFile",
    "normalize_relative_path",
    "LocalFolderDataSource",
    "BoxCloudDataSource",
    "BoxDownloadCache",
    "LocalThenBoxDataSource",
    "DataSourceSettings",
    "load_settings",
    "open_data_source",
]
