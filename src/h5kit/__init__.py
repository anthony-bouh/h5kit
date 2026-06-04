"""
h5kit - Convenience extensions for working with HDF5 files in Python.

This package builds on the h5py library with higher-level wrappers and utilities
for common HDF5 workflows.

Main components:
- File: Enhanced file handling with context management
- Group: Extended group operations with recursive capabilities
- Dataset: Enhanced dataset operations with conversion utilities
- Query: Advanced query and search functionality
- Visualization: Tools for visualizing HDF5 structure
"""

__version__ = '0.1.2'

# Import core components to make them available at the top level
from .file import File, open_file
from .group import Group
from .dataset import Dataset
from .query import query, find, find_by_regex, get_attr
from .iterate import walk, visit
from .convert import to_dataframe, from_dataframe, to_dict, from_dict, to_json, from_json
from .metadata import get_metadata, set_metadata, has_metadata, delete_metadata, clear_metadata
from .visualization import print_structure, print_info
from .utils import is_hdf5_file, get_hdf5_version

# Define what's available when using `from h5kit import *`
__all__ = [
    'File', 'open_file',
    'Group', 'Dataset',
    'query', 'find', 'find_by_regex', 'get_attr',
    'walk', 'visit',
    'to_dataframe', 'from_dataframe', 'to_dict', 'from_dict', 'to_json', 'from_json',
    'get_metadata', 'set_metadata', 'has_metadata', 'delete_metadata', 'clear_metadata',
    'print_structure', 'print_info',
    'is_hdf5_file', 'get_hdf5_version',
]

