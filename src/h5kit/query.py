"""
Query and search functionality for h5kit.

This module provides query and search functionality for h5kit, including
path-based querying with wildcards and attribute-based filtering.
"""

import os
import re
import fnmatch
import h5py
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, Callable, TypeVar, Generic, Set

from .core import H5Object

def get_attr(obj: H5Object, attr_path: str, default=None) -> Any:
    """
    Get an attribute value using a '/' separated path.
    
    Parameters
    ----------
    obj : H5Object
        The object to get the attribute from.
    attr_path : str
        The '/' separated path to the attribute.
        
    Returns
    -------
    Any
        The attribute value.
    """
    parts = attr_path.strip('/').split('/')
    current_obj = obj
    
    for part in parts[:-1]:
        if part in current_obj:
            current_obj = current_obj[part]
        else:
            return default
    
    attr_name = parts[-1]
    if attr_name in current_obj.attrs:
        return current_obj.attrs[attr_name]
    else:
        return default


def query(obj: H5Object, pattern: str) -> List[H5Object]:
    """
    Query objects in an HDF5 object using a path pattern.
    
    The pattern can include wildcards (* and ?) and special patterns:
    - ** matches any number of path segments
    - * matches any characters within a path segment
    - ? matches a single character
    
    Parameters
    ----------
    obj : H5Object
        The object to query.
    pattern : str
        The path pattern to match.
        
    Returns
    -------
    List[H5Object]
        The matching objects.
    """
    from .iterate import walk
    
    # Normalize the pattern
    pattern = pattern.strip('/')
    
    # Convert the pattern to a regex
    regex_pattern = pattern
    regex_pattern = regex_pattern.replace('.', r'\.')
    regex_pattern = regex_pattern.replace('**', r'.*')
    regex_pattern = regex_pattern.replace('*', r'[^/]*')
    regex_pattern = regex_pattern.replace('?', r'.')
    regex_pattern = f'^{regex_pattern}$'
    
    # Compile the regex
    regex = re.compile(regex_pattern)
    
    # Find matching objects
    results = []
    
    for path, item in walk(obj):
        # Normalize the path
        rel_path = path
        if rel_path.startswith(obj.name):
            rel_path = rel_path[len(obj.name):].strip('/')
        
        # Check if the path matches the pattern
        if regex.match(rel_path):
            results.append(item)
    
    return results


def find(obj: H5Object, name: str, max_depth: Optional[int] = None) -> List[H5Object]:
    """
    Find objects by name in an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to search in.
    name : str
        The name to search for.
    max_depth : Optional[int], optional
        The maximum depth to search, by default None (no limit).
        
    Returns
    -------
    List[H5Object]
        The matching objects.
    """
    from .iterate import find as iterate_find
    
    return iterate_find(obj, name, max_depth)


def find_by_attr(obj: H5Object, attr_name: str, attr_value: Any = None, max_depth: Optional[int] = None) -> List[H5Object]:
    """
    Find objects by attribute in an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to search in.
    attr_name : str
        The attribute name to search for.
    attr_value : Any, optional
        The attribute value to match, by default None (match any value).
    max_depth : Optional[int], optional
        The maximum depth to search, by default None (no limit).

    Returns
    -------
    List[H5Object]
        The matching objects.
    """
    from .iterate import find_by_attr as iterate_find_by_attr
    
    return iterate_find_by_attr(obj, attr_name, attr_value, max_depth)


def find_datasets_by_shape(obj: H5Object, shape: Tuple[int, ...]) -> List['H5Dataset']:
    """
    Find datasets by shape in an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to search in.
    shape : Tuple[int, ...]
        The shape to match.
        
    Returns
    -------
    List[H5Dataset]
        The matching datasets.
    """
    from .iterate import collect_datasets
    
    datasets = collect_datasets(obj)
    return [ds for ds in datasets if ds.shape == shape]


def find_datasets_by_dtype(obj: H5Object, dtype: np.dtype) -> List['H5Dataset']:
    """
    Find datasets by data type in an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to search in.
    dtype : np.dtype
        The data type to match.
        
    Returns
    -------
    List[H5Dataset]
        The matching datasets.
    """
    from .iterate import collect_datasets
    
    datasets = collect_datasets(obj)
    return [ds for ds in datasets if ds.dtype == dtype]


def find_datasets_by_dimensions(obj: H5Object, ndim: int) -> List['H5Dataset']:
    """
    Find datasets by number of dimensions in an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to search in.
    ndim : int
        The number of dimensions to match.
        
    Returns
    -------
    List[H5Dataset]
        The matching datasets.
    """
    from .iterate import collect_datasets
    
    datasets = collect_datasets(obj)
    return [ds for ds in datasets if ds.ndim == ndim]


def find_datasets_by_size(obj: H5Object, min_size: Optional[int] = None, max_size: Optional[int] = None) -> List['H5Dataset']:
    """
    Find datasets by size in an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to search in.
    min_size : Optional[int], optional
        The minimum size to match, by default None (no minimum).
    max_size : Optional[int], optional
        The maximum size to match, by default None (no maximum).
        
    Returns
    -------
    List[H5Dataset]
        The matching datasets.
    """
    from .iterate import collect_datasets
    
    datasets = collect_datasets(obj)
    
    if min_size is None and max_size is None:
        return datasets
    
    if min_size is None:
        return [ds for ds in datasets if ds.size <= max_size]
    
    if max_size is None:
        return [ds for ds in datasets if ds.size >= min_size]
    
    return [ds for ds in datasets if min_size <= ds.size <= max_size]


def find_empty_groups(obj: H5Object) -> List['H5Group']:
    """
    Find empty groups in an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to search in.
        
    Returns
    -------
    List[H5Group]
        The empty groups.
    """
    from .iterate import collect_groups
    
    groups = collect_groups(obj)
    return [group for group in groups if len(group.keys()) == 0]


def find_by_regex(obj: H5Object, pattern: str, include_attrs: bool = False) -> List[H5Object]:
    """
    Find objects by regex pattern in an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to search in.
    pattern : str
        The regex pattern to match.
    include_attrs : bool, optional
        Whether to search attribute names as well, by default False.
        
    Returns
    -------
    List[H5Object]
        The matching objects.
    """
    from .iterate import walk
    
    # Compile the regex
    regex = re.compile(pattern)
    
    # Find matching objects
    results = []
    
    for path, item in walk(obj, include_self=True):
        # Check if the path matches the pattern
        if path and regex.search(path):
            results.append(item)
        
        # Check attributes if requested
        if include_attrs:
            for attr_name in item.attrs:
                # Construct a path-like string for the attribute
                attr_path = f"{path}/{attr_name}"
                if regex.search(attr_path):
                    results.append(item)
                    break  # Add the item only once if any attribute matches
    
    return results

def find_attributes(obj: H5Object, regex: str) -> List[Any]:
    """
    Find attributes by regex pattern in an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to search in.
    regex : str
        The regex pattern to match.
        
    Returns
    -------
    List[Any]
        The matching attributes.
    """
    from .iterate import walk
    
    # Compile the regex
    pattern = re.compile(regex)
    
    # Find matching attributes
    results = []
    for path, item in walk(obj, include_self=True):
        for key, value in item.attrs.items():
            if pattern.search(path+'/'+key):
                results.append(key)
                
    return results


def find_by_glob(obj: H5Object, pattern: str) -> List[H5Object]:
    """
    Find objects by glob pattern in an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to search in.
    pattern : str
        The glob pattern to match.
        
    Returns
    -------
    List[H5Object]
        The matching objects.
    """
    from .iterate import walk
    
    # Find matching objects
    results = []
    
    for path, item in walk(obj):
        # Check if the path matches the pattern
        if fnmatch.fnmatch(path, pattern):
            results.append(item)
    
    return results

def find_attr_by_regex(obj: H5Object, attr_pattern: str) -> List[Any]:
    """
    Find attributes value by regex pattern in an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to search in.
    attr_pattern : str
        The regex pattern to match attribute names.
    max_depth : Optional[int], optional
        The maximum depth to search, by default None (no limit).
        
    Returns
    -------
    List[Any]
        The matching attribute values.
    """
    from .iterate import walk
    
    # Compile the regex
    regex = re.compile(attr_pattern)
    
    results = []
    
    for path, item in walk(obj, include_self=True):
        for key, value in item.attrs.items():
            if regex.search(path+'/'+key):
                results.append(value)
    
    return results