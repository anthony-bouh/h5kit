"""
Enhanced iteration utilities for h5kit.

This module provides enhanced iteration utilities for h5kit, including
Pythonic iterators for recursive traversal of HDF5 files and groups.
"""

import os
import h5py
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, Callable, TypeVar, Generic, Set

from .core import H5Object


def walk(obj: H5Object, topdown: bool = True, include_self: bool = False) -> Iterator[Tuple[str, H5Object]]:
    """
    Walk through all items in an HDF5 object recursively.
    
    This is a more Pythonic alternative to h5py's visit and visititems methods,
    which use callbacks.
    
    Parameters
    ----------
    obj : H5Object
        The object to walk through.
    topdown : bool, optional
        If True, yield parent groups before their children, by default True.
    include_self : bool, optional
        If True, include the object itself in the results, by default False.
        
    Returns
    -------
    Iterator[Tuple[str, H5Object]]
        An iterator yielding (path, object) pairs.
    """
    from .core import is_h5py_group, wrap_h5py_object
    
    # Include the object itself if requested
    if include_self:
        yield obj.name, obj
    
    # Get the items in the object
    if is_h5py_group(obj.h5py_obj):
        items = list(obj.items())
        
        # Process the items
        for name, item in items:
            # Get the full path
            path = f"{obj.name}/{name}".replace("//", "/")
            
            # Yield the item if topdown
            if topdown:
                yield path, item
            
            # Recursively walk through groups
            if is_h5py_group(item.h5py_obj):
                yield from walk(item, topdown, False)
            
            # Yield the item if not topdown
            if not topdown:
                yield path, item


def visit(obj: H5Object, func: Callable[[str, H5Object], Any], topdown: bool = True, include_self: bool = False) -> List[Any]:
    """
    Visit all items in an HDF5 object recursively and apply a function to each.
    
    Parameters
    ----------
    obj : H5Object
        The object to visit.
    func : Callable[[str, H5Object], Any]
        The function to apply to each item.
    topdown : bool, optional
        If True, visit parent groups before their children, by default True.
    include_self : bool, optional
        If True, include the object itself in the results, by default False.
        
    Returns
    -------
    List[Any]
        The results of applying the function to each item.
    """
    results = []
    
    for path, item in walk(obj, topdown, include_self):
        results.append(func(path, item))
    
    return results


def find(obj: H5Object, name: str, max_depth: Optional[int] = None) -> List[H5Object]:
    """
    Find items by name in an HDF5 object.
    
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
        The matching items.
    """
    results = []
    
    def _find_helper(path: str, item: H5Object, depth: int = 0) -> None:
        # Check if we've reached the maximum depth
        if max_depth is not None and depth > max_depth:
            return
        
        # Check if the item name matches
        if os.path.basename(path) == name:
            results.append(item)
        
        # Recursively search in groups
        from .core import is_h5py_group
        if is_h5py_group(item.h5py_obj):
            for subpath, subitem in item.items():
                _find_helper(f"{path}/{subpath}", subitem, depth + 1)
    
    _find_helper(obj.name, obj)
    
    return results


def find_by_attr(obj: H5Object, attr_name: str, attr_value: Any = None, max_depth: Optional[int] = None) -> List[H5Object]:
    """
    Find items by attribute in an HDF5 object.
    
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
        The matching items.
    """
    results = []
    
    def _find_helper(path: str, item: H5Object, depth: int = 0) -> None:
        # Check if we've reached the maximum depth
        if max_depth is not None and depth > max_depth:
            return
        
        # Check if the item has the attribute
        if attr_name in item.attrs:
            # Check if the attribute value matches
            if attr_value is None or item.attrs[attr_name] == attr_value:
                results.append(item)
        
        # Recursively search in groups
        from .core import is_h5py_group
        if is_h5py_group(item.h5py_obj):
            for subpath, subitem in item.items():
                _find_helper(f"{path}/{subpath}", subitem, depth + 1)
    
    _find_helper(obj.name, obj)
    
    return results


def collect_paths(obj: H5Object, topdown: bool = True, include_self: bool = False) -> List[str]:
    """
    Collect all paths in an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to collect paths from.
    topdown : bool, optional
        If True, collect parent groups before their children, by default True.
    include_self : bool, optional
        If True, include the object itself in the results, by default False.
        
    Returns
    -------
    List[str]
        The paths.
    """
    return [path for path, _ in walk(obj, topdown, include_self)]


def collect_objects(obj: H5Object, topdown: bool = True, include_self: bool = False) -> List[H5Object]:
    """
    Collect all objects in an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to collect objects from.
    topdown : bool, optional
        If True, collect parent groups before their children, by default True.
    include_self : bool, optional
        If True, include the object itself in the results, by default False.
        
    Returns
    -------
    List[H5Object]
        The objects.
    """
    return [item for _, item in walk(obj, topdown, include_self)]


def collect_datasets(obj: H5Object) -> List['H5Dataset']:
    """
    Collect all datasets in an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to collect datasets from.
        
    Returns
    -------
    List[H5Dataset]
        The datasets.
    """
    from .core import is_h5py_dataset
    
    return [item for _, item in walk(obj) if is_h5py_dataset(item.h5py_obj)]


def collect_groups(obj: H5Object, include_self: bool = False) -> List['H5Group']:
    """
    Collect all groups in an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to collect groups from.
    include_self : bool, optional
        If True, include the object itself in the results, by default False.
        
    Returns
    -------
    List[H5Group]
        The groups.
    """
    from .core import is_h5py_group, is_h5py_file
    
    groups = []
    
    if include_self and (is_h5py_group(obj.h5py_obj) or is_h5py_file(obj.h5py_obj)):
        groups.append(obj)
    
    groups.extend([item for _, item in walk(obj) if is_h5py_group(item.h5py_obj) and not is_h5py_file(item.h5py_obj)])
    
    return groups

