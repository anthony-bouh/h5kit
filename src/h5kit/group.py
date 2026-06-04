"""
Enhanced group operations for h5kit.

This module provides enhanced group operations, including recursive traversal,
path-based access, and other utilities for working with HDF5 groups.
"""

import os
import h5py
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, Callable, TypeVar, Generic

from .core import H5Object, PathLike


class H5Group(H5Object):
    """
    Enhanced HDF5 group class with additional functionality.
    
    This class wraps an h5py.Group object and provides additional functionality,
    including recursive traversal and path-based access.
    """
    
    def __init__(self, group_obj: h5py.Group):
        """
        Initialize an H5Group.
        
        Parameters
        ----------
        group_obj : h5py.Group
            The h5py.Group object to wrap.
        """
        super().__init__(group_obj)
    
    def __getitem__(self, name: str) -> H5Object:
        """
        Get an item from the group by name.
        
        Parameters
        ----------
        name : str
            The name of the item to get.
            
        Returns
        -------
        H5Object
            The requested item.
            
        Raises
        ------
        KeyError
            If the item doesn't exist.
        """
        from .core import wrap_h5py_object
        
        obj = self._obj[name]
        wrapped = wrap_h5py_object(obj)
        if wrapped is None:
            raise ValueError(f"Unknown object type: {type(obj)}")
        return wrapped
    
    def __contains__(self, name: str) -> bool:
        """
        Check if an item exists in the group.
        
        Parameters
        ----------
        name : str
            The name of the item to check.
            
        Returns
        -------
        bool
            True if the item exists, False otherwise.
        """
        return name in self._obj
    
    def keys(self) -> List[str]:
        """
        Get the keys (names) of items in the group.
        
        Returns
        -------
        List[str]
            The keys of items in the group.
        """
        return list(self._obj.keys())
    
    def values(self) -> List[H5Object]:
        """
        Get the values (items) in the group.
        
        Returns
        -------
        List[H5Object]
            The values of items in the group.
        """
        from .core import wrap_h5py_object
        
        return [wrap_h5py_object(obj) for obj in self._obj.values()]
    
    def items(self) -> List[Tuple[str, H5Object]]:
        """
        Get the items (key-value pairs) in the group.
        
        Returns
        -------
        List[Tuple[str, H5Object]]
            The items in the group.
        """
        from .core import wrap_h5py_object
        
        return [(key, wrap_h5py_object(obj)) for key, obj in self._obj.items()]
    
    def get(self, name: str, default: Any = None) -> Optional[H5Object]:
        """
        Get an item from the group by name with a default value.
        
        Parameters
        ----------
        name : str
            The name of the item to get.
        default : Any, optional
            The default value to return if the item doesn't exist.
            
        Returns
        -------
        Optional[H5Object]
            The requested item or the default value.
        """
        from .core import wrap_h5py_object
        
        if name not in self._obj:
            return default
        
        obj = self._obj[name]
        wrapped = wrap_h5py_object(obj)
        if wrapped is None:
            raise ValueError(f"Unknown object type: {type(obj)}")
        return wrapped
    
    def create_group(self, name: str) -> 'H5Group':
        """
        Create a new group.
        
        Parameters
        ----------
        name : str
            The name of the group to create.
            
        Returns
        -------
        H5Group
            The newly created group.
        """
        return H5Group(self._obj.create_group(name))
    
    def create_dataset(self, name: str, shape=None, dtype=None, data=None, **kwargs) -> 'H5Dataset':
        """
        Create a new dataset.
        
        Parameters
        ----------
        name : str
            The name of the dataset to create.
        shape : Tuple[int, ...], optional
            The shape of the dataset.
        dtype : np.dtype, optional
            The data type of the dataset.
        data : Any, optional
            The data to initialize the dataset with.
        **kwargs
            Additional keyword arguments to pass to h5py.create_dataset.
            
        Returns
        -------
        H5Dataset
            The newly created dataset.
        """
        from .dataset import H5Dataset
        
        return H5Dataset(self._obj.create_dataset(name, shape, dtype, data, **kwargs))
    
    def require_group(self, name: str) -> 'H5Group':
        """
        Return a group, creating it if it doesn't exist.
        
        Parameters
        ----------
        name : str
            The name of the group.
            
        Returns
        -------
        H5Group
            The group.
        """
        return H5Group(self._obj.require_group(name))
    
    def require_dataset(self, name: str, shape, dtype, **kwargs) -> 'H5Dataset':
        """
        Return a dataset, creating it if it doesn't exist.
        
        Parameters
        ----------
        name : str
            The name of the dataset.
        shape : Tuple[int, ...]
            The shape of the dataset.
        dtype : np.dtype
            The data type of the dataset.
        **kwargs
            Additional keyword arguments to pass to h5py.require_dataset.
            
        Returns
        -------
        H5Dataset
            The dataset.
        """
        from .dataset import H5Dataset
        
        return H5Dataset(self._obj.require_dataset(name, shape, dtype, **kwargs))
    
    def walk(self) -> Iterator[Tuple[str, H5Object]]:
        """
        Walk through all items in the group recursively.
        
        This is a more Pythonic alternative to h5py's visit and visititems methods,
        which use callbacks.
        
        Returns
        -------
        Iterator[Tuple[str, H5Object]]
            An iterator yielding (path, object) pairs.
        """
        from .iterate import walk
        
        yield from walk(self)
    
    def copy(self, source: str, dest: str) -> None:
        """
        Copy an object within the group.
        
        Parameters
        ----------
        source : str
            The source path.
        dest : str
            The destination path.
        """
        self._obj.copy(source, dest)
    
    def move(self, source: str, dest: str) -> None:
        """
        Move an object within the group.
        
        Parameters
        ----------
        source : str
            The source path.
        dest : str
            The destination path.
        """
        self._obj.move(source, dest)
    
    def query(self, pattern: str) -> List[H5Object]:
        """
        Query objects in the group using a path pattern.
        
        Parameters
        ----------
        pattern : str
            The path pattern to match.
            
        Returns
        -------
        List[H5Object]
            The matching objects.
        """
        from .query import query
        
        return query(self, pattern)
    
    def print_structure(self, max_depth: int = None) -> None:
        """
        Print the structure of the group.
        
        Parameters
        ----------
        max_depth : int, optional
            The maximum depth to print.
        """
        from .visualization import print_structure
        
        print_structure(self, max_depth)
    
    def print_info(self) -> None:
        """Print information about the group."""
        from .visualization import print_info
        
        print_info(self)
    
    def create_groups_recursive(self, path: str) -> 'H5Group':
        """
        Create a hierarchy of groups recursively.
        
        This is similar to `mkdir -p` in Unix.
        
        Parameters
        ----------
        path : str
            The path to create.
            
        Returns
        -------
        H5Group
            The deepest group created.
        """
        if not path:
            return self
        
        parts = path.strip('/').split('/')
        current = self
        
        for part in parts:
            if not part:
                continue
            current = current.require_group(part)
        
        return current
    
    def get_by_path(self, path: str, default: Any = None) -> Optional[H5Object]:
        """
        Get an object by path.
        
        Parameters
        ----------
        path : str
            The path to the object.
        default : Any, optional
            The default value to return if the object doesn't exist.
            
        Returns
        -------
        Optional[H5Object]
            The object or the default value.
        """
        try:
            return self[path]
        except (KeyError, ValueError):
            return default
    
    def get_all_datasets(self) -> List['H5Dataset']:
        """
        Get all datasets in the group and its subgroups.
        
        Returns
        -------
        List[H5Dataset]
            All datasets.
        """
        from .core import is_h5py_dataset
        
        datasets = []
        
        for path, obj in self.walk():
            if is_h5py_dataset(obj.h5py_obj):
                datasets.append(obj)
        
        return datasets
    
    def get_all_groups(self) -> List['H5Group']:
        """
        Get all groups in the group and its subgroups.
        
        Returns
        -------
        List[H5Group]
            All groups.
        """
        from .core import is_h5py_group
        
        groups = []
        
        for path, obj in self.walk():
            if is_h5py_group(obj.h5py_obj) and not is_h5py_file(obj.h5py_obj):
                groups.append(obj)
        
        return groups
    
    def delete(self, name: str) -> None:
        """
        Delete an item from the group.
        
        Parameters
        ----------
        name : str
            The name of the item to delete.
        """
        del self._obj[name]
    
    def clear(self) -> None:
        """Delete all items in the group."""
        for name in list(self._obj.keys()):
            del self._obj[name]


# Alias for backward compatibility
Group = H5Group

