"""
Core functionality and base classes for h5kit.

This module provides the foundation for the h5kit package, including base classes
and core functionality that is used throughout the package.
"""

import os
import h5py
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, Callable, TypeVar, Generic

# Type variables for generic typing
T = TypeVar('T')
PathLike = Union[str, bytes, os.PathLike]

# Registry to track open files
_open_files: Dict[str, 'H5File'] = {}

class H5Attributes:
    """Wrapper for HDF5 attributes."""
    
    def __init__(self, h5py_attrs) -> None:
        """
        Initialize an H5Attributes object.
        
        Parameters
        ----------
        h5py_attrs : h5py.AttributeManager
            The h5py attributes to wrap.
        """
        self._attrs = h5py_attrs
    
    def __getitem__(self, key) -> Any:
        """Get an attribute value."""
        return self._attrs[key]

    def __setitem__(self, key, value) -> None:
        """Set an attribute value."""
        self._attrs[key] = value
    
    def __delitem__(self, key) -> None:
        """Delete an attribute."""
        del self._attrs[key]
    
    def __contains__(self, key) -> bool:
        """Check if an attribute exists."""
        return key in self._attrs
    
    def __iter__(self) -> Iterator:
        """Iterate over attribute names."""
        return iter(self._attrs)
    
    def __len__(self) -> int:
        """Get the number of attributes."""
        return len(self._attrs)
    
    def keys(self) -> List:
        """Get the attribute names."""
        return self._attrs.keys()
    
    def values(self) -> List:
        """Get the attribute values."""
        return self._attrs.values()
    
    def items(self) -> List[Tuple]:
        """Get the attribute items."""
        return self._attrs.items()
    
    def get(self, key, default=None) -> Any:
        """Get an attribute value with a default."""
        try:
            return self[key]
        except KeyError:
            return default


class H5Object:
    """Base class for all h5kit objects."""
    
    def __init__(self, h5py_obj: Any):
        """
        Initialize an H5Object.
        
        Parameters
        ----------
        h5py_obj : Any
            The underlying h5py object.
        """
        self._obj = h5py_obj
    
    @property
    def h5py_obj(self) -> Any:
        """Get the underlying h5py object."""
        return self._obj
    
    @property
    def name(self) -> str:
        """Get the name of the object."""
        return self._obj.name
    
    @property
    def stem(self) -> str:
        """Get the stem (last part) of the object's name."""
        return self._obj.name.rsplit('/', 1)[-1]
    
    @property
    def file(self) -> 'H5File':
        """Get the file containing this object."""
        from .file import H5File
        return H5File(self._obj.file)
    
    @property
    def parent(self) -> 'H5Group':
        """Get the parent group of this object."""
        from .group import H5Group
        parent_obj = self._obj.parent
        if parent_obj is None:
            return None
        return H5Group(parent_obj)
    
    @property
    def attrs(self) -> H5Attributes:
        """Get the attributes of this object."""
        return H5Attributes(self._obj.attrs)
    
    @attrs.setter
    def attrs(self, attrs:H5Attributes) -> None:
        """
        Set multiple attributes at once.
        
        Parameters
        ----------
        attrs : H5Attributes
            The attributes to set.
        """
        for key, value in attrs.items():
            self._obj.attrs[key] = value
    
    def get_attr(self, name: str, default: Any = None) -> Any:
        """
        Get an attribute by name with an optional default value.
        
        Parameters
        ----------
        name : str
            The name of the attribute.
        default : Any, optional
            The default value to return if the attribute doesn't exist.
            
        Returns
        -------
        Any
            The attribute value or the default.
        """
        if name in self._obj.attrs:
            return self._obj.attrs[name]
        return default
    
    def set_attr(self, name: str, value: Any) -> None:
        """
        Set an attribute.
        
        Parameters
        ----------
        name : str
            The name of the attribute.
        value : Any
            The value to set.
        """
        self._obj.attrs[name] = value
    
    def del_attr(self, name: str) -> None:
        """
        Delete an attribute.
        
        Parameters
        ----------
        name : str
            The name of the attribute to delete.
        """
        if name in self._obj.attrs:
            del self._obj.attrs[name]
    
    def has_attr(self, name: str) -> bool:
        """
        Check if an attribute exists.
        
        Parameters
        ----------
        name : str
            The name of the attribute.
            
        Returns
        -------
        bool
            True if the attribute exists, False otherwise.
        """
        return name in self._obj.attrs
    
    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return f"{self.__class__.__name__}({self.name!r})"
    
    def __delitem__(self, key) -> None:
        """Delete a member from the object if applicable."""
        del self._obj[key]


def register_file(file_obj: 'H5File') -> None:
    """
    Register an open file in the global registry.
    
    Parameters
    ----------
    file_obj : H5File
        The file object to register.
    """
    _open_files[file_obj.filename] = file_obj


def unregister_file(file_obj: 'H5File') -> None:
    """
    Unregister a file from the global registry.
    
    Parameters
    ----------
    file_obj : H5File
        The file object to unregister.
    """
    # Find the file in the registry by object identity
    for filename, registered_file in list(_open_files.items()):
        if registered_file is file_obj:
            del _open_files[filename]
            break


def get_open_files() -> List['H5File']:
    """
    Get a list of all open files.
    
    Returns
    -------
    List[H5File]
        A list of all open file objects.
    """
    return list(_open_files.values())


def close_all_files() -> None:
    """Close all open files."""
    for file_obj in list(_open_files.values()):
        file_obj.close()
    _open_files.clear()


def is_h5py_file(obj: Any) -> bool:
    """
    Check if an object is an h5py.File.
    
    Parameters
    ----------
    obj : Any
        The object to check.
        
    Returns
    -------
    bool
        True if the object is an h5py.File, False otherwise.
    """
    return isinstance(obj, h5py.File)


def is_h5py_group(obj: Any) -> bool:
    """
    Check if an object is an h5py.Group.
    
    Parameters
    ----------
    obj : Any
        The object to check.
        
    Returns
    -------
    bool
        True if the object is an h5py.Group, False otherwise.
    """
    return isinstance(obj, h5py.Group)


def is_h5py_dataset(obj: Any) -> bool:
    """
    Check if an object is an h5py.Dataset.
    
    Parameters
    ----------
    obj : Any
        The object to check.
        
    Returns
    -------
    bool
        True if the object is an h5py.Dataset, False otherwise.
    """
    return isinstance(obj, h5py.Dataset)


def wrap_h5py_object(obj: Any) -> Optional[H5Object]:
    """
    Wrap an h5py object in the appropriate h5kit class.
    
    Parameters
    ----------
    obj : Any
        The h5py object to wrap.
        
    Returns
    -------
    Optional[H5Object]
        The wrapped object or None if the object is not an h5py object.
    """
    from .file import H5File
    from .group import H5Group
    from .dataset import H5Dataset
    
    if is_h5py_file(obj):
        return H5File(obj)
    elif is_h5py_group(obj):
        return H5Group(obj)
    elif is_h5py_dataset(obj):
        return H5Dataset(obj)
    else:
        return None

