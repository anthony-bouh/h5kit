"""
Metadata handling utilities for h5kit.

This module provides metadata handling utilities for h5kit, including
schema definition and validation, extended attribute types, and metadata inheritance.
"""

import os
import h5py
import numpy as np
import json
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, Callable, TypeVar, Generic, Set

from .core import H5Object


def get_metadata(obj: H5Object, key: Optional[str] = None, default: Any = None) -> Any:
    """
    Get metadata from an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to get metadata from.
    key : Optional[str], optional
        The metadata key to get, by default None (get all metadata).
    default : Any, optional
        The default value to return if the key doesn't exist.
        
    Returns
    -------
    Any
        The metadata value or the default.
    """
    # Check if the object has a metadata attribute
    if 'metadata' not in obj.attrs:
        if key is None:
            return {}
        return default
    
    # Get the metadata
    try:
        metadata = json.loads(obj.attrs['metadata'])
    except (json.JSONDecodeError, TypeError):
        if key is None:
            return {}
        return default
    
    # Return the metadata
    if key is None:
        return metadata
    
    return metadata.get(key, default)


def set_metadata(obj: H5Object, key: str, value: Any) -> None:
    """
    Set metadata on an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to set metadata on.
    key : str
        The metadata key to set.
    value : Any
        The metadata value to set.
    """
    # Get the current metadata
    metadata = get_metadata(obj)
    
    # Update the metadata
    metadata[key] = value
    
    # Set the metadata
    obj.attrs['metadata'] = json.dumps(metadata)


def delete_metadata(obj: H5Object, key: str) -> None:
    """
    Delete metadata from an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to delete metadata from.
    key : str
        The metadata key to delete.
    """
    # Get the current metadata
    metadata = get_metadata(obj)
    
    # Delete the metadata
    if key in metadata:
        del metadata[key]
    
    # Set the metadata
    obj.attrs['metadata'] = json.dumps(metadata)


def clear_metadata(obj: H5Object) -> None:
    """
    Clear all metadata from an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to clear metadata from.
    """
    if 'metadata' in obj.attrs:
        del obj.attrs['metadata']


def has_metadata(obj: H5Object, key: str) -> bool:
    """
    Check if an HDF5 object has metadata.
    
    Parameters
    ----------
    obj : H5Object
        The object to check.
    key : str
        The metadata key to check.
        
    Returns
    -------
    bool
        True if the object has the metadata key, False otherwise.
    """
    metadata = get_metadata(obj)
    return key in metadata


def get_metadata_keys(obj: H5Object) -> List[str]:
    """
    Get the metadata keys of an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to get metadata keys from.
        
    Returns
    -------
    List[str]
        The metadata keys.
    """
    metadata = get_metadata(obj)
    return list(metadata.keys())


def copy_metadata(source: H5Object, target: H5Object, keys: Optional[List[str]] = None) -> None:
    """
    Copy metadata from one HDF5 object to another.
    
    Parameters
    ----------
    source : H5Object
        The object to copy metadata from.
    target : H5Object
        The object to copy metadata to.
    keys : Optional[List[str]], optional
        The metadata keys to copy, by default None (copy all metadata).
    """
    # Get the source metadata
    source_metadata = get_metadata(source)
    
    # Get the target metadata
    target_metadata = get_metadata(target)
    
    # Copy the metadata
    if keys is None:
        keys = list(source_metadata.keys())
    
    for key in keys:
        if key in source_metadata:
            target_metadata[key] = source_metadata[key]
    
    # Set the target metadata
    target.attrs['metadata'] = json.dumps(target_metadata)


def inherit_metadata(obj: H5Object, recursive: bool = False) -> None:
    """
    Inherit metadata from parent objects.
    
    Parameters
    ----------
    obj : H5Object
        The object to inherit metadata to.
    recursive : bool, optional
        Whether to recursively inherit metadata to child objects, by default False.
    """
    # Get the parent
    parent = obj.parent
    
    # If there's no parent, there's nothing to inherit
    if parent is None:
        return
    
    # Copy metadata from the parent
    copy_metadata(parent, obj)
    
    # Recursively inherit metadata to child objects
    if recursive:
        from .core import is_h5py_group
        
        if is_h5py_group(obj.h5py_obj):
            for child in obj.values():
                inherit_metadata(child, recursive)


def propagate_metadata(obj: H5Object, keys: Optional[List[str]] = None) -> None:
    """
    Propagate metadata to child objects.
    
    Parameters
    ----------
    obj : H5Object
        The object to propagate metadata from.
    keys : Optional[List[str]], optional
        The metadata keys to propagate, by default None (propagate all metadata).
    """
    from .core import is_h5py_group
    
    if is_h5py_group(obj.h5py_obj):
        for child in obj.values():
            copy_metadata(obj, child, keys)
            propagate_metadata(child, keys)


def validate_metadata_schema(obj: H5Object, schema: Dict[str, Any]) -> bool:
    """
    Validate metadata against a schema.
    
    Parameters
    ----------
    obj : H5Object
        The object to validate metadata for.
    schema : Dict[str, Any]
        The schema to validate against.
        
    Returns
    -------
    bool
        True if the metadata is valid, False otherwise.
    """
    try:
        import jsonschema
    except ImportError:
        raise ImportError("jsonschema is required for validate_metadata_schema()")
    
    # Get the metadata
    metadata = get_metadata(obj)
    
    # Validate the metadata
    try:
        jsonschema.validate(metadata, schema)
        return True
    except jsonschema.exceptions.ValidationError:
        return False


def set_metadata_schema(obj: H5Object, schema: Dict[str, Any]) -> None:
    """
    Set a metadata schema on an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to set the schema on.
    schema : Dict[str, Any]
        The schema to set.
    """
    obj.attrs['metadata_schema'] = json.dumps(schema)


def get_metadata_schema(obj: H5Object) -> Optional[Dict[str, Any]]:
    """
    Get the metadata schema of an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to get the schema from.
        
    Returns
    -------
    Optional[Dict[str, Any]]
        The schema or None if there is no schema.
    """
    if 'metadata_schema' not in obj.attrs:
        return None
    
    try:
        return json.loads(obj.attrs['metadata_schema'])
    except (json.JSONDecodeError, TypeError):
        return None


def validate_metadata(obj: H5Object) -> bool:
    """
    Validate metadata against the object's schema.
    
    Parameters
    ----------
    obj : H5Object
        The object to validate metadata for.
        
    Returns
    -------
    bool
        True if the metadata is valid, False otherwise.
    """
    # Get the schema
    schema = get_metadata_schema(obj)
    
    # If there's no schema, the metadata is valid
    if schema is None:
        return True
    
    # Validate the metadata
    return validate_metadata_schema(obj, schema)

