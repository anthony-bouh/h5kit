"""
General utility functions for h5kit.

This module provides general utility functions for h5kit, including
file validation, version checking, and other utilities.
"""

import os
import h5py
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, Callable, TypeVar, Generic, Set

from .core import PathLike


def is_hdf5_file(path: PathLike) -> bool:
    """
    Check if a file is an HDF5 file.
    
    Parameters
    ----------
    path : PathLike
        The path to the file.
        
    Returns
    -------
    bool
        True if the file is an HDF5 file, False otherwise.
    """
    try:
        with h5py.File(path, 'r') as f:
            return True
    except (IOError, OSError):
        return False


def get_hdf5_version(path: PathLike) -> str:
    """
    Get the HDF5 version of a file.
    
    Parameters
    ----------
    path : PathLike
        The path to the file.
        
    Returns
    -------
    str
        The HDF5 version.
        
    Raises
    ------
    IOError
        If the file is not an HDF5 file.
    """
    try:
        with h5py.File(path, 'r') as f:
            return f.libver
    except (IOError, OSError) as e:
        raise IOError(f"Failed to open HDF5 file: {e}")


def get_file_size(path: PathLike) -> int:
    """
    Get the size of a file in bytes.
    
    Parameters
    ----------
    path : PathLike
        The path to the file.
        
    Returns
    -------
    int
        The size of the file in bytes.
        
    Raises
    ------
    IOError
        If the file doesn't exist.
    """
    try:
        return os.path.getsize(path)
    except (IOError, OSError) as e:
        raise IOError(f"Failed to get file size: {e}")


def format_bytes(size: int) -> str:
    """
    Format a size in bytes as a human-readable string.
    
    Parameters
    ----------
    size : int
        The size in bytes.
        
    Returns
    -------
    str
        The formatted size.
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if size < 1024 or unit == 'PB':
            return f"{size:.2f} {unit}"
        size /= 1024


def get_optimal_chunk_size(shape: Tuple[int, ...], dtype: np.dtype) -> Tuple[int, ...]:
    """
    Get the optimal chunk size for a dataset.
    
    Parameters
    ----------
    shape : Tuple[int, ...]
        The shape of the dataset.
    dtype : np.dtype
        The data type of the dataset.
        
    Returns
    -------
    Tuple[int, ...]
        The optimal chunk size.
    """
    # Target chunk size: 1 MB
    target_size = 1024 * 1024
    
    # Calculate the item size
    item_size = dtype.itemsize
    
    # Calculate the number of items per chunk
    items_per_chunk = target_size // item_size
    
    # Calculate the chunk shape
    ndim = len(shape)
    
    if ndim == 0:
        return ()
    
    if ndim == 1:
        chunk_shape = (min(shape[0], items_per_chunk),)
    else:
        # Start with a chunk that's the same shape as the dataset
        chunk_shape = list(shape)
        
        # Calculate the total number of items
        total_items = np.prod(chunk_shape)
        
        # If the total number of items is less than the target, we're done
        if total_items <= items_per_chunk:
            return tuple(chunk_shape)
        
        # Otherwise, reduce the chunk size
        reduction_factor = (items_per_chunk / total_items) ** (1 / ndim)
        
        for i in range(ndim):
            chunk_shape[i] = max(1, int(chunk_shape[i] * reduction_factor))
        
        # Make sure the chunk size is not larger than the dataset
        for i in range(ndim):
            chunk_shape[i] = min(chunk_shape[i], shape[i])
    
    return tuple(chunk_shape)


def validate_file(path: PathLike) -> bool:
    """
    Validate an HDF5 file.
    
    Parameters
    ----------
    path : PathLike
        The path to the file.
        
    Returns
    -------
    bool
        True if the file is valid, False otherwise.
    """
    try:
        with h5py.File(path, 'r') as f:
            # Check if the file is readable
            f.keys()
            return True
    except (IOError, OSError):
        return False


def repair_file(path: PathLike, output_path: Optional[PathLike] = None) -> bool:
    """
    Repair an HDF5 file.
    
    Parameters
    ----------
    path : PathLike
        The path to the file.
    output_path : Optional[PathLike], optional
        The path to save the repaired file to, by default None (overwrite the original).
        
    Returns
    -------
    bool
        True if the file was repaired, False otherwise.
    """
    if output_path is None:
        output_path = path + '.repaired'
    
    try:
        # Try to open the file
        with h5py.File(path, 'r') as f_in:
            # Create a new file
            with h5py.File(output_path, 'w') as f_out:
                # Copy all objects
                for key in f_in.keys():
                    f_in.copy(key, f_out)
        
        # If we got here, the file was repaired
        if output_path != path:
            os.rename(output_path, path)
        
        return True
    except (IOError, OSError):
        return False


def copy_file(source: PathLike, target: PathLike) -> bool:
    """
    Copy an HDF5 file.
    
    Parameters
    ----------
    source : PathLike
        The path to the source file.
    target : PathLike
        The path to the target file.
        
    Returns
    -------
    bool
        True if the file was copied, False otherwise.
    """
    try:
        # Open the source file
        with h5py.File(source, 'r') as f_in:
            # Create the target file
            with h5py.File(target, 'w') as f_out:
                # Copy all objects
                for key in f_in.keys():
                    f_in.copy(key, f_out)
                
                # Copy all attributes
                for key, value in f_in.attrs.items():
                    f_out.attrs[key] = value
        
        return True
    except (IOError, OSError):
        return False


def compare_files(file1: PathLike, file2: PathLike) -> bool:
    """
    Compare two HDF5 files.
    
    Parameters
    ----------
    file1 : PathLike
        The path to the first file.
    file2 : PathLike
        The path to the second file.
        
    Returns
    -------
    bool
        True if the files are identical, False otherwise.
    """
    try:
        # Open the files
        with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
            # Compare the keys
            if set(f1.keys()) != set(f2.keys()):
                return False
            
            # Compare the attributes
            if set(f1.attrs.keys()) != set(f2.attrs.keys()):
                return False
            
            for key in f1.attrs.keys():
                if f1.attrs[key] != f2.attrs[key]:
                    return False
            
            # Compare the objects
            for key in f1.keys():
                if not _compare_objects(f1[key], f2[key]):
                    return False
        
        return True
    except (IOError, OSError):
        return False


def _compare_objects(obj1: h5py.HLObject, obj2: h5py.HLObject) -> bool:
    """
    Compare two HDF5 objects.
    
    Parameters
    ----------
    obj1 : h5py.HLObject
        The first object.
    obj2 : h5py.HLObject
        The second object.
        
    Returns
    -------
    bool
        True if the objects are identical, False otherwise.
    """
    # Check if the objects are the same type
    if type(obj1) != type(obj2):
        return False
    
    # Compare the attributes
    if set(obj1.attrs.keys()) != set(obj2.attrs.keys()):
        return False
    
    for key in obj1.attrs.keys():
        if obj1.attrs[key] != obj2.attrs[key]:
            return False
    
    # Compare groups
    if isinstance(obj1, h5py.Group):
        # Compare the keys
        if set(obj1.keys()) != set(obj2.keys()):
            return False
        
        # Compare the objects
        for key in obj1.keys():
            if not _compare_objects(obj1[key], obj2[key]):
                return False
    
    # Compare datasets
    elif isinstance(obj1, h5py.Dataset):
        # Compare the shape
        if obj1.shape != obj2.shape:
            return False
        
        # Compare the dtype
        if obj1.dtype != obj2.dtype:
            return False
        
        # Compare the data
        if not np.array_equal(obj1[...], obj2[...]):
            return False
    
    return True


def get_memory_usage(obj: h5py.HLObject) -> int:
    """
    Get the memory usage of an HDF5 object in bytes.
    
    Parameters
    ----------
    obj : h5py.HLObject
        The object to get the memory usage of.
        
    Returns
    -------
    int
        The memory usage in bytes.
    """
    # Calculate memory usage
    if isinstance(obj, h5py.Dataset):
        # Calculate dataset size
        size = obj.size * obj.dtype.itemsize
        
        # Calculate chunk overhead
        if obj.chunks:
            dcpl = obj.id.get_create_plist()
            
            if hasattr(dcpl, 'get_num_chunks'):
                # Use the HDF5 1.10.5+ API if available
                space = obj.id.get_space()
                allocated_chunks = dcpl.get_num_chunks(space)
                chunk_overhead = allocated_chunks * np.prod(obj.chunks) * obj.dtype.itemsize
                chunk_overhead -= size
                if chunk_overhead > 0:
                    size += chunk_overhead
        
        return size
    elif isinstance(obj, h5py.Group):
        # Calculate total size of all objects
        total_size = 0
        
        for key in obj.keys():
            total_size += get_memory_usage(obj[key])
        
        return total_size
    else:
        return 0

