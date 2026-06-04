"""
Structure visualization tools for h5kit.

This module provides structure visualization tools for h5kit, including
ASCII structure visualization, dataset statistics, and memory usage analysis.
"""

import os
import h5py
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, Callable, TypeVar, Generic, Set, TextIO

from .core import H5Object


def print_structure(obj: H5Object, max_depth: Optional[int] = None, file: TextIO = None, show_attrs: bool = False) -> None:
    """
    Print the structure of an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to print the structure of.
    max_depth : Optional[int], optional
        The maximum depth to print, by default None (no limit).
    file : TextIO, optional
        The file to print to, by default None (print to stdout).
    show_attrs : bool, optional
        Whether to show attributes, by default False.
    """
    _print_structure_helper(obj, 0, max_depth, file, show_attrs)


def _print_structure_helper(obj: H5Object, depth: int, max_depth: Optional[int], file: TextIO, show_attrs: bool) -> None:
    """
    Helper function for print_structure.
    
    Parameters
    ----------
    obj : H5Object
        The object to print the structure of.
    depth : int
        The current depth.
    max_depth : Optional[int]
        The maximum depth to print.
    file : TextIO
        The file to print to.
    show_attrs : bool
        Whether to show attributes.
    """
    from .core import is_h5py_group, is_h5py_dataset
    
    # Check if we've reached the maximum depth
    if max_depth is not None and depth > max_depth:
        return
    
    # Print the object
    indent = '  ' * depth
    
    if is_h5py_dataset(obj.h5py_obj):
        print(f"{indent}{os.path.basename(obj.name)} (Dataset) {obj.shape} {obj.dtype}", file=file)
    elif is_h5py_group(obj.h5py_obj):
        print(f"{indent}{os.path.basename(obj.name)} (Group)", file=file)
    else:
        print(f"{indent}{os.path.basename(obj.name)} (Unknown)", file=file)
    
    # Print attributes
    if show_attrs and obj.attrs:
        attr_indent = indent + '  '
        print(f"{attr_indent}Attributes:", file=file)
        for key, value in obj.attrs.items():
            print(f"{attr_indent}  {key}: {value}", file=file)
    
    # Print children
    if is_h5py_group(obj.h5py_obj):
        for name, child in obj.items():
            _print_structure_helper(child, depth + 1, max_depth, file, show_attrs)


def print_info(obj: H5Object, file: TextIO = None) -> None:
    """
    Print information about an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to print information about.
    file : TextIO, optional
        The file to print to, by default None (print to stdout).
    """
    from .core import is_h5py_file, is_h5py_group, is_h5py_dataset
    
    # Print basic information
    print(f"Name: {obj.name}", file=file)
    
    if is_h5py_file(obj.h5py_obj):
        print(f"Type: File", file=file)
        print(f"Filename: {obj.filename}", file=file)
        print(f"Mode: {obj.mode}", file=file)
        print(f"Driver: {obj.driver}", file=file)
        print(f"Libver: {obj.libver}", file=file)
        print(f"Userblock size: {obj.userblock_size}", file=file)
    elif is_h5py_group(obj.h5py_obj):
        print(f"Type: Group", file=file)
    elif is_h5py_dataset(obj.h5py_obj):
        print(f"Type: Dataset", file=file)
        print(f"Shape: {obj.shape}", file=file)
        print(f"Dtype: {obj.dtype}", file=file)
        print(f"Size: {obj.size}", file=file)
        print(f"Ndim: {obj.ndim}", file=file)
        print(f"Chunks: {obj.chunks}", file=file)
        print(f"Maxshape: {obj.maxshape}", file=file)
        print(f"Fillvalue: {obj.fillvalue}", file=file)
        print(f"Compression: {obj.compression}", file=file)
        print(f"Compression opts: {obj.compression_opts}", file=file)
        print(f"Shuffle: {obj.shuffle}", file=file)
        print(f"Fletcher32: {obj.fletcher32}", file=file)
        print(f"Scaleoffset: {obj.scaleoffset}", file=file)
    else:
        print(f"Type: Unknown", file=file)
    
    # Print attributes
    if obj.attrs:
        print(f"Attributes:", file=file)
        for key, value in obj.attrs.items():
            print(f"  {key}: {value}", file=file)
    
    # Print children
    if is_h5py_group(obj.h5py_obj) or is_h5py_file(obj.h5py_obj):
        print(f"Children: {len(obj.keys())}", file=file)
        for name in obj.keys():
            print(f"  {name}", file=file)


def print_dataset_stats(dataset: 'H5Dataset', file: TextIO = None) -> None:
    """
    Print statistics about an HDF5 dataset.
    
    Parameters
    ----------
    dataset : H5Dataset
        The dataset to print statistics about.
    file : TextIO, optional
        The file to print to, by default None (print to stdout).
    """
    from .core import is_h5py_dataset
    
    if not is_h5py_dataset(dataset.h5py_obj):
        raise TypeError("Object is not a dataset")
    
    # Print basic information
    print(f"Name: {dataset.name}", file=file)
    print(f"Shape: {dataset.shape}", file=file)
    print(f"Dtype: {dataset.dtype}", file=file)
    print(f"Size: {dataset.size}", file=file)
    print(f"Ndim: {dataset.ndim}", file=file)
    
    # Print statistics
    if dataset.dtype.kind in 'iuf':  # Integer, unsigned integer, or float
        data = dataset.to_array()
        print(f"Min: {np.min(data)}", file=file)
        print(f"Max: {np.max(data)}", file=file)
        print(f"Mean: {np.mean(data)}", file=file)
        print(f"Median: {np.median(data)}", file=file)
        print(f"Std: {np.std(data)}", file=file)
        print(f"Non-zero: {np.count_nonzero(data)}", file=file)
        print(f"Unique: {len(np.unique(data))}", file=file)
    
    # Print chunk information
    if dataset.chunks:
        print(f"Chunks: {dataset.chunks}", file=file)
        chunk_info = dataset.get_chunk_info()
        print(f"Chunk dims: {chunk_info['chunk_dims']}", file=file)
        print(f"Total chunks: {chunk_info['total_chunks']}", file=file)
        print(f"Allocated chunks: {chunk_info['allocated_chunks']}", file=file)
        print(f"Allocation ratio: {chunk_info['allocation_ratio']:.2%}", file=file)
    else:
        print(f"Chunks: None", file=file)


def print_memory_usage(obj: H5Object, file: TextIO = None) -> None:
    """
    Print memory usage of an HDF5 object.
    
    Parameters
    ----------
    obj : H5Object
        The object to print memory usage of.
    file : TextIO, optional
        The file to print to, by default None (print to stdout).
    """
    from .core import is_h5py_group, is_h5py_dataset
    
    # Print basic information
    print(f"Memory usage of {obj.name}:", file=file)
    
    # Calculate memory usage
    if is_h5py_dataset(obj.h5py_obj):
        # Calculate dataset size
        size = obj.size * obj.dtype.itemsize
        print(f"  Dataset size: {_format_bytes(size)}", file=file)
        
        # Calculate chunk overhead
        if obj.chunks:
            chunk_info = obj.get_chunk_info()
            chunk_overhead = chunk_info['allocated_chunks'] * np.prod(obj.chunks) * obj.dtype.itemsize
            chunk_overhead -= size
            if chunk_overhead > 0:
                print(f"  Chunk overhead: {_format_bytes(chunk_overhead)}", file=file)
                print(f"  Total size: {_format_bytes(size + chunk_overhead)}", file=file)
    elif is_h5py_group(obj.h5py_obj):
        # Calculate total size of all datasets
        total_size = 0
        total_chunk_overhead = 0
        
        for path, item in obj.walk():
            if is_h5py_dataset(item.h5py_obj):
                # Calculate dataset size
                size = item.size * item.dtype.itemsize
                total_size += size
                
                # Calculate chunk overhead
                if item.chunks:
                    chunk_info = item.get_chunk_info()
                    chunk_overhead = chunk_info['allocated_chunks'] * np.prod(item.chunks) * item.dtype.itemsize
                    chunk_overhead -= size
                    if chunk_overhead > 0:
                        total_chunk_overhead += chunk_overhead
        
        print(f"  Total dataset size: {_format_bytes(total_size)}", file=file)
        if total_chunk_overhead > 0:
            print(f"  Total chunk overhead: {_format_bytes(total_chunk_overhead)}", file=file)
            print(f"  Total size: {_format_bytes(total_size + total_chunk_overhead)}", file=file)


def _format_bytes(size: int) -> str:
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


def print_chunk_allocation(dataset: 'H5Dataset', file: TextIO = None) -> None:
    """
    Print chunk allocation of an HDF5 dataset.
    
    Parameters
    ----------
    dataset : H5Dataset
        The dataset to print chunk allocation of.
    file : TextIO, optional
        The file to print to, by default None (print to stdout).
    """
    from .core import is_h5py_dataset
    
    if not is_h5py_dataset(dataset.h5py_obj):
        raise TypeError("Object is not a dataset")
    
    if not dataset.chunks:
        print(f"Dataset {dataset.name} is not chunked", file=file)
        return
    
    # Print basic information
    print(f"Chunk allocation of {dataset.name}:", file=file)
    print(f"  Shape: {dataset.shape}", file=file)
    print(f"  Chunks: {dataset.chunks}", file=file)
    
    # Get chunk information
    chunk_info = dataset.get_chunk_info()
    print(f"  Chunk dims: {chunk_info['chunk_dims']}", file=file)
    print(f"  Total chunks: {chunk_info['total_chunks']}", file=file)
    print(f"  Allocated chunks: {chunk_info['allocated_chunks']}", file=file)
    print(f"  Allocation ratio: {chunk_info['allocation_ratio']:.2%}", file=file)
    
    # Print chunk allocation map for 2D datasets
    if dataset.ndim == 2 and dataset.shape[0] <= 50 and dataset.shape[1] <= 50:
        print(f"  Chunk allocation map:", file=file)
        
        # Create a map of allocated chunks
        chunk_map = np.zeros(chunk_info['chunk_dims'], dtype=bool)
        
        for chunk_index, _ in dataset.iter_allocated_chunks():
            chunk_map[chunk_index] = True
        
        # Print the map
        for i in range(chunk_info['chunk_dims'][0]):
            line = '  '
            for j in range(chunk_info['chunk_dims'][1]):
                line += '#' if chunk_map[i, j] else '.'
            print(line, file=file)


def print_file_summary(file: 'H5File', file_obj: TextIO = None) -> None:
    """
    Print a summary of an HDF5 file.
    
    Parameters
    ----------
    file : H5File
        The file to print a summary of.
    file_obj : TextIO, optional
        The file to print to, by default None (print to stdout).
    """
    from .core import is_h5py_file, is_h5py_group, is_h5py_dataset
    from .iterate import collect_datasets, collect_groups
    
    if not is_h5py_file(file.h5py_obj):
        raise TypeError("Object is not a file")
    
    # Print basic information
    print(f"Summary of {file.filename}:", file=file_obj)
    print(f"  Mode: {file.mode}", file=file_obj)
    print(f"  Driver: {file.driver}", file=file_obj)
    
    # Count objects
    datasets = collect_datasets(file)
    groups = collect_groups(file)
    
    print(f"  Groups: {len(groups)}", file=file_obj)
    print(f"  Datasets: {len(datasets)}", file=file_obj)
    
    # Calculate total size
    total_size = 0
    total_chunk_overhead = 0
    
    for dataset in datasets:
        # Calculate dataset size
        size = dataset.size * dataset.dtype.itemsize
        total_size += size
        
        # Calculate chunk overhead
        if dataset.chunks:
            chunk_info = dataset.get_chunk_info()
            chunk_overhead = chunk_info['allocated_chunks'] * np.prod(dataset.chunks) * dataset.dtype.itemsize
            chunk_overhead -= size
            if chunk_overhead > 0:
                total_chunk_overhead += chunk_overhead
    
    print(f"  Total dataset size: {_format_bytes(total_size)}", file=file_obj)
    if total_chunk_overhead > 0:
        print(f"  Total chunk overhead: {_format_bytes(total_chunk_overhead)}", file=file_obj)
        print(f"  Total size: {_format_bytes(total_size + total_chunk_overhead)}", file=file_obj)
    
    # Print top-level structure
    print(f"  Top-level structure:", file=file_obj)
    for name, item in file.items():
        if is_h5py_dataset(item.h5py_obj):
            print(f"    {name} (Dataset) {item.shape} {item.dtype}", file=file_obj)
        elif is_h5py_group(item.h5py_obj):
            print(f"    {name} (Group)", file=file_obj)

