"""
Data conversion utilities for h5kit.

This module provides data conversion utilities for h5kit, including
conversion between HDF5 and other formats like pandas DataFrames and dictionaries.
"""

import os
import h5py
import numpy as np
import json
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, Callable, TypeVar, Generic, Set

from .core import H5Object, PathLike


def to_dataframe(dataset: 'H5Dataset', index_col: Optional[str] = None) -> 'pd.DataFrame':
    """
    Convert an HDF5 dataset to a pandas DataFrame.
    
    Parameters
    ----------
    dataset : H5Dataset
        The dataset to convert.
    index_col : Optional[str], optional
        The name of the column to use as the index.
        
    Returns
    -------
    pd.DataFrame
        The dataset as a pandas DataFrame.
    """
    return dataset.to_dataframe(index_col)


def from_dataframe(df: 'pd.DataFrame', file: Union[h5py.File, str, PathLike], name: str, **kwargs) -> 'H5Dataset':
    """
    Convert a pandas DataFrame to an HDF5 dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to convert.
    file : Union[h5py.File, str, PathLike]
        The HDF5 file to store the dataset in.
    name : str
        The name of the dataset.
    **kwargs
        Additional keyword arguments to pass to h5py.create_dataset.
        
    Returns
    -------
    H5Dataset
        The created dataset.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for from_dataframe()")
    
    from .file import H5File
    from .dataset import H5Dataset
    
    # Open the file if it's a path
    if isinstance(file, (str, os.PathLike)):
        file = H5File(file, 'a')
    elif isinstance(file, h5py.File):
        file = H5File(file)
    
    # Convert the DataFrame to a structured array
    if df.index.name is not None:
        df = df.reset_index()
    
    data = df.to_records(index=False)
    
    # Create the dataset
    dataset = file.create_dataset(name, data=data, **kwargs)
    
    return dataset


def to_dict(obj: H5Object, include_attrs: bool = True, include_data: bool = True) -> Dict[str, Any]:
    """
    Convert an HDF5 object to a dictionary.
    
    Parameters
    ----------
    obj : H5Object
        The object to convert.
    include_attrs : bool, optional
        Whether to include attributes, by default True.
    include_data : bool, optional
        Whether to include dataset data, by default True.
        
    Returns
    -------
    Dict[str, Any]
        The object as a dictionary.
    """
    from .core import is_h5py_group, is_h5py_dataset
    
    result = {}
    
    # Add attributes
    if include_attrs:
        result['attrs'] = dict(obj.attrs)
    
    # Add data or children
    if is_h5py_dataset(obj.h5py_obj):
        if include_data:
            data = obj.h5py_obj[...]
            
            # Convert bytes to strings
            if data.dtype.kind == 'S':
                data = np.array([s.decode('utf-8') for s in data.flatten()]).reshape(data.shape)
            
            # Convert to Python types
            if data.shape == () or data.size == 1:
                data = data.item()
            else:
                data = data.tolist()
            
            result['data'] = data
    elif is_h5py_group(obj.h5py_obj):
        for name, child in obj.items():
            result[name] = to_dict(child, include_attrs, include_data)
    
    return result


def from_dict(data: Dict[str, Any], file: Union[h5py.File, str, PathLike], path: str = '/') -> H5Object:
    """
    Convert a dictionary to an HDF5 object.
    
    Parameters
    ----------
    data : Dict[str, Any]
        The dictionary to convert.
    file : Union[h5py.File, str, PathLike]
        The HDF5 file to store the object in.
    path : str, optional
        The path to store the object at, by default '/'.
        
    Returns
    -------
    H5Object
        The created object.
    """
    from .file import H5File
    
    # Open the file if it's a path
    if isinstance(file, (str, os.PathLike)):
        file = H5File(file, 'a')
    elif isinstance(file, h5py.File):
        file = H5File(file)
    
    # Get or create the group
    if path == '/':
        group = file
    else:
        group = file.require_group(path)
    
    # Add attributes
    if 'attrs' in data:
        for key, value in data['attrs'].items():
            group.attrs[key] = value
    
    # Add data or children
    for key, value in data.items():
        if key == 'attrs':
            continue
        elif key == 'data':
            # Create a dataset
            if isinstance(value, (list, np.ndarray)):
                group.create_dataset('data', data=np.array(value))
            else:
                group.create_dataset('data', data=value)
        else:
            # Create a group
            if isinstance(value, dict):
                from_dict(value, file, f"{path}/{key}".replace("//", "/"))
    
    return group


def to_json(obj: H5Object, include_attrs: bool = True, include_data: bool = True) -> str:
    """
    Convert an HDF5 object to a JSON string.
    
    Parameters
    ----------
    obj : H5Object
        The object to convert.
    include_attrs : bool, optional
        Whether to include attributes, by default True.
    include_data : bool, optional
        Whether to include dataset data, by default True.
        
    Returns
    -------
    str
        The object as a JSON string.
    """
    data = to_dict(obj, include_attrs, include_data)
    return json.dumps(data)


def from_json(json_str: str, file: Union[h5py.File, str, PathLike], path: str = '/') -> H5Object:
    """
    Convert a JSON string to an HDF5 object.
    
    Parameters
    ----------
    json_str : str
        The JSON string to convert.
    file : Union[h5py.File, str, PathLike]
        The HDF5 file to store the object in.
    path : str, optional
        The path to store the object at, by default '/'.
        
    Returns
    -------
    H5Object
        The created object.
    """
    data = json.loads(json_str)
    return from_dict(data, file, path)


def to_csv(dataset: 'H5Dataset', path: str, **kwargs) -> None:
    """
    Convert an HDF5 dataset to a CSV file.
    
    Parameters
    ----------
    dataset : H5Dataset
        The dataset to convert.
    path : str
        The path to save the CSV file to.
    **kwargs
        Additional keyword arguments to pass to pandas.DataFrame.to_csv.
    """
    df = dataset.to_dataframe()
    df.to_csv(path, **kwargs)


def from_csv(path: str, file: Union[h5py.File, str, PathLike], name: str, **kwargs) -> 'H5Dataset':
    """
    Convert a CSV file to an HDF5 dataset.
    
    Parameters
    ----------
    path : str
        The path to the CSV file.
    file : Union[h5py.File, str, PathLike]
        The HDF5 file to store the dataset in.
    name : str
        The name of the dataset.
    **kwargs
        Additional keyword arguments to pass to pandas.read_csv.
        
    Returns
    -------
    H5Dataset
        The created dataset.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for from_csv()")
    
    df = pd.read_csv(path, **kwargs)
    return from_dataframe(df, file, name)


def to_excel(dataset: 'H5Dataset', path: str, **kwargs) -> None:
    """
    Convert an HDF5 dataset to an Excel file.
    
    Parameters
    ----------
    dataset : H5Dataset
        The dataset to convert.
    path : str
        The path to save the Excel file to.
    **kwargs
        Additional keyword arguments to pass to pandas.DataFrame.to_excel.
    """
    df = dataset.to_dataframe()
    df.to_excel(path, **kwargs)


def from_excel(path: str, file: Union[h5py.File, str, PathLike], name: str, **kwargs) -> 'H5Dataset':
    """
    Convert an Excel file to an HDF5 dataset.
    
    Parameters
    ----------
    path : str
        The path to the Excel file.
    file : Union[h5py.File, str, PathLike]
        The HDF5 file to store the dataset in.
    name : str
        The name of the dataset.
    **kwargs
        Additional keyword arguments to pass to pandas.read_excel.
        
    Returns
    -------
    H5Dataset
        The created dataset.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for from_excel()")
    
    df = pd.read_excel(path, **kwargs)
    return from_dataframe(df, file, name)

