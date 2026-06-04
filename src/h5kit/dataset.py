"""
Enhanced dataset operations for h5kit.

This module provides enhanced dataset operations, including data conversion,
chunk management, and other utilities for working with HDF5 datasets.
"""

import os
import h5py
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, Callable, TypeVar, Generic, Iterable
from .core import H5Object, PathLike


class H5Dataset(H5Object):
    """
    Enhanced HDF5 dataset class with additional functionality.
    
    This class wraps an h5py.Dataset object and provides additional functionality,
    including data conversion, chunk management, and other utilities.
    """
    
    def __init__(self, dataset_obj: h5py.Dataset):
        """
        Initialize an H5Dataset.
        
        Parameters
        ----------
        dataset_obj : h5py.Dataset
            The h5py.Dataset object to wrap.
        """
        super().__init__(dataset_obj)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the dataset."""
        return self._obj.shape
    
    @property
    def dtype(self) -> np.dtype:
        """Get the data type of the dataset."""
        return self._obj.dtype
    
    @property
    def size(self) -> int:
        """Get the size of the dataset."""
        return self._obj.size
    
    @property
    def ndim(self) -> int:
        """Get the number of dimensions of the dataset."""
        return self._obj.ndim
    
    @property
    def chunks(self) -> Optional[Tuple[int, ...]]:
        """Get the chunk shape of the dataset."""
        return self._obj.chunks
    
    @property
    def maxshape(self) -> Tuple[Optional[int], ...]:
        """Get the maximum shape of the dataset."""
        return self._obj.maxshape
    
    @property
    def fillvalue(self) -> Any:
        """Get the fill value of the dataset."""
        return self._obj.fillvalue
    
    @property
    def compression(self) -> Optional[str]:
        """Get the compression filter of the dataset."""
        return self._obj.compression
    
    @property
    def compression_opts(self) -> Any:
        """Get the compression options of the dataset."""
        return self._obj.compression_opts
    
    @property
    def shuffle(self) -> bool:
        """Get whether the shuffle filter is enabled."""
        return self._obj.shuffle
    
    @property
    def fletcher32(self) -> bool:
        """Get whether the Fletcher32 filter is enabled."""
        return self._obj.fletcher32
    
    @property
    def scaleoffset(self) -> Optional[Tuple[str, int]]:
        """Get the scale-offset filter settings."""
        return self._obj.scaleoffset
    
    def __getitem__(self, args) -> Union[np.ndarray, Any]:
        """
        Get data from the dataset.
        
        Parameters
        ----------
        args : Any
            The indices to get.
            
        Returns
        -------
        Union[np.ndarray, Any]
            The data.
        """
        return self._obj[args]
    
    def __setitem__(self, args, value) -> None:
        """
        Set data in the dataset.
        
        Parameters
        ----------
        args : Any
            The indices to set.
        value : Any
            The value to set.
        """
        self._obj[args] = value
    
    def read_direct(self, dest, source_sel=None, dest_sel=None) -> None:
        """
        Read data directly into an existing NumPy array.
        
        Parameters
        ----------
        dest : np.ndarray
            The destination array.
        source_sel : slice, optional
            The selection in the source dataset.
        dest_sel : slice, optional
            The selection in the destination array.
        """
        self._obj.read_direct(dest, source_sel, dest_sel)
    
    def write_direct(self, source, source_sel=None, dest_sel=None) -> None:
        """
        Write data directly from a NumPy array.
        
        Parameters
        ----------
        source : np.ndarray
            The source array.
        source_sel : slice, optional
            The selection in the source array.
        dest_sel : slice, optional
            The selection in the destination dataset.
        """
        self._obj.write_direct(source, source_sel, dest_sel)
    
    def resize(self, size: Union[int, Tuple[int, ...]]) -> None:
        """
        Resize the dataset.
        
        Parameters
        ----------
        size : Union[int, Tuple[int, ...]]
            The new size.
        """
        self._obj.resize(size)
    
    def to_array(self) -> np.ndarray:
        """
        Convert the dataset to a NumPy array.
        
        Returns
        -------
        np.ndarray
            The dataset as a NumPy array.
        """
        return self._obj[...]
    
    def to_list(self) -> List:
        """
        Convert the dataset to a Python list.
        
        Returns
        -------
        List
            The dataset as a Python list.
        """
        return self.to_array().tolist()
    
    def to_dataframe(self, index_col: Optional[str] = None) -> 'pd.DataFrame':
        """
        Convert the dataset to a pandas DataFrame.
        
        Parameters
        ----------
        index_col : Optional[str], optional
            The name of the column to use as the index.
            
        Returns
        -------
        pd.DataFrame
            The dataset as a pandas DataFrame.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")
        
        data = self.to_array()
        
        if self.ndim == 1 and self.dtype.kind == 'S':
            # Convert bytes to strings
            data = np.array([s.decode('utf-8') for s in data])
        
        if self.dtype.names:
            # Compound dataset
            df = pd.DataFrame(data)
            if index_col and index_col in self.dtype.names:
                df.set_index(index_col, inplace=True)
            return df
        else:
            # Regular dataset
            return pd.DataFrame(data)
    
    def iter_chunks(self, sel: Optional[Union[slice, Tuple[slice, ...]]] = None) -> Iterator[Tuple[Tuple[int, ...], np.ndarray]]:
        """
        Iterate over chunks in the dataset.
        
        Parameters
        ----------
        sel : Optional[Union[slice, Tuple[slice, ...]]], optional
            The selection to iterate over.
            
        Returns
        -------
        Iterator[Tuple[Tuple[int, ...], np.ndarray]]
            An iterator yielding (chunk_index, chunk_data) pairs.
        """
        for chunk_index in self._obj.iter_chunks(sel):
            chunk_data = self._obj[chunk_index]
            yield chunk_index, chunk_data
    
    def iter_allocated_chunks(self) -> Iterator[Tuple[Tuple[int, ...], np.ndarray]]:
        """
        Iterate over allocated chunks in the dataset.
        
        This is more efficient than iter_chunks() for sparse datasets.
        
        Returns
        -------
        Iterator[Tuple[Tuple[int, ...], np.ndarray]]
            An iterator yielding (chunk_index, chunk_data) pairs.
        """
        if not self.chunks:
            # Not chunked, just return the whole dataset
            yield (0,) * self.ndim, self.to_array()
            return
        
        # Get the number of chunks in each dimension
        chunk_dims = tuple((s + c - 1) // c for s, c in zip(self.shape, self.chunks))
        
        # Get the allocated chunks using the low-level API
        dcpl = self._obj.id.get_create_plist()
        
        if hasattr(dcpl, 'get_num_chunks'):
            # Use the HDF5 1.10.5+ API if available
            space = self._obj.id.get_space()
            nchunks = dcpl.get_num_chunks(space)
            
            for i in range(nchunks):
                chunk_info = dcpl.get_chunk_info(space, i)
                chunk_offset = chunk_info.chunk_offset
                chunk_size = self.chunks
                
                # Convert byte offset to chunk index
                chunk_index = tuple(o // s for o, s in zip(chunk_offset, chunk_size))
                
                # Calculate the selection for this chunk
                sel = tuple(slice(o, min(o + s, d)) for o, s, d in zip(
                    chunk_offset, chunk_size, self.shape))
                
                # Read the chunk data
                chunk_data = self._obj[sel]
                
                yield chunk_index, chunk_data
        else:
            # Fall back to iterating over all chunks and checking if they're allocated
            # This is less efficient but works with older HDF5 versions
            for chunk_index, chunk_data in self.iter_chunks():
                # Check if the chunk is allocated by looking for non-fill values
                if not np.all(chunk_data == self.fillvalue):
                    yield chunk_index, chunk_data
    
    def get_chunk_info(self) -> Dict[str, Any]:
        """
        Get information about the chunking of the dataset.
        
        Returns
        -------
        Dict[str, Any]
            Information about the chunking.
        """
        if not self.chunks:
            return {
                'is_chunked': False,
                'chunks': None,
                'chunk_dims': None,
                'total_chunks': 0,
                'allocated_chunks': 0,
                'allocation_ratio': 0.0,
            }
        
        # Get the number of chunks in each dimension
        chunk_dims = tuple((s + c - 1) // c for s, c in zip(self.shape, self.chunks))
        
        # Calculate the total number of chunks
        total_chunks = np.prod(chunk_dims)
        
        # Count the allocated chunks
        allocated_chunks = 0
        
        # Get the allocated chunks using the low-level API
        dcpl = self._obj.id.get_create_plist()
        
        if hasattr(dcpl, 'get_num_chunks'):
            # Use the HDF5 1.10.5+ API if available
            space = self._obj.id.get_space()
            allocated_chunks = dcpl.get_num_chunks(space)
        else:
            # Fall back to iterating over all chunks and checking if they're allocated
            # This is less efficient but works with older HDF5 versions
            for chunk_index, chunk_data in self.iter_chunks():
                # Check if the chunk is allocated by looking for non-fill values
                if not np.all(chunk_data == self.fillvalue):
                    allocated_chunks += 1
        
        # Calculate the allocation ratio
        allocation_ratio = allocated_chunks / total_chunks if total_chunks > 0 else 0.0
        
        return {
            'is_chunked': True,
            'chunks': self.chunks,
            'chunk_dims': chunk_dims,
            'total_chunks': total_chunks,
            'allocated_chunks': allocated_chunks,
            'allocation_ratio': allocation_ratio,
        }
    
    def append(self, data: np.ndarray, axis: int = 0) -> None:
        """
        Append data to the dataset along the specified axis.
        
        Parameters
        ----------
        data : np.ndarray
            The data to append.
        axis : int, optional
            The axis along which to append, by default 0.
            
        Raises
        ------
        ValueError
            If the dataset is not resizable along the specified axis.
        """
        if self.maxshape[axis] is None or self.maxshape[axis] > self.shape[axis]:
            # Calculate the new shape
            new_shape = list(self.shape)
            new_shape[axis] += data.shape[axis] if data.ndim > axis else 1
            
            # Resize the dataset
            self.resize(tuple(new_shape))
            
            # Calculate the slice for the new data
            slices = [slice(None)] * self.ndim
            slices[axis] = slice(self.shape[axis] - data.shape[axis] if data.ndim > axis else 1, None)
            
            # Write the data
            self[tuple(slices)] = data
        else:
            raise ValueError(f"Dataset is not resizable along axis {axis}")
    
    def print_info(self) -> None:
        """Print information about the dataset."""
        from .visualization import print_info
        
        print_info(self)
    
# Alias for backward compatibility
Dataset = H5Dataset

