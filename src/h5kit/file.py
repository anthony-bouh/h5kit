"""
Enhanced file operations for h5kit.

This module provides enhanced file handling functionality, including context management,
automatic file closing, and simplified file creation with sensible defaults.
"""

import h5py
import numpy as np

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Iterator,
    Callable,
    TypeVar,
    Generic,
    overload
)

from .core import (
    H5Object,
    register_file,
    unregister_file,
    PathLike
)

from .dataset import H5Dataset
from .group import H5Group


class H5File(H5Object):
    """
    Enhanced HDF5 file class with additional functionality.
    
    This class wraps an h5py.File object and provides additional functionality,
    including context management, automatic file closing, and simplified file creation.
    """
    
    def __init__(self, file_obj: Union[h5py.File, str, PathLike], mode: str = 'r', **kwargs):
        """
        Initialize an H5File.
        
        Parameters
        ----------
        file_obj : Union[h5py.File, str, PathLike]
            Either an existing h5py.File object or a path to an HDF5 file.
        mode : str, optional
            The file mode if opening a new file, by default 'r'.
            Ignored if file_obj is an h5py.File.
        **kwargs
            Additional keyword arguments to pass to h5py.File.
        """
        if isinstance(file_obj, h5py.File):
            super().__init__(file_obj)
        else:
            super().__init__(h5py.File(file_obj, mode, **kwargs))
        
        # Register this file in the global registry
        register_file(self)
    
    @property
    def filename(self) -> str:
        """Get the filename of the HDF5 file."""
        return self._obj.filename
    
    @property
    def mode(self) -> str:
        """Get the file mode."""
        return self._obj.mode
    
    @property
    def driver(self) -> str:
        """Get the file driver."""
        return self._obj.driver
    
    @property
    def libver(self) -> Tuple[str, str]:
        """Get the HDF5 library version bounds."""
        return self._obj.libver
    
    @property
    def userblock_size(self) -> int:
        """Get the user block size."""
        return self._obj.userblock_size
    
    @property
    def is_open(self) -> bool:
        """Check if the file is open."""
        return bool(self._obj)
    
    def close(self) -> None:
        """Close the file."""
        if self.is_open:
            try:
                self._obj.close()
            finally:
                unregister_file(self)
    
    def flush(self) -> None:
        """Flush the file to disk."""
        self._obj.flush()
    
    def __enter__(self) -> 'H5File':
        """Enter the context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager and close the file."""
        self.close()
    
    def __getitem__(self, name: str) -> H5Object:
        """
        Get an item from the file by name.
        
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
        
        # Make name non-case sensitive by default
        name = name.lower()
        obj = self._obj[name]
        wrapped = wrap_h5py_object(obj)
        if wrapped is None:
            raise ValueError(f"Unknown object type: {type(obj)}")
        return wrapped
    
    def __contains__(self, name: str) -> bool:
        """
        Check if an item exists in the file.
        
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
        Get the keys (names) of items in the file.
        
        Returns
        -------
        List[str]
            The keys of items in the file.
        """
        return list(self._obj.keys())
    
    def values(self) -> List[H5Object]:
        """
        Get the values (items) in the file.
        
        Returns
        -------
        List[H5Object]
            The values of items in the file.
        """
        from .core import wrap_h5py_object
        
        return [wrap_h5py_object(obj) for obj in self._obj.values()]
    
    def items(self) -> List[Tuple[str, H5Object]]:
        """
        Get the items (key-value pairs) in the file.
        
        Returns
        -------
        List[Tuple[str, H5Object]]
            The items in the file.
        """
        from .core import wrap_h5py_object
        
        return [(key, wrap_h5py_object(obj)) for key, obj in self._obj.items()]
    
    def get(self, name: str, default: Any = None) -> Optional[H5Object]:
        """
        Get an item from the file by name with a default value.
        
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
    
    def create_group(self, name: str) -> H5Group:
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
        from .group import H5Group
        
        return H5Group(self._obj.create_group(name))
    
    def create_dataset(self, name: str, shape=None, dtype=None, data=None, **kwargs) -> H5Dataset:
        """
        Create a new dataset.

        Defaults:
        - Makes datasets chunked and resizable (axis 0) by default.
        - If you pass your own `chunks` or `maxshape`, those are respected.
        - Scalars (0D) are left contiguous (no chunking).

        Parameters
        ----------
        name : str
            Dataset path.
        shape : tuple[int, ...] | int | None
            Shape of the dataset (ignored if `data` provided and array-like).
        dtype : np.dtype | None
            Dataset dtype.
        data : Any | None
            Initial data.
        **kwargs
            Passed through to h5py.File.create_dataset.

        Returns
        -------
        H5Dataset
        """
        # If caller already set chunks or maxshape, respect their choice.
        user_specified_chunks = "chunks" in kwargs
        user_specified_maxshape = "maxshape" in kwargs

        # Determine effective shape if possible
        eff_shape = None
        if data is not None and not np.isscalar(data):
            try:
                eff_shape = np.asarray(data).shape
            except Exception:
                pass  # fall back to provided `shape`
        if eff_shape is None and shape is not None:
            eff_shape = (shape,) if isinstance(shape, int) else tuple(shape)

        # For array datasets (ndim >= 1), default to chunked + resizable on axis 0
        if eff_shape is not None and len(eff_shape) >= 1:
            if not user_specified_chunks:
                # Let h5py pick reasonable chunk sizes
                kwargs.setdefault("chunks", True)
            if not user_specified_maxshape:
                # Make axis 0 unlimited by default
                kwargs.setdefault("maxshape", (None,) + tuple(eff_shape[1:]))

        # Note: for scalars (eff_shape is None or ()), we don't force chunking

        ds = self._obj.create_dataset(name, shape=shape, dtype=dtype, data=data, **kwargs)
        return H5Dataset(ds)
    
    def require_group(self, name: str) -> H5Group:
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
        from .group import H5Group
        
        return H5Group(self._obj.require_group(name))
    
    def create_groups_recursive(self, path: str) -> H5Group:
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
        from .group import H5Group
        
        if not path:
            return self
        
        parts = path.strip('/').split('/')
        current = self
        
        for part in parts:
            if not part:
                continue
            if part in current:
                current = current[part]
            else:
                current = current.create_group(part)
        
        return current
    
    def walk(self) -> Iterator[Tuple[str, H5Object]]:
        """
        Walk through all items in the file recursively.
        
        This is a more Pythonic alternative to h5py's visit and visititems methods,
        which use callbacks.
        
        Returns
        -------
        Iterator[Tuple[str, H5Object]]
            An iterator yielding (path, object) pairs.
        """
        from .iterate import walk
        
        yield from walk(self)
    
    def copy(self, source: str, dest: str, overwrite: bool = False) -> None:
        """
        Copy an object within the file.
        
        Parameters
        ----------
        source : str
            The source path.
        dest : str
            The destination path.
        """
        if overwrite and dest in self:
            del self[dest]
        self._obj.copy(source, dest)
    
    def move(self, source: str, dest: str) -> None:
        """
        Move an object within the file.
        
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
        Query objects in the file using a path pattern.
        
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
    
    def find_by_regex(self, regex: str, include_attrs: bool = False) -> List[H5Object]:
        """
        Find objects in the file by a regular expression.
        
        Parameters
        ----------
        regex : str
            The regular expression to match.
            
        include_attrs : bool, optional
            Whether to include attributes in the search, by default False.
            
        Returns
        -------
        List[H5Object]
            The matching objects.
        """
        from .query import find_by_regex
        
        return find_by_regex(self, regex, include_attrs=include_attrs)
    
    def find_attributes(self, attr_pattern: str) -> List[Any]:
        """
        Find attributes in the file by name pattern.
        
        Parameters
        ----------
        attr_pattern : str
            The attribute name pattern to match (supports wildcards).
            
        Returns
        -------
        List[Any]
            The matching attribute values.
        """
        from .query import find_attributes
        
        return find_attributes(self, attr_pattern)
    
    def find_attr_by_regex(self, regex: str) -> List[Any]:
        """
        Find attribute's values in the file by a regular expression.
        
        Parameters
        ----------
        regex : str
            The regular expression to match.
            
        Returns
        -------
        List[Any]
            The matching attribute values.
        """
        from .query import find_attr_by_regex
        
        return find_attr_by_regex(self, regex)
    
    def print_structure(self, max_depth: int = None) -> None:
        """
        Print the structure of the file.
        
        Parameters
        ----------
        max_depth : int, optional
            The maximum depth to print.
        """
        from .visualization import print_structure
        
        print_structure(self, max_depth)
    
    def print_info(self) -> None:
        """Print information about the file."""
        from .visualization import print_info
        
        print_info(self)
        
    def save_as(
        self,
        dest_path: str,
        overwrite: bool = False,
        *,
        preserve_attrs: bool = True,
        expand_soft: bool = False,
        expand_external: bool = True,
        expand_refs: bool = True,
        libver: tuple[str, str] | None = None,
        userblock_size: int | None = None,
    ) -> None:
        """
        Save a full copy of this HDF5 file to `dest_path`.

        Parameters
        ----------
        dest_path : str
            Destination file path where the copy will be saved.
        overwrite : bool, optional
            Overwrite existing file if True. Default False.
        preserve_attrs : bool, optional
            Copy root (file-level) attributes. Default True.
        expand_soft : bool, optional
            Resolve soft links as real copies. Default False (keep soft links).
        expand_external : bool, optional
            Resolve external links by copying target objects into the new file.
            Default True. Set to False to keep external links.
        expand_refs : bool, optional
            Expand object/region references. Default True.
        libver : tuple[str, str] | None, optional
            libver bounds for the new file. Defaults to source file's `libver`.
        userblock_size : int | None, optional
            User block size for the new file. Defaults to source's `userblock_size`.

        Raises
        ------
        FileExistsError
            If `dest_path` exists and `overwrite=False`.
        """
        import os
        
        if os.path.abspath(dest_path) == os.path.abspath(self.filename):
            raise ValueError("Destination path is the same as the source file.")

        if os.path.exists(dest_path):
            if not overwrite:
                raise FileExistsError(f"Destination file '{dest_path}' already exists.")
            # Ensure we can recreate it from scratch
            os.remove(dest_path)

        # Use the source file's settings unless overridden
        libver = libver if libver is not None else getattr(self._obj, "libver", None)
        kwargs = {}
        if libver is not None:
            kwargs["libver"] = libver
        if userblock_size is None:
            try:
                userblock_size = self._obj.userblock_size
            except Exception:
                userblock_size = None
        if userblock_size is not None:
            kwargs["userblock_size"] = int(userblock_size)

        # Create destination file and copy all top-level members
        with h5py.File(dest_path, "w", **kwargs) as new_file:
            # Copy root attributes if requested
            if preserve_attrs:
                for k, v in self._obj.attrs.items():
                    new_file.attrs[k] = v

            # Copy each top-level object under "/"
            for name in self._obj:
                # self._obj.copy accepts (source, dest, name=...) where:
                # - source can be a path string (top-level member name here)
                # - dest is the destination group/file object
                # - name is the new name in dest
                self._obj.copy(
                    source=name,
                    dest=new_file,
                    name=name,
                    shallow=False,
                    expand_soft=expand_soft,
                    expand_external=expand_external,
                    expand_refs=expand_refs,
                )

            # Make sure all data hits disk
            new_file.flush()


@overload
def open_file(file: h5py.File, mode: str = 'r', **kwargs) -> H5File:
    ...


@overload
def open_file(file: Union[str, PathLike], mode: str = 'r', **kwargs) -> H5File:
    ...


def open_file(file, mode: str = 'r', **kwargs) -> H5File:
    """
    Open an HDF5 file.
    
    This is a convenience function that creates an H5File object.
    
    Parameters
    ----------
    file : Union[h5py.File, str, PathLike]
        Either an existing h5py.File object or a path to an HDF5 file.
    mode : str, optional
        The file mode if opening a new file, by default 'r'.
        Ignored if file is an h5py.File.
    **kwargs
        Additional keyword arguments to pass to h5py.File.
        
    Returns
    -------
    H5File
        The opened file.
    """
    return H5File(file, mode, **kwargs)


# Alias for backward compatibility
File = H5File

