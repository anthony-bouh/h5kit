# h5kit

`h5kit` is a small extension layer on top of [`h5py`](https://www.h5py.org/) for common HDF5 workflows: safer file wrappers, recursive traversal, querying, metadata helpers, conversion utilities, and simple structure printing.

The package is intentionally imported as `h5kit`, not `h5py`, so it can depend on the upstream `h5py` package without shadowing it.

## Main Differences With h5py

`h5py` is the low-level, standard Python interface to HDF5. `h5kit` keeps the same HDF5 model, but wraps `h5py` objects with convenience methods for everyday workflows.

Main additions:

- `open_file(...)` returns an enhanced file wrapper with context management and helper methods.
- Accessed files, groups, and datasets are automatically wrapped as `h5kit` objects.
- Datasets expose convenience methods such as `to_array()`, `to_list()`, `to_dataframe()`, and `append()`.
- Datasets created from `h5kit.File.create_dataset()` are chunked and resizable on axis 0 by default, unless you pass your own `chunks` or `maxshape`.
- Recursive traversal is available with `walk(...)` and `visit(...)`.
- Path-pattern search is available with `query(...)`, plus helpers such as `find(...)` and `find_by_regex(...)`.
- Metadata helpers make common attribute operations explicit: `get_metadata(...)`, `set_metadata(...)`, `has_metadata(...)`, and related functions.
- Simple inspection helpers such as `print_structure(...)` and `print_info(...)` make HDF5 files easier to explore.

In short: use `h5py` when you want direct HDF5 control; use `h5kit` when you want a more ergonomic layer for reading, writing, exploring, converting, and querying HDF5 files.

## Install

Install directly from GitHub:

```bash
python -m pip install "h5kit @ git+https://github.com/anthony-bouh/h5kit.git"
```

By default, `pip` installs the latest commit from the repository's default branch. For a stable, reproducible install, pin a version tag:

```bash
python -m pip install "h5kit @ git+https://github.com/anthony-bouh/h5kit.git@v0.1.2"
```

With optional DataFrame/CSV/Excel helpers:

```bash
python -m pip install "h5kit[data] @ git+https://github.com/anthony-bouh/h5kit.git@v0.1.2"
```

To install from another project's `requirements.txt`, add:

```txt
h5kit @ git+https://github.com/anthony-bouh/h5kit.git
```

## Quick Start

Create an HDF5 file, add groups, datasets, and metadata:

```python
import numpy as np
from h5kit import open_file

with open_file("sample.h5", "w") as h5:
    experiment = h5.create_group("experiment")
    experiment.attrs["operator"] = "Anthony"

    temperature = h5.create_dataset(
        "experiment/temperature",
        data=np.array([20.1, 20.4, 20.7, 21.0]),
    )
    pressure = h5.create_dataset(
        "experiment/pressure",
        data=np.array([1.01, 1.02, 1.02, 1.03]),
    )

    temperature.attrs["unit"] = "degC"
    pressure.attrs["unit"] = "bar"
```

Read the file back and access data as NumPy arrays or Python lists:

```python
from h5kit import open_file

with open_file("sample.h5", "r") as h5:
    temperature = h5["experiment/temperature"]

    values = temperature.to_array()
    values_as_list = temperature.to_list()
    unit = temperature.attrs["unit"]

    print(values)
    print(values_as_list)
    print(unit)
```

Print the file structure:

```python
from h5kit import open_file, print_structure

with open_file("sample.h5", "r") as h5:
    print_structure(h5)
```

Find objects by path pattern:

```python
from h5kit import open_file, query

with open_file("sample.h5", "r") as h5:
    matches = query(h5, "**/temperature")

    for item in matches:
        print(item.name, item.shape, item.attrs.get("unit"))
```

Walk through every object in the file:

```python
from h5kit import open_file, walk

with open_file("sample.h5", "r") as h5:
    for path, item in walk(h5):
        print(path, type(item).__name__)
```

Append data to a resizable dataset created by `h5kit`:

```python
import numpy as np
from h5kit import open_file

with open_file("sample.h5", "a") as h5:
    temperature = h5["experiment/temperature"]
    temperature.append(np.array([21.3, 21.5]))
```

Convert a dataset to a pandas DataFrame when `h5kit[data]` is installed:

```python
from h5kit import open_file

with open_file("sample.h5", "r") as h5:
    df = h5["experiment/temperature"].to_dataframe()
    print(df.head())
```

## License

This project is licensed under the MIT License. See `LICENSE` for details.
