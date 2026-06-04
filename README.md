# h5kit

`h5kit` is a small extension layer on top of [`h5py`](https://www.h5py.org/) for common HDF5 workflows: safer file wrappers, recursive traversal, querying, metadata helpers, conversion utilities, and simple structure printing.

The package is intentionally imported as `h5kit`, not `h5py`, so it can depend on the upstream `h5py` package without shadowing it.

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

## Quick Start

```python
import numpy as np
from h5kit import open_file, print_structure

with open_file("sample.h5", "w") as h5:
    dataset = h5.create_dataset("measurements", data=np.arange(5))
    dataset.attrs["unit"] = "mm"

with open_file("sample.h5", "r") as h5:
    print_structure(h5)
    values = h5["measurements"].to_array()
```

## Development

Run the test suite:

```bash
pytest
```

Build a source distribution and wheel:

```bash
python -m build
```

## License

This project is licensed under the MIT License. See `LICENSE` for details.
