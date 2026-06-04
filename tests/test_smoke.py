import importlib

import numpy as np

import h5kit


def test_imports_upstream_h5py_dependency():
    h5py = importlib.import_module("h5py")

    assert hasattr(h5py, "File")
    assert "src/h5py" not in (h5py.__file__ or "").replace("\\", "/")


def test_public_api_exports_version():
    assert h5kit.__version__ == "0.1.2"
    assert h5kit.File is h5kit.file.H5File
    assert h5kit.Dataset is h5kit.dataset.H5Dataset


def test_create_read_and_validate_hdf5_file(tmp_path):
    path = tmp_path / "sample.h5"

    with h5kit.open_file(path, "w") as h5:
        dataset = h5.create_dataset("measurements", data=np.arange(4))
        dataset.attrs["unit"] = "mm"

        assert dataset.to_list() == [0, 1, 2, 3]
        assert dataset.attrs["unit"] == "mm"

    assert h5kit.is_hdf5_file(path)
