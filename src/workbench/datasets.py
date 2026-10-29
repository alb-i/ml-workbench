"""Dataset adapter interface (MNIST is one adapter).

The workbench gets images and original labels only through an adapter, so other image datasets
need a new adapter, not changes in the pages:

    list_samples(split) -> sample ids      get_image(sample_id) -> (H, W) uint8
    get_label(sample_id) -> int            splits() -> split names

ArrayAdapter holds arrays in RAM; MnistAdapter loads the IDX files of $DATA_DIR/mnist/raw (read
only, 55 MB) into one. Images are served as PNG via png_bytes.
"""

import gzip
import io
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image


class DatasetAdapter(Protocol):
    def splits(self) -> list[str]: ...
    def list_samples(self, split: str) -> list[str]: ...
    def get_image(self, sample_id: str) -> np.ndarray: ...
    def get_label(self, sample_id: str) -> int: ...


class ArrayAdapter:
    """{split: (ids, images (N, H, W) uint8, labels (N,))} in memory."""

    def __init__(self, data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]):
        self._data = data
        self._index = {
            sample_id: (split, i)
            for split, (ids, _, _) in data.items()
            for i, sample_id in enumerate(ids)
        }

    def splits(self) -> list[str]:
        return list(self._data)

    def list_samples(self, split: str) -> list[str]:
        return [str(i) for i in self._data[split][0]]

    def _locate(self, sample_id: str) -> tuple[str, int]:
        if sample_id not in self._index:
            raise KeyError(sample_id)
        return self._index[sample_id]

    def get_image(self, sample_id: str) -> np.ndarray:
        split, i = self._locate(sample_id)
        return self._data[split][1][i]

    def get_label(self, sample_id: str) -> int:
        split, i = self._locate(sample_id)
        return int(self._data[split][2][i])


MNIST_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", 5),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", 4),
}


def read_idx(path: Path) -> np.ndarray:
    """IDX file (gzip) with unsigned bytes: magic 0x0000 08 <ndim>, big-endian int32 dims, data."""
    raw = gzip.decompress(path.read_bytes())
    if raw[:3] != b"\x00\x00\x08":
        raise ValueError(f"{path}: not an unsigned-byte IDX file")
    ndim = raw[3]
    shape = tuple(int.from_bytes(raw[4 + 4 * k : 8 + 4 * k], "big") for k in range(ndim))
    data = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim)
    if data.size != int(np.prod(shape)):
        raise ValueError(f"{path}: expected {shape}, got {data.size} values")
    return data.reshape(shape)


class MnistAdapter(ArrayAdapter):
    """MNIST from <data_dir>/raw; ids train-00000 ... train-59999, test-0000 ... test-9999."""

    def __init__(self, data_dir: Path):
        data = {}
        for split, (images_file, labels_file, width) in MNIST_FILES.items():
            images = read_idx(Path(data_dir) / "raw" / images_file)
            labels = read_idx(Path(data_dir) / "raw" / labels_file)
            ids = np.array([f"{split}-{i:0{width}d}" for i in range(len(labels))])
            data[split] = (ids, images, labels)
        super().__init__(data)


def png_bytes(image: np.ndarray, scale: int = 1) -> bytes:
    """Greyscale PNG, enlarged by `scale` with nearest neighbour (sharp pixels)."""
    picture = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if scale > 1:
        size = (picture.width * scale, picture.height * scale)
        picture = picture.resize(size, Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return buffer.getvalue()
