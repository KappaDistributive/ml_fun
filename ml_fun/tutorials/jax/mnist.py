import gzip
from pathlib import Path
from urllib import request

import numpy as np
import jax
import jax.numpy as jnp


def load_mnist():
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    def download(filename):
        data_dir = Path("/tmp/ml_fun/data")
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / filename
        if not path.exists():
            print(f"Downloading {filename}...")
            request.urlretrieve(base_url + filename, path)
        return path

    # Load images
    with gzip.open(download(files[0]), "rb") as f:
        train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(download(files[2]), "rb") as f:
        test_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    # Load labels
    with gzip.open(download(files[1]), "rb") as f:
        train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open(download(files[3]), "rb") as f:
        test_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

    return (
        jnp.array(train_images, dtype=jnp.float32) / 255.0,
        jnp.array(train_labels, dtype=jnp.int32),
        jnp.array(test_images, dtype=jnp.float32) / 255.0,
        jnp.array(test_labels, dtype=jnp.int32),
    )


if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = load_mnist()
    print("Train images shape:", train_images.shape)
    print("Train labels shape:", train_labels.shape)
    print("Test images shape:", test_images.shape)
    print("Test labels shape:", test_labels.shape)
