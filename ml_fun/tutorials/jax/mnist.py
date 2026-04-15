import gzip
from pathlib import Path
from urllib import request

import jax
import jax.numpy as jnp
import numpy as np
import optax


def load_mnist() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    def download(filename):
        data_dir = Path(__file__).parent / "data"
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


def ffn_init(input_dim: int, hidden_dim: int, output_dim: int) -> optax.Params:
    rk1, rk2 = jax.random.split(jax.random.PRNGKey(0))
    w1 = jax.random.normal(rk1, (input_dim, hidden_dim)) * 0.01
    b1 = jnp.zeros((hidden_dim,))
    w2 = jax.random.normal(rk2, (hidden_dim, output_dim)) * 0.01
    b2 = jnp.zeros((output_dim,))
    return {
        "W1": w1,
        "b1": b1,
        "W2": w2,
        "b2": b2,
    }


def ffn_forward(params: dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    z1 = jnp.dot(x, params["W1"]) + params["b1"]
    a1 = jax.nn.relu(z1)
    z2 = jnp.dot(a1, params["W2"]) + params["b2"]
    return z2


def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    print("Shapes logits / labels:", logits.shape, labels.shape)
    one_hot_labels = jax.nn.one_hot(labels, num_classes=logits.shape[1])
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(one_hot_labels * log_probs, axis=1))


@jax.jit
def train_step(
    params: dict[str, jnp.ndarray],
    opt_state: optax.OptState,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> tuple[optax.Params, optax.OptState, jnp.ndarray]:
    def loss_fn(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        logits = ffn_forward(params, x)
        return cross_entropy_loss(logits, y)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss


if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = load_mnist()

    # Flatten images to (N, 784)
    train_images = train_images.reshape(-1, 28 * 28)
    test_images = test_images.reshape(-1, 28 * 28)

    optimizer = optax.adam(learning_rate=1e-3)
    params = ffn_init(input_dim=28 * 28, hidden_dim=128, output_dim=10)
    opt_state = optimizer.init(params)

    epochs = 10
    batch_size = 128
    num_samples = train_images.shape[0]

    for epoch in range(epochs):
        key = jax.random.PRNGKey(epoch)
        perm = jax.random.permutation(key, num_samples)
        train_images = train_images[perm]
        train_labels = train_labels[perm]

        epoch_loss = 0.0
        num_batches = 0
        for i in range(0, num_samples, batch_size):
            x_batch = train_images[i : i + batch_size]
            y_batch = train_labels[i : i + batch_size]
            params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch)
            epoch_loss += loss
            num_batches += 1

        # Evaluate
        test_logits = ffn_forward(params, test_images)
        test_preds = jnp.argmax(test_logits, axis=1)
        accuracy = jnp.mean(test_preds == test_labels)
        print(
            f"Epoch {epoch + 1}/{epochs} — "
            f"loss: {epoch_loss / num_batches:.4f}, "
            f"test accuracy: {accuracy:.4f}"
        )
