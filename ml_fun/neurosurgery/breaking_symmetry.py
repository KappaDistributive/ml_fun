import time
from typing import Optional, Tuple

import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()


def xavier_W(num_fan_in: int, num_fan_out: int) -> tf.Tensor:
    initializer = np.random.normal(
        size=(num_fan_in, num_fan_out),
        scale=np.sqrt(2.0 / float(num_fan_in + num_fan_out)),
    ).astype("float32")
    return tf.compat.v1.get_variable(
        initializer=initializer, name="W", dtype=tf.float32
    )


def xavier_b(num_fan_out: int) -> tf.Tensor:
    return tf.compat.v1.get_variable(
        initializer=np.zeros(shape=(num_fan_out,), dtype="float32"),
        name="b",
        dtype=tf.float32,
    )


def linear_layer(
    x: tf.Tensor, num_fan_out: int, layer_name: str, activation: Optional[str] = None
) -> tf.Tensor:
    with tf.compat.v1.variable_scope(layer_name):
        # W = tf.Variable(np.ones(shape=(x.get_shape()[1], num_fan_out)) / float(x.get_shape()[1]), dtype=tf.float32)
        # b = tf.Variable(0., dtype=tf.float32)
        W = xavier_W(x.get_shape()[1], num_fan_out)
        b = xavier_b(num_fan_out)
        x = tf.einsum("bi,ij->bj", x, W) + b
        if activation is None:
            return x
        elif activation.lower() == "relu":
            x = tf.nn.relu(x)
            return x
        else:
            raise ValueError(f"Unknown activation: {activation}")


def model(x: tf.Tensor) -> tf.Tensor:
    x = linear_layer(x, 28 * 28, "l1", "relu")
    x = linear_layer(x, 28 * 28, "l2", "relu")
    x = linear_layer(x, 28 * 28, "l3", "relu")
    x = linear_layer(x, 10, "logits")

    return x


def get_batch(
    xs: np.ndarray, ys: np.ndarray, batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    x_batch, y_batch = [], []
    assert xs.shape[0] == ys.shape[0]
    for index in range(xs.shape[0]):
        x_batch.append(xs[index])
        y_batch.append(ys[index])
        if len(x_batch) >= batch_size:
            yield np.stack(x_batch, axis=0), np.stack(y_batch, axis=0)
            x_batch, y_batch = [], []
    if x_batch:
        yield np.stack(x_batch, axis=0), np.stack(y_batch, axis=0)


def one_hot(batch: np.array, depth: int) -> np.array:
    assert len(batch.shape) == 1
    result = np.zeros(shape=(batch.shape[0], depth))
    for index, x in enumerate(batch):
        result[index, int(x)] = 1

    return result


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    devices = tf.config.list_physical_devices("GPU")
    cuda_is_available = len(devices) > 0
    print(f"Devices: {devices}")
    device = tf.test.gpu_device_name() if cuda_is_available else "/CPU:0"
    logging_interval = 100

    with tf.device(device):
        x_ph = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=(None, 28 * 28), name="x"
        )
        y_true_ph = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=(None, 10), name="y_true"
        )
        logits_op = model(x_ph)
        probs_op = tf.sigmoid(logits_op)
        loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_true_ph, logits=logits_op)
        )
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
        train_op = optimizer.minimize(loss_op)

        initializer = tf.compat.v1.global_variables_initializer()
        total_loss = 0.0
        num_steps = 0
        time_start = time.perf_counter()

        with tf.compat.v1.Session() as session:
            session.run(initializer)

            # train loop
            for epoch_index in range(100):
                print(f"Epoch: {epoch_index + 1}")
                for batch_index, batch in enumerate(get_batch(x_train, y_train, 32)):
                    num_steps += 1
                    _, loss = session.run(
                        [train_op, loss_op],
                        feed_dict={
                            x_ph: np.reshape(batch[0], (-1, 28 * 28)),
                            y_true_ph: one_hot(batch[1], depth=10),
                        },
                    )
                    total_loss += loss

                    # evaluation
                    if (batch_index + 1) % logging_interval == 0:
                        print(f"Step: {batch_index + 1}")
                        print(
                            f"Time elapsed during the last {logging_interval} steps: {time.perf_counter() - time_start:.4f}"
                        )
                        print(f"Batch Loss: {loss:.4f}")
                        print(f"Average Loss: {total_loss / num_steps:.4f}")
                        # eval_batch = next(eval_generator)
                        # y_pred = np.squeeze(
                        #   np.round(session.run(probs_op, feed_dict={x_ph: np.reshape(eval_batch[0], (-1, 100))})).astype("int64")
                        # )
                        # y_true = eval_batch[1]
                        # print(f"y_pred: {y_pred[:10]} y_true: {y_true[:10]}")
                        # print(f"Accuracy: {np.sum(y_true == y_pred) / len(y_pred)}")
                        time_start = time.perf_counter()
