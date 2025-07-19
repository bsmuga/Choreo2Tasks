import numpy as np
import tensorflow as tf


def compute_fn(T: tf.Variable, p: tf.Variable) -> tf.Variable:
    return tf.stack(
        [
            T[0],
            T[1] + T[0] * p,
            T[2] + (T[1] + T[0] * p) * p,
            T[3] + (T[2] + T[1] + T[0] * p),
        ]
    )


def compute_fn_no_redundant_ops(T: tf.Variable, p: tf.Variable) -> tf.Variable:
    """Avoid redundant computations."""

    t0_p = T[0] * p
    t1_plus_t0p = T[1] + t0_p
    return tf.stack(
        [
            T[0],
            t1_plus_t0p,
            T[2] + t1_plus_t0p * p,
            T[3] + T[2] + t1_plus_t0p,
        ]
    )


def compute_fn_with_matmul(T: tf.Variable, p: tf.Variable) -> tf.Variable:
    p_2 = p**2
    matrix = tf.constant(
        [
            [1, 0, 0, 0],
            [1, p.numpy(), 0, 0],
            [1, p.numpy(), 1 + p_2.numpy(), 0],
            [0, p.numpy(), 1 + p_2.numpy(), 1],
        ],
        dtype=tf.float32,
    )
    return tf.linalg.matvec(matrix, T)
