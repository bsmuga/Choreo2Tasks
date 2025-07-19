import random
import timeit
from typing import Callable

import tensorflow as tf

from task2_tensorflow_optimization.src.funcs import (
    compute_fn,
    compute_fn_no_redundant_ops,
    compute_fn_with_matmul,
)


def benchmark_fn_perf_time(
    func: Callable,
    *args,
    warm_up: int = 5,
    n_runs: int = 100,
    repeat: int = 5,
) -> float:
    """Benchmark a function's execution time.

    Note: In Python docs it is recommended to be interested in min value
    of computed results, because `the lowest value gives a lower bound
    for how fast your machine can run the given code snippet
    and  higher values in the result vector are typically not caused
    by variability in Python’s speed, but by other processes interfering
    with your timing accuracy.`
    """

    for _ in range(warm_up):
        _ = func(*args)

    times = timeit.repeat(lambda: func(*args), repeat=repeat, number=n_runs)

    return min(times)


def calculate_gradients(
    func: Callable, T: tf.Variable, p: tf.Variable
) -> tuple[tf.Tensor | None, tf.Tensor | None]:
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([T, p])
        output = func(T, p)

    grad_T = tape.gradient(output, T)
    grad_p = tape.gradient(output, p)
    return grad_T, grad_p


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")

    T = tf.Variable(
        [
            random.uniform(-1e5, 1e5)
            for _ in range(
                4,
            )
        ]
    )
    p = tf.Variable(random.random())

    for fn in [compute_fn, compute_fn_no_redundant_ops]:
        basic_fn_perf = benchmark_fn_perf_time(fn, T, p)
        compiled_fn_perf = benchmark_fn_perf_time(tf.function()(fn), T, p)
        compiled_jit_fn_perf = benchmark_fn_perf_time(
            tf.function(jit_compile=True)(fn), T, p
        )
        print(
            "=" * 100
            + f"\nPerformance for: {fn.__name__}\n"
            + f"  basic: {round(basic_fn_perf, 4)}\n"
            + f"  compiled: {round(compiled_fn_perf, 4)}\n"
            f"  compiled with jit: {round(compiled_jit_fn_perf, 4)}\n" + "=" * 100
        )

        grad_T, grad_p = calculate_gradients(fn, T, p)
        assert (grad_T is not None) and (
            grad_p is not None
        ), "Unable to calculate gradients"
