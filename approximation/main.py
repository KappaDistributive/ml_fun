from typing import Callable

import matplotlib.pyplot as plt
import seaborn as sns


def euler_method(
    df: Callable[[float], float], xs: list[float], y_0: float
) -> list[tuple[float, float]]:
    f_est = [(xs[0], y_0)]
    for x in xs[1:]:
        x_prev, y_prev = f_est[-1]
        f_est.append((x, y_prev + (x - x_prev) * df(x_prev)))
    return f_est


if __name__ == "__main__":
    f = lambda x: x**2
    df = lambda x: 2 * x

    xs = [(i + 1) * 0.5 for i in range(10)]
    estimate = euler_method(df, xs, f(xs[0]))

    print(estimate)
    sns.scatterplot(x=(e[0] for e in estimate), y=(e[1] for e in estimate), color="red")
    sns.scatterplot(
        x=(e[0] for e in estimate), y=(f(e[0]) for e in estimate), color="green"
    )
    plt.show()
