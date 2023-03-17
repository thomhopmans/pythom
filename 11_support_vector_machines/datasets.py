import pandas as pd
from sklearn import datasets


def linear_separable_df():
    # Linearly separable in 2D
    iris = datasets.load_iris()

    return (
        pd.DataFrame(iris.data[:, :2], columns=["length", "width"])
        .assign(target=iris.target.tolist())
        .assign(
            target=lambda d: d["target"].apply(lambda x: 1 if x == 0 else 0),
            name=lambda d: d["target"].apply(get_name),
        )
        .loc[lambda d: d["target"].isin([0, 1])]
    )


def non_linear_separable_df():
    # Non-linearly separable in 2D
    iris = datasets.load_iris()

    return (
        pd.DataFrame(iris.data[:, :2], columns=["length", "width"])
        .assign(target=iris.target.tolist())
        .loc[lambda d: d["target"].isin([1, 2])]
        .assign(
            target=lambda d: d["target"].apply(lambda x: 0 if x == 2 else x),
            name=lambda d: d["target"].apply(get_name),
        )
    )


def linear_separable_in_3d_df():
    # Non-linearly separable in 2D, but linear separable in 3D
    X, y = datasets.make_circles(n_samples=100, random_state=42)

    return (
        pd.DataFrame(X, columns=["length", "width"])
        .assign(target=y)
        .assign(
            target=lambda d: d["target"].apply(lambda x: 1 if x == 0 else 0),
            name=lambda d: d["target"].apply(get_name),
        )
        .sample(40, random_state=43)
    )


def get_name(target):
    if target == 0:
        return "rejected"
    elif target == 1:
        return "approved"
    else:
        return "other"
