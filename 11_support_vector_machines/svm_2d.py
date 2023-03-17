import numpy as np
import pandas as pd
from sklearn import svm


def get_contour(clf, clip=True, interval=0.1):
    if len(clf.coef_[0]) > 2:
        raise ValueError("Not supported for more than 2 features")

    x = np.arange(3, 9, interval)
    y = np.arange(2, 6, interval)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = (xx * clf.coef_[0][0] + yy * clf.coef_[0][1]) + clf.intercept_[0]

    if clip:
        z = np.round(np.clip(z, -0.01, 0.01) * 100, 0)

    return x, y, z


def get_separating_hyperplane(clf, x_start=4, x_end=8):
    if len(clf.coef_[0]) > 2:
        raise ValueError("Not supported for more than 2 features")

    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(x_start, x_end)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    return xx, yy, a


def get_parallels(clf, x_start=4, x_end=8):
    # plot the parallels to the separating hyperplane that pass through the
    # support vectors (margin away from hyperplane in direction
    # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in 2-d.

    _, yy, a = get_separating_hyperplane(clf, x_start=x_start, x_end=x_end)

    margin = 1 / np.sqrt(np.sum(clf.coef_**2))
    yy_down = yy - np.sqrt(1 + a**2) * margin
    yy_up = yy + np.sqrt(1 + a**2) * margin

    return yy_down, yy_up, margin


def fit_classifier(df, features, **kwargs):
    # Fit model
    clf = svm.SVC(random_state=42, **kwargs)
    clf.fit(df[features], df["target"])
    return clf


def predict_score(clf, x1, x2):
    return clf.decision_function(pd.DataFrame(dict(length=x1, width=x2), index=[0]))[0]


def print_formula(clf):
    formula = f"y = {clf.intercept_[0]:.2f} + {clf.coef_[0][0]:.2f}*x1 + {clf.coef_[0][1]:.2f}*x2"
    print(formula)
    return formula
