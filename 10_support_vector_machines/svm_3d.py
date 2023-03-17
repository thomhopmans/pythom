import numpy as np


def print_formula_3d(clf):
    formula = f"y = {clf.intercept_[0]:.2f}"
    formula += f" + {clf.coef_[0][0]:.2f}*x1 + {clf.coef_[0][1]:.2f}*x2 + {clf.coef_[0][2]:.2f}*x3"
    print(formula)

    return formula


def get_separating_hyperplane_3d(clf, x_start=4, x_end=8):
    if len(clf.coef_[0]) != 3:
        raise ValueError("Not supported for other than 3 features")

    # The equation of the separating plane is given by all x in R^3 such that:
    # np.dot(svc.coef_[0], x) + b = 0. We should solve for the last coordinate
    # to plot the plane in terms of x and y.

    def z(x, y):
        return (
            -clf.intercept_[0] - clf.coef_[0][0] * x - clf.coef_[0][1] * y
        ) / clf.coef_[0][2]

    tmp = np.linspace(x_start, x_end, 51)
    x, y = np.meshgrid(tmp, tmp)

    return x, y, z(x, y)


def hyperplane_from_3d_to_2d(clf):
    """Reduces a 3D hyperplane with a quadratic kernel, i.e. x^2+y^2, to 2D"""

    def quadratic(x, y):
        return (x**2) + (y**2)

    results = []

    # Calculate z over a grid
    for x in np.linspace(-2, 2, 1000):
        for y in np.linspace(-2, 2, 1000):
            z = quadratic(x, y)
            z_hat = get_z(clf, x, y)

            if np.round(z, 2) == np.round(z_hat, 2):
                results.append([x, y])

    return np.array(results)


def get_z(clf, x, y):
    coeff = clf.coef_[0]
    return (-clf.intercept_[0] - coeff[0] * x - coeff[1] * y) / coeff[2]
