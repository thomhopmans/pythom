from svm_2d import (
    get_contour,
    get_parallels,
    get_separating_hyperplane,
)
from svm_3d import hyperplane_from_3d_to_2d
from template import go

MARKER_COLORS = {
    "rejected": "#EA4763",
    "approved": "#09B54E",
}
PLOT_WIDTH = 1600
PLOT_HEIGHT = 900


def get_figure(title, x_range=[4, 8], y_range=[2, 5]):
    return go.Figure(
        layout=go.Layout(
            title=title,
            template="plotly_dark+pythom",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            xaxis=dict(range=x_range, title="Problems", title_font_size=18),
            yaxis=dict(range=y_range, title="Money", title_font_size=18),
            width=PLOT_WIDTH,
            height=PLOT_HEIGHT,
        )
    )


def get_figure_3d(title):
    return go.Figure(
        layout=go.Layout(
            title=title,
            legend=dict(
                orientation="h",
            ),
            scene=dict(
                xaxis=dict(
                    range=[-1.5, 1.5],
                    title="Number of problems",
                    title_font_size=18,
                ),
                yaxis=dict(
                    range=[-1.5, 1.5],
                    title="Money on the bank",
                    title_font_size=18,
                ),
                zaxis=dict(
                    range=[0, 2],
                    title="Kernel calculation",
                    title_font_size=18,
                ),
            ),
            width=PLOT_WIDTH,
            height=PLOT_HEIGHT,
            template="plotly_dark+pythom",
            margin=dict(l=20, r=20, t=60, b=10),
            autosize=False,
            scene_camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.75, y=1.75, z=0.05),
            ),
        )
    )


# Traces: Observations
def add_observations(fig, df):
    for name, category_df in df.groupby("name"):
        fig.add_trace(
            go.Scatter(
                x=category_df["length"],
                y=category_df["width"],
                mode="markers",
                name=name,
                marker_color=MARKER_COLORS[name],
            )
        )
    return fig


def add_observations_3d(fig, df):
    for name, category_df in df.groupby("name"):
        fig.add_trace(
            go.Scatter3d(
                x=category_df["length"],
                y=category_df["width"],
                z=category_df["z"],
                mode="markers",
                name=name,
                marker_color=MARKER_COLORS[name],
            )
        )
    return fig


def add_hyperplane(fig, clf, x_start=4, x_end=8, **kwargs):
    sh_xx, sh_yy, _ = get_separating_hyperplane(clf, x_start=x_start, x_end=x_end)
    fig.add_trace(
        go.Scatter(
            x=sh_xx,
            y=sh_yy,
            mode="lines",
            name="separating hyperplane",
            marker_color="white",
            line_width=4,
            **kwargs,
        )
    )

    return fig


def add_parallels(fig, clf, space=True, x_start=4, x_end=8):
    sh_xx, _, _ = get_separating_hyperplane(clf, x_start=x_start, x_end=x_end)
    yy_down, yy_up, margin = get_parallels(clf, x_start=x_start, x_end=x_end)
    print("Margin:", margin)

    fig.add_trace(
        go.Scatter(
            x=sh_xx,
            y=yy_down,
            mode="lines",
            name=f"Margin (d={margin:.2f})",
            legendgroup="margin",
            marker_color="white",
            line_dash="dash",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sh_xx,
            y=yy_up,
            mode="lines",
            name=f"Margin (d={margin:.2f})",
            legendgroup="margin",
            showlegend=False,
            marker_color="white",
            line_dash="dash",
        )
    )

    # Add space
    if space:
        path = (
            f" M {sh_xx[0]} {yy_down[0]} L {sh_xx[-1]} {yy_down[-1]}"
            + f" L {sh_xx[-1]} {yy_up[-1]} L {sh_xx[0]} {yy_up[0]} Z"
        )
        fig.add_shape(
            dict(
                type="path",
                path=path,
                fillcolor="White",
                line_color="White",
                opacity=0.5,
                layer="below",
            ),
        )

    return fig


def add_contours(fig, clf, inverse_colorscale: bool = False):
    colorscale = [[0, "red"], [1, "green"]]
    if inverse_colorscale:
        colorscale = [[0, "green"], [1, "red"]]

    x, y, z = get_contour(clf, clip=True, interval=0.01)
    fig.add_trace(
        go.Contour(
            z=z,
            x=x,
            y=y,
            contours=dict(
                start=0,
                end=0,
            ),
            colorscale=colorscale,
            opacity=0.3,
            showscale=False,
        )
    )
    return fig


def annotate_support_vectors(fig, clf):
    for sv in clf.support_vectors_:
        fig.add_shape(
            type="circle",
            xref="x",
            x0=sv[0] - 0.05,
            y0=sv[1] - 0.05,
            x1=sv[0] + 0.05,
            y1=sv[1] + 0.05,
            line_color="white",
            line_width=2,
        )
    return fig


def add_line(fig, x, y):
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            marker_color="white",
            line_dash="dash",
            showlegend=False,
            line_width=2,
        )
    )
    return fig


def add_point(fig, x, y, color):
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker_color=color,
            showlegend=False,
        )
    )
    return fig


# def add_hyperplane_3d(fig, clf):
#     xx, yy, zz = get_separating_hyperplane_3d(clf, x_start=-1.5, x_end=1.5)
#     colorscale = [[0, "rgb(255, 255, 255)"], [1, "rgb(255, 255, 255)"]]
#     fig.add_trace(go.Surface(x=xx, y=yy, z=zz, colorscale=colorscale, showscale=False))
#     return fig


def add_hyperplane_3d(fig, clf):
    hyperplane_coordinates = hyperplane_from_3d_to_2d(clf=clf)

    fig.add_trace(
        go.Scatter(
            x=hyperplane_coordinates[:, 0],
            y=hyperplane_coordinates[:, 1],
            mode="markers",
            name="separating_hyperplane",
            marker_color="white",
            marker_size=5,
        )
    )
    return fig
