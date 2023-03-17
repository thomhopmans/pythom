import plotly.graph_objects as go
import plotly.io as pio

pio.templates["pythom"] = go.layout.Template(
    layout=go.Layout(
        title_font=dict(family="Rockwell", size=24),
        annotationdefaults=dict(font=dict(color="crimson")),
        annotations=[
            dict(
                name="internal use watermark",
                text="INTERNAL USE ONLY",
                textangle=0,
                opacity=0.15,
                font=dict(color="black", size=16),
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.20,
                showarrow=False,
            )
        ],
        colorscale={
            "diverging": [
                [0, "#00abe9"],
                [1, "#09b54e"],
            ],
            "sequential": [
                [0.0, "#00abe9"],
                [1.0, "#09b54e"],
            ],
            "sequentialminus": [
                [0.0, "#00abe9"],
                [1.0, "#09b54e"],
            ],
        },
        colorway=[
            "#00abe9",
            "#09b54e",
            "#636efa",
            "#EF553B",
            "#00cc96",
            "#ab63fa",
            "#FFA15A",
            "#19d3f3",
            "#FF6692",
            "#B6E880",
            "#FF97FF",
            "#FECB52",
        ],
    ),
    data=dict(
        scatter=[
            go.Scatter(marker=dict(symbol="diamond", size=10)),
            go.Scatter(marker=dict(symbol="square", size=10)),
            go.Scatter(marker=dict(symbol="circle", size=10)),
        ]
    ),
)
