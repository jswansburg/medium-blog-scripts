import datarobot as dr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple
from itertools import product

DEFAULT_HOVER_LABEL = dict(
    bgcolor="white", font_size=16, font_family="Rockwell", namelength=-1
)

def prep_feature_impact(
    df: pd.DataFrame,
    n: int = 25,
) -> pd.DataFrame:
    """
    Calculate the total absolute feature strength for each feature in the input DataFrame and 
    return the top n features sorted by their absolute feature strength in ascending order.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'feature_name' and 'strength' columns.
    n : int, optional, default=25
        Number of top features to return based on their absolute feature strength.

    Returns
    -------
    df_subset : pd.DataFrame
        Subset of the input DataFrame containing the top n features sorted by their absolute
        feature strength in ascending order.
    """

    df = df.groupby("feature_name")["strength"].apply(
        lambda x: np.abs(x).sum()
    )
    df_subset = df.reset_index().sort_values(by="strength", ascending=True)[-n:]

    return df_subset

def plot_feature_impact(
    df: pd.DataFrame,
    n: int = 25,
    title: str = "<b>Feature Impact<b>",
    height: int = 600,
) -> go.Figure:
    """
    Plot a horizontal bar chart of the top n features based on their absolute feature strength
    from the input DataFrame using Plotly Express.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'feature_name' and 'strength' columns.
    n : int, optional, default=25
        Number of top features to plot based on their absolute feature strength.
    title : str, optional, default="<b>Feature Impact<b>"
        Title of the plot.
    height : int, optional, default=400
        Height of the plot in pixels.
    prep_feature_impact_func : function, optional, default=prep_feature_impact
        Function to prepare the feature impact data.

    Returns
    -------
    fig : px.Figure
        Plotly Express Figure object containing the horizontal bar chart.
    """

    df_subset = prep_feature_impact(df, n)
    
    fig = px.bar(
        df_subset,
        y="feature_name",
        x="strength",
        orientation="h",
        height=height,
    )
    fig.update_traces(
        hovertemplate="<b>Feature Name:</b> %{y} <br><b>Feature Strength:</b> %{x}<extra></extra>"
    )

    fig.update_layout(
        #title={"text": f"{title}"},
        hoverlabel=DEFAULT_HOVER_LABEL,
    )

    fig.update_yaxes({
        'title': "Feature Name",
        'tickvals': list(range(len(df_subset['feature_name']))),
        'ticktext': df_subset['feature_name'].str.slice(0,35).tolist(),
    })

    fig.update_xaxes(title="Impact")

    return fig

def plot_signed_feature_impact(
    df: pd.DataFrame,
    n: int = 25,
    title: str = "<b>Feature Impact</b>",
    height: int = 600,
) -> go.Figure:
    """
    Plot a stacked horizontal bar chart of the top n features based on their absolute positive and negative
    feature strength from the input DataFrame using Plotly.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing 'feature_name' and 'strength' columns.
    n : int, optional (default=25)
        The number of features to plot based on their absolute feature strength.
    title : str, optional (default="<b>Feature Impact</b>")
        The title of the plot. Uses bold text by default.
    height : int, optional (default=600)
        The height of the plot in pixels.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        A stacked horizontal bar chart of the top n features based on their absolute positive and negative
        feature strength.

    Raises
    ------
    ValueError
        If the input DataFrame doesn't contain the 'feature_name' and 'strength' columns.

    """
    df = df.copy()

    df["positive_strength"] = np.where(
        df["strength"] >= 0, "positive", "negative"
    )
    df = (
        df.groupby(["feature_name", "positive_strength"])["strength"]
        .apply(lambda x: np.abs(x).sum())
        .reset_index()
    )
    df["abs_strength"] = df.groupby("feature_name")["strength"].transform(
        lambda x: np.abs(x).sum()
    )
    strength_index = dict(
        df.groupby("feature_name")["abs_strength"].sum()
    )

    names_df = pd.DataFrame(
        list(product(df.feature_name.unique(), ["positive", "negative"])),
        columns=["feature_name", "positive_strength"],
    ).assign(tmp=1)

    plot_ready_data = (
        df.merge(names_df, how="outer")
        .drop(columns="tmp")
        .fillna(0)
        .assign(sort_key=lambda x: x.feature_name.map(strength_index))
        .sort_values(by=["sort_key", "positive_strength"], ascending=True)
        .drop(columns=["sort_key"])
        .reset_index(drop=True)
    )
    y = plot_ready_data.feature_name.unique()[-n:]
    x_pos = plot_ready_data.loc[plot_ready_data.positive_strength == "positive"][
        -n:
    ]
    x_neg = plot_ready_data.loc[plot_ready_data.positive_strength == "negative"][
        -n:
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=y, 
            x=x_pos.abs_strength, 
            name="Positive Impact", 
            orientation="h",
        )
    )
    fig.add_trace(
        go.Bar(
            y=y, 
            x=x_neg.abs_strength, 
            name="Negative Impact", 
            orientation="h"
        )
    )

    fig.update_layout(
        barmode="stack",
        margin=dict(
            l=20, 
            r=0, 
            t=20, 
            b=20,
        ),
        height=height,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
        ),
    )

    fig.update_layout(
        #title={"text": f"{title}"},
        hoverlabel=DEFAULT_HOVER_LABEL,
    )
    fig.update_yaxes({
        'title': "Feature Name",
        'tickvals': list(range(len(x_pos['feature_name']))),
        'ticktext': x_pos['feature_name'].str.slice(0,35).tolist(),
    })
    fig.update_xaxes(title="Impact")

    fig.update_traces(
        hovertemplate="<b>Feature Name:</b> %{y} <br><b>Feature Strength:</b> %{x}<extra></extra>"
    )

    return fig