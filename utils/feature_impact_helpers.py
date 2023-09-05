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

    df_subset['max_strength'] = max(df_subset['strength'])
    df_subset['normalized_strength'] = df_subset.apply(lambda x: x['strength'] / x['max_strength'], axis=1)
    
    return df_subset.drop('max_strength', axis=1)

def plot_feature_impact(
    df: pd.DataFrame,
    n: int = 25,
    title: str = "<b>Feature Impact<b>",
    normalized=True,
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
    col = "normalized_strength" if normalized else "strength" 
    tickformat = ",.0%" if normalized else "0"

    fig = px.bar(
        df_subset,
        y="feature_name",
        x=col,
        orientation="h",
        height=height,
    )
    fig.update_traces(
        hovertemplate="<b>Feature Name:</b> %{y} <br><b>Feature Strength:</b> %{x}<extra></extra>"
    )

    fig.update_layout(
        #title={"text": f"{title}"},
        plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=DEFAULT_HOVER_LABEL,
    )

    fig.update_yaxes({
        'title': "Feature Name",
        'tickvals': list(range(len(df_subset['feature_name']))),
        'ticktext': df_subset['feature_name'].str.slice(0,45).tolist(),
        },
        showline=True, 
        linewidth=2, 
        linecolor='black',
    )

    fig.update_xaxes(
        tickformat = tickformat,
        #tickmode='linear',
        ticks="inside",
        # tickfont=dict(
        #     size=15,
        # ),
        #tickson="boundaries",
        ticklen=6,
        title="Impact",
        showline=True, 
        linewidth=2, 
        linecolor='black',
    )

    return fig

def plot_signed_feature_impact(
    df: pd.DataFrame,
    n: int = 25,
    title: str = "<b>Feature Impact</b>",
    normalized = True,
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

    col = "pct_strength" if normalized else "strength" 
    tickformat = ",.0%" if normalized else "0"

    df["positive_strength"] = np.where(
        df["strength"] >= 0, "positive", "negative"
    )

    df = (
        df.groupby(["feature_name", "positive_strength"])["strength"]
        .apply(lambda x: np.abs(x).sum())
        .reset_index()
    )
    
    strength_index = dict(
        df.groupby("feature_name")["strength"].sum()
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
        .sort_values(by=["sort_key", "strength"], ascending=True)
        .drop(columns=["sort_key"])
        .reset_index(drop=True)
    )

    plot_ready_data["total_strength"] = plot_ready_data.groupby("feature_name")["strength"].transform(
        lambda x: np.abs(x).sum()
    )
    
    cols_to_keep = plot_ready_data["feature_name"].unique()[-n:]
    plot_ready_data = plot_ready_data.loc[plot_ready_data["feature_name"].isin(cols_to_keep), :]

    plot_ready_data['pct_strength'] = plot_ready_data['strength'] / plot_ready_data['total_strength']
    plot_ready_data['pct_strength'] = np.where(
        pd.isna(plot_ready_data['pct_strength']),
        0.0,
        plot_ready_data['pct_strength'],
    )

    plot_ready_data['max_strength'] = max(plot_ready_data['total_strength'])
    plot_ready_data['normalized_strength'] = plot_ready_data.apply(lambda x: x['total_strength'] / x['max_strength'], axis=1)
    
    plot_ready_data['pct_strength'] = plot_ready_data['pct_strength'] * plot_ready_data['normalized_strength']

    x_pos = plot_ready_data.loc[plot_ready_data.positive_strength == "positive"]

    # x_pos['max_strength'] = max(x_pos['strength'])
    # x_pos['normalized_strength'] = x_pos.apply(lambda x: (x['strength'] / x['max_strength']) / 2, axis=1)

    y = cols_to_keep

    x_neg = plot_ready_data.loc[plot_ready_data.positive_strength == "negative"]

    # x_neg['max_strength'] = max(x_neg['strength'])
    # x_neg['normalized_strength'] = x_neg.apply(lambda x: (x['strength'] / x['max_strength']) / 2, axis=1)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=y, 
            x=x_pos[col], 
            name="Positive Impact", 
            orientation="h",
        )
    )
    fig.add_trace(
        go.Bar(
            y=y, 
            x=x_neg[col], 
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
        plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=DEFAULT_HOVER_LABEL,
    )
    fig.update_yaxes({
        'categoryorder':'total ascending',
        'title': "Feature Name",
        'tickvals': list(range(len(x_pos['feature_name']))),
        'ticktext': x_pos['feature_name'].str.slice(0,45).tolist(),
        },
        showline=True, 
        linewidth=2, 
        linecolor='black',
    )
    fig.update_xaxes(
        title="Impact",
        tickformat=tickformat,
        ticks="inside",
        ticklen=6,
        showline=True, 
        linewidth=2, 
        linecolor='black',
    )

    fig.update_traces(
        hovertemplate="<b>Feature Name:</b> %{y} <br><b>Feature Strength:</b> %{x}<extra></extra>"
    )

    return fig