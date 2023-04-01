import pandas as pd
import datarobot as dr
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

DEFAULT_HOVER_LABEL = dict(
    bgcolor="white", font_size=16, font_family="Rockwell", namelength=-1
)

def get_column_name_mappings(project_id) -> dict:
    """
    Returns a dictionary of column name mappings based on the project type.

    Returns:
        dict: A dictionary containing the column name mappings.
    """
    project = dr.Project.get(project_id)
    project_type = project.target_type
    positive_class = project.positive_class
    target = project.target

    if project_type == "regression":
        name_mappings = {
            f"{target}_PREDICTION": "Prediction",
        }
    else:
        name_mappings = {
            # f"{target}_{positive_class}_PREDICTION": "Class_1_Prediction",
            # f"{target}_PREDICTION": "Prediction",
            f"class_{positive_class}"
        }

    return name_mappings


def prep_pe_data(
    df: pd.DataFrame,
    project_id: str,
    date_col: str,
    freq: str="QS",
    max_features: int=5,
) -> pd.DataFrame:

    # Calculate the sum of absolute feature strengths for each feature
    ranked_features = (
        df.groupby("feature_name")["strength"]
        .apply(lambda x: np.abs(x).sum())
        .reset_index()
        .sort_values(by="strength", ascending=True)
    )

    # Keep only the top max_features features
    features_to_keep = ranked_features[-max_features:]["feature_name"]

    # Filter the grouped data by keeping only the top features
    filtered_df = df.loc[df["feature_name"].isin(features_to_keep), :].reset_index(drop=True).copy()

    filtered_df[date_col] = pd.to_datetime(filtered_df[date_col])
    group = ["feature_name"]
    pred_col = list(get_column_name_mappings(project_id))[0]
    
    def try_mean_else_mode(x):
        try:
            return x.astype(float).median()
        except:
            return x.value_counts().index[0]

    counts = (
        filtered_df.resample(freq, on=date_col)
        .agg(
            {
                "row_id": "nunique",
                "strength": "count",
                pred_col: "mean",
            }
        )
        .reset_index()
        .rename(
            {
                "row_id": "row_count",
                "strength": "count",
                pred_col: "average_prediction",
            },
            axis=1,
        )
    )

    resampled = (
        filtered_df.groupby(group)
        .resample(freq, on=date_col)
        .agg({
            "strength": "sum",
            "actual_value": try_mean_else_mode,
        })
        .reset_index()
    )

    normalized = resampled.merge(
        counts[[date_col, "row_count"]],
        how="left",
        on=[date_col],
    )

    normalized["strength_normalized"] = (
        normalized["strength"] / normalized["row_count"]
    )

    barplot_data = normalized

    scatterplot_data = (
        filtered_df[[date_col, pred_col]]
        .resample(freq, on=date_col)
        .mean()
        .reset_index()
        .rename({pred_col: "Average Prediction"}, axis=1)
    )

    return barplot_data, scatterplot_data


def plot_pe_over_time(
    bar: pd.DataFrame, 
    scatter: pd.DataFrame,
    date_col: str,
    showlegend: bool=False,
    height: int=600,
) -> go.Figure:
    """
    Plot the normalized feature strengths and average predictions over time.

    Parameters
    ----------
    bar : pandas.DataFrame
        The feature strength data for the bar plot.
    scatter : pandas.DataFrame
        The average prediction data for the scatter plot.
    date_col : str
        The name of the date column in both bar and scatter.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        The plotly figure object containing the plot.

    """

    # Create subplot
    fig = make_subplots(
        specs=[[{"secondary_y": True}]]
    )

    # Add scatter plot for average prediction
    fig.add_trace(
        go.Scatter(
            x=scatter[date_col],
            y=scatter["Average Prediction"],
            name="Average Prediction",
            mode="lines+markers",
            marker=dict(
                color="Black",
                size=6,
            ),
        ),
        secondary_y=True,
    )

    fig.update_layout(barmode="relative")

    # Add bar plot for feature strengths
    features = np.sort(bar["feature_name"].unique())
    colors = px.colors.qualitative.Plotly * 10
    colors = colors[0 : len(features)]
    marker_color = {column: color for column, color in zip(features, colors)}

    for i, trace in enumerate(bar["feature_name"].unique()):
        dft = bar[bar["feature_name"] == trace]
        median_val = (
            "<br>Most Frequent Value</b>: %{customdata[0]}"
            if isinstance(dft["actual_value"].iloc[0], str)
            else "<br>Median Value</b>: %{customdata[0]: .3}"
        )
        fig.add_traces(
            go.Bar(
                x=dft[date_col],
                y=dft["strength_normalized"],
                name=trace,
                opacity=0.5,
                marker_color=marker_color[trace],
                customdata=dft[
                    ["actual_value", "strength", "feature_name"]
                ],
                hovertemplate="<br>Period</b>: %{x}"
                + "<br>Feature</b>: %{customdata[2]}"
                + "<br>Strength</b>: %{y: .2}"
                + median_val
                + "<extra></extra>",
                hoverlabel={
                    "align": "left",
                    "bgcolor": "white",
                    "font_size": 16,
                    "font_family": "Rockwell",
                },
            )
        )

    # Update axes and layout
    fig.update_yaxes(title="Normalized Feature Strength")
    fig.update_xaxes(title="Date")
    fig.update_layout(
        yaxis2={
            "title": "Average Prediction", 
            "tickformat": ",.0%"
        },
        hoverlabel=DEFAULT_HOVER_LABEL,
        legend=dict(
            x=1.1,
            y=1,
            xanchor='left',
            yanchor='top'
        ),
        showlegend=showlegend, 
        height=height,
    )

    return fig


def prep_and_plot_pe_over_time(
    df: pd.DataFrame, 
    project_id: str,
    date_col: str,
    freq: str='QS',
    max_features: int=5,
    showlegend: bool=False,
    height: int=600,
) -> go.Figure:
    """
    Prep and plot the normalized feature strengths and average predictions over time.

    Parameters
    ----------
    df : pandas.DataFrame
        The melted feature strength data for the plot.
    project_id : str
        The project identifier.
    date_col : str
        The name of the date column in both bar and scatter.
    frew : str
        The aggregated time periods displayed in the plot.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        The plotly figure object containing the plot.

    """

    # Prep data
    a, b = prep_pe_data(
        df=df, 
        project_id=project_id, 
        date_col=date_col, 
        freq=freq,
        max_features=max_features,
    )

    fig = plot_pe_over_time(a, b, date_col, showlegend=showlegend, height=height)

    return fig