import datarobot as dr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



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


def add_bins_to_data(
    project_id: str,
    df: pd.DataFrame,
    bins: int = 10,
    data_subset: str = dr.enums.DATA_SUBSET.HOLDOUT,
    **kwargs,
) -> pd.DataFrame:
    """
    Adds bins to the input DataFrame based on quantiles and returns the modified DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        bins (int, optional): The number of bins to create. Defaults to 10.
        data_subset (str, optional): The data subset to use. Defaults to dr.enums.DATA_SUBSET.HOLDOUT.

    Returns:
        pd.DataFrame: The modified DataFrame with bins added.
    """
    col = list(get_column_name_mappings(project_id))[0]
    df = df.loc[~pd.isna(df[col]), :].reset_index(drop=True)

    df["bins"] = pd.qcut(
        df[col],
        q=bins,
        labels=False,
        duplicates="drop",
    )
    mean = lambda x: np.average(x)

    func = {col: mean, "actuals": mean}

    return df, func


def group_data_by_bins(
    df: pd.DataFrame,
    project_id: str,
    aggregation_functions: dict,
    bins: int = 10,
) -> pd.DataFrame:
    """
    Groups the data by bins and calculates the aggregated values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target (str): The target column name.
        aggregation_functions (dict): A dictionary of aggregation functions to apply.
        positive_class (str): The positive class label.
        bins (int, optional): The number of bins to create. Defaults to 10.

    Returns:
        pd.DataFrame: The DataFrame grouped by bins with aggregated values.
    """
    project = dr.Project.get(project_id)
    project_type = project.target_type
    positive_class = project.positive_class
    target = project.target
    
    # Set 'actuals' column values based on the target and positive class
    df["actuals"] = np.where(df[target] == positive_class, 1, 0)

    # Group the data by 'bins' and aggregate the data using the provided functions
    binned_data = df.groupby(["bins"]).agg(aggregation_functions).reset_index()

    # Assign a range of bin numbers to the 'bins' column
    binned_data["bins"] = range(1, bins + 1)

    return binned_data


def plot_lift_chart(
    df: pd.DataFrame,
) -> go.Figure:
    """
    Plots a lift chart using a DataFrame with grouped data by bins.

    Args:
        df (pd.DataFrame): DataFrame containing binned data.
        column_name_mappings (dict): Dictionary of column name mappings.
        weights (bool, optional): Whether the data is weighted or not. Defaults to False.

    Returns:
        go.Figure: A Plotly figure object containing the lift chart.
    """

    # Get the column name for the prediction values
    col = list(column_name_mappings.values())[0]

    # Create the figure object
    fig = go.Figure()

    # Add the trace for the predictions
    fig.add_trace(
        go.Scatter(
            x=df["bins"],
            y=df[col],
            mode="lines+markers",
            name="Predictions",
            marker=dict(
                size=5,
                color="blue",
                symbol="cross-open",
                line=dict(
                    color="blue",
                    width=2,
                ),
            ),
            line=dict(
                color="blue",
                width=2,
            ),
        )
    )

    # Add the trace for the actuals
    fig.add_trace(
        go.Scatter(
            x=df["bins"],
            y=df["actuals"],
            mode="lines+markers",
            name="Actuals",
            marker=dict(
                size=5,
                color="#ff7f0e",
                symbol="circle-open",
                line=dict(
                    color="#ff7f0e",
                    width=1,
                ),
            ),
            line=dict(
                color="#ff7f0e",
                width=2,
            ),
        )
    )

    # Update the layout of the figure
    fig.update_layout(
        title={
            "text": f"<b>Lift Chart</b>",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        legend=dict(
            bgcolor="rgba(255, 255, 255, 0)",
            bordercolor="rgba(255, 255, 255, 0)",
        ),
        legend_title=f"Data: ",
    )

    # Update the y-axis and x-axis titles
    fig.update_yaxes(title=f"Average Target")
    fig.update_xaxes(title=f"Bins")

    return fig


def get_prediction_explanations_per_bin(
    df: pd.DataFrame,
    project_id: str,
    max_features: int = 5,
    **kwargs,
) -> pd.DataFrame:
    """
    Get prediction explanations for each bin by calculating the
    aggregated feature strengths and values.

    Args:
        df (pd.DataFrame): DataFrame containing feature strengths and values.
        project_id (str): Project identifier.
        max_features (int, optional): Maximum number of features to keep. Defaults to 5.
        **kwargs: Additional keyword arguments.

    Returns:
        pd.DataFrame: A DataFrame containing aggregated feature strengths and values per bin.
    """

    # Calculate the sum of absolute feature strengths for each feature
    ranked_features = (
        df.groupby("feature_name")["strength"]
        .apply(lambda x: np.abs(x).sum())
        .reset_index()
        .sort_values(by="strength", ascending=True)
    )

    # Keep only the top max_features features
    features_to_keep = ranked_features[-max_features:]["feature_name"]
    
    # Add bins to the data
    binned_data, _ = add_bins_to_data(project_id, df, bins=10, **kwargs)

    # Define a function to compute the median if possible, otherwise return the mode
    def try_mean_else_mode(x):
        try:
            return x.astype(float).median()
        except:
            return x.value_counts().index[0]

    # Group the binned data by 'bins' and 'feature_name', then aggregate
    grouped = (
        binned_data.groupby(["bins", "feature_name"])[
            "strength", "actual_value"
        ]
        .agg(
            strength=("strength", "sum"),
            actual_value=("actual_value", try_mean_else_mode),
        )
        .sort_values(by="strength", ascending=True)
        .reset_index()
    )

    # Filter the grouped data by keeping only the top features
    filtered_df = grouped.loc[grouped["feature_name"].isin(features_to_keep), :].copy()
    
    # Increment the 'bins' column values by 1
    filtered_df["bins"] += 1

    return filtered_df


def plot_prediction_explanations_and_lift_chart(
    df: pd.DataFrame,
    # grouped_df: pd.DataFrame,
    project_id: str,
    bins: int=10,
    max_features: int=5,
    showlegend: bool=False,
    **kwargs,
):
    """
    Create a Plotly figure with prediction explanations and lift chart.

    Args:
        df (pd.DataFrame): DataFrame containing feature strengths and values.
        grouped_df (pd.DataFrame): DataFrame containing aggregated feature strengths and values per bin.
        project_id (str): Project identifier.
        bins (int): Number of bins in the lift plot.
        **kwargs: Additional keyword arguments.

    Returns:
        plotly.graph_objs.Figure: The generated Plotly figure.
    """
    binned, func = add_bins_to_data(project_id, df, bins=bins)
    df1 = group_data_by_bins(binned, project_id, func)
    df2 = get_prediction_explanations_per_bin(df, project_id, max_features=max_features)

    DEFAULT_HOVER_LABEL = dict(
        bgcolor="white", font_size=16, font_family="Rockwell", namelength=-1
    )
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    col = list(get_column_name_mappings(project_id))[0]
    
    # Add predictions
    fig.add_trace(
        go.Scatter(
            x=df1["bins"],
            y=df1[col],
            mode="lines+markers",
            name="Predictions",
            marker=dict(
                size=5,
                color="blue",
                symbol="cross-open",
                line=dict(
                    color="blue",
                    width=2,
                ),
            ),
            line=dict(
                color="blue",
                width=2,
            ),
        ),
        secondary_y=True,
    )
    
    # Add actuals
    fig.add_trace(
        go.Scatter(
            x=df1["bins"],
            y=df1["actuals"],
            mode="lines+markers",
            name="Actuals",
            marker=dict(
                size=5,
                color="#ff7f0e",
                symbol="circle-open",
                line=dict(
                    color="#ff7f0e",
                    width=1,
                ),
            ),
            line=dict(
                color="#ff7f0e",
                width=2,
            ),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title={
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        legend=dict(
            bgcolor="rgba(255, 255, 255, 0)",
            bordercolor="rgba(255, 255, 255, 0)",
        ),
        legend_title="Features: ",
        hoverlabel=DEFAULT_HOVER_LABEL,
    )
    fig.update_yaxes(title=f"Average Target")
    fig.update_xaxes(title=f"Bins")
    fig.update_layout(
        barmode="relative",
        yaxis2={"title": "Average Prediction", "tickformat": ",.0%"},
    )
    
    # Add bins
    features = np.sort(df2["feature_name"].unique())
    colors = px.colors.qualitative.Plotly[0 : len(features)]
    marker_color = {column: color for column, color in zip(features, colors * 5)}

    for trace in features:
        dft = df2[df2["feature_name"] == trace]
        median_val = (
            "<br>Most Frequent Value</b>: %{customdata[0]}"
            if isinstance(dft["actual_value"].iloc[0], str)
            else "<br>Median Value</b>: %{customdata[0]: .3}"
        )
        fig.add_traces(
            go.Bar(
                x=dft["bins"],
                y=dft["strength"],
                name=trace,
                marker_color=marker_color[trace],
                opacity=0.5,
                customdata=dft[
                    ["actual_value", "strength", "feature_name"]
                ],
                hovertemplate="<br>Bin</b>: %{x}"
                + "<br>Feature</b>: %{customdata[2]}"
                + "<br>Strength</b>: %{y: .2}"
                + median_val
                + "<extra></extra>",
                hoverlabel=DEFAULT_HOVER_LABEL,
            )
        )

    fig.update_yaxes(title="Feature Strength")
    fig.update_xaxes(title="Bins")
    fig.update_layout(
        yaxis2={"title": "Average Prediction", "tickformat": ",.0%"},
        height=600,
        legend_title="Features: ",
        hoverlabel=DEFAULT_HOVER_LABEL,
    )
    fig.update_layout(showlegend=showlegend)

    return fig