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
    fig.update_yaxes(
        title="Normalized Feature Strength",
        showline=True, 
        linewidth=1, 
        linecolor='black',
        showgrid=True, 
        gridwidth=0.1, 
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=2, 
        zerolinecolor='black',
    )
    fig.update_xaxes(
        title="Date",
        showline=True, 
        linewidth=1, 
        linecolor='black',
        showgrid=False, 
        gridwidth=1, 
        gridcolor='#7f7f7f',
    )
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
        plot_bgcolor='rgba(0,0,0,0)',
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

    fig = plot_pe_over_time(
        a, 
        b, 
        date_col, 
        showlegend=showlegend, 
        height=height
    )

    return fig

def plot_values_over_time(
    df,
    project_id: str,
    feature: str,
    date_col: str, 
    freq: str='QS',
    class_type: str='actuals',
    height: int=500,
    showlegend: bool=False,
):
    """
    Plot the distribution of a feature over time, either as a bar chart (for categorical features)
    or as a line chart (for numerical features) showing percentiles and mean.
    
    Args:
        df (pd.DataFrame): Input dataframe containing the data to be plotted.
        project_id (str): DataRobot project ID.
        feature (str): Feature column name to be plotted.
        date_col (str): Date column name.
        freq (str, optional): Frequency for resampling (e.g., 'M' for month, 'Q' for quarter, 'Y' for year).
        class_type (str, optional): Whether to display the predicted or actual class labels. Defautls to actual.
        height (int, optional): Height of the plot. Defaults to 500.
        
    Returns:
        fig (plotly.graph_objects.Figure): A Plotly figure of the distribution of the feature over time.
    """
    mapping = {
        'H':'H',
        'W':'W',
        'MS':'M',
        'M':'M',
        'QS':'Q',
        'Q':'Q',
        'YS':'Y',
        'Y':'Y',
    }
    freq = mapping[freq]
    
    # Convert date column to datetime and create an 'index' column with resampled date periods
    df[date_col] = pd.to_datetime(df[date_col])
    df["index"] = (
        pd.Series(df[date_col].dt.to_period(freq))
        .astype(str)
        .reset_index(drop=True)
    )
    
    # Create subplot
    fig = make_subplots(
        specs=[[{"secondary_y": True}]]
    )

    # Calculate average predictions per time period
    pred_col = list(get_column_name_mappings(project_id))[0]

    df["index"] = pd.to_datetime(df["index"])
    
    target = dr.Project.get(project_id).target
    col = pred_col if class_type=='predictions' else target

    scatterplot_data = (
        df[["index", col]]
        .groupby("index")
        .mean()
        .reset_index()
        .rename({col: "Average Prediction"}, axis=1)
    )
        
    # Add scatter plot for average prediction
    fig.add_trace(
        go.Scatter(
            x=scatterplot_data["index"],
            y=scatterplot_data["Average Prediction"],
            name=f"Average {class_type.capitalize()}",
            mode="lines+markers",
            marker=dict(
                color="Black",
                size=6,
            ),
        ),
        secondary_y=True,
    )
    
    # Check if the feature is numerical
    if np.issubdtype(df[feature].dtype, np.number):
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Resample and calculate percentiles and mean
        resampled = (
            df[[feature, "index"]]
            .groupby("index")
            .agg({
                feature: [
                    lambda x: np.percentile(x, 25),
                    lambda x: np.percentile(x, 50),
                    lambda x: np.mean(x),
                    lambda x: np.percentile(x, 75),
                ]
            }).reset_index()
        )
        resampled.columns = [date_col, '25th Percentile', '50th Percentile', 'Average', '75th Percentile']
        
        # Configure traces and styles for the line chart
        traces = ['25th Percentile', '50th Percentile', 'Average', '75th Percentile']
        colors = ['lightblue', 'blue', 'blue', 'lightblue']
        line_type = [None, None, 'dash', None]
        
        # Create the line chart
        for trace, color, line_type in zip(traces, colors, line_type):
            fig.add_trace(
                go.Scatter(
                    x=resampled[date_col],
                    y=resampled[trace],
                    mode='lines+markers',
                    name=trace,
                    line=dict(dash=line_type, width=3),
                    line_color=color,
                )
            )
            fig.update_yaxes(
                title=f"{feature}",
                showline=True, 
                linewidth=1, 
                linecolor='black',
                showgrid=True, 
                gridwidth=1, 
                gridcolor='lightgray',
            )
            fig.update_xaxes(
                title="Date",
                showline=True, 
                linewidth=1, 
                linecolor='black',
            )
            fig.update_layout(
                legend=dict(
                    bgcolor="rgba(255, 255, 255, 0)",
                    bordercolor="rgba(255, 255, 255, 0)",
                    x=1.1,
                    y=1,
                    xanchor='left',
                    yanchor='top',
                ),
                yaxis2={
                    "title": f"Average {class_type.capitalize()}", 
                    "tickformat": ",.0%"
                },
                height=height,
                #title=f"Distribution of {feature.lower()} over time",
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=showlegend,
                hoverlabel=DEFAULT_HOVER_LABEL,
            )
        return fig
    else:
        # Calculate the count and percentage for each category of the feature
        df = (
            df.groupby("index")[feature]
            .apply(lambda x: x.value_counts())
            .reset_index()
            .rename({"level_1": feature, feature: "count"}, axis=1)
        )
        df["total"] = df.groupby("index")["count"].transform("sum")
        df["percentage"] = df["count"] / df["total"]

        fig.update_layout(barmode="relative")
        for i, trace in enumerate(df[feature].unique()):
            dft = df[df[feature] == trace]
            fig.add_traces(
                go.Bar(
                    x=dft["index"],
                    y=dft["percentage"],
                    name=trace,
                    #marker_color=marker_color[trace],
                    opacity=0.5,
                )
            )

        fig.update_yaxes(
            title=f"{feature} (Percentage)",
            showline=True, 
            linewidth=1, 
            linecolor='black',
            showgrid=True, 
            gridwidth=1, 
            gridcolor='lightgray',
        )
        fig.update_xaxes(
            title="Date",
            showline=True, 
            linewidth=1, 
            linecolor='black',
        )
        fig.update_layout(
            legend=dict(
                bgcolor="rgba(255, 255, 255, 0)",
                bordercolor="rgba(255, 255, 255, 0)",
                x=1.1,
                y=1,
                xanchor='left',
                yanchor='top',
            ),
            yaxis2={
                "title": f"Average {class_type.capitalize()}", 
                "tickformat": ",.0%"
            },
            height=height,
            #title=f"Distribution of {feature.lower()} over time",
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=showlegend,
            hoverlabel=DEFAULT_HOVER_LABEL,
        )
        return fig