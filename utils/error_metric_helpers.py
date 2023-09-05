import datarobot as dr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

    if project_type == "Regression":
        name_mappings = [
            "prediction",
            f"{target}_PREDICTION",
        ]
    else:
        name_mappings = {
            f"class_{positive_class}"
        }
    return name_mappings

def log_loss(act, pred, eps=1e-15):
    """
    Calculate the log loss for binary classification problems.

    Parameters
    ----------
    y_true : list or np.array
        True binary labels (0 or 1).
    y_pred : list or np.array
        Predicted probabilities for the positive class (1).
    eps : float, optional, default=1e-15
        Small constant for numerical stability.

    Returns
    -------
    loss : float
        Log loss value.
    """

    # Ensure the input lists/arrays are numpy arrays
    act = np.array(act)
    pred = np.array(pred)

    # Clip predicted probabilities for numerical stability
    pred = np.clip(pred, eps, 1 - eps)

    # Calculate log loss
    loss = -(act * np.log(pred) + (1 - act) * np.log(1 - pred)).mean()

    return loss

def mae(act, pred, weight=None):
    """
    MAE = Mean Absolute Error = mean( abs(act - pred) )
    """
    if len(pred.shape) > 1:
        if pred.shape[1] == 2:
            pred = pred[:, 1]
        else:
            pred = pred.ravel()

    pred = pred.astype(np.float64, copy=False)
    d = act - pred
    ad = np.abs(d)
    if weight is not None:
        if weight.sum() == 0:
            return 0
        ad = ad * weight / weight.mean()
    mae = ad.mean()

    if np.isnan(mae):
        return np.finfo(np.float64).max
    else:
        return mae


def mape(act, pred, nan='ignore'):

    # ignore NAN (drop rows), do nothing, replace Nan with 0
    if nan not in ['ignore', 'set_to_zero', 'error']:
        raise ValueError(f'{nan} must be either ignore, set_to_zero, or error')

    act, pred = np.array(act), np.array(pred)
    pred = pred.astype(np.float64, copy=False)
    n = np.abs(act - pred)
    d = act
    ape = n / d

    if nan == 'set_to_zero':
        ape[~np.isfinite(ape)] = 0
    elif nan == 'ignore':
        ape = ape[np.isfinite(ape)]

    smape = np.mean(ape)

    if np.isnan(smape):
        return np.finfo(np.float64).max

    return smape


def smape(act, pred):
    pred = pred.astype(np.float64, copy=False)
    n = np.abs(pred - act)
    d = (np.abs(pred) + np.abs(act)) / 2
    ape = n / d
    smape = np.mean(ape)

    if np.isnan(smape):
        return np.finfo(np.float64).max

    return smape


def rmse(act, pred, weight=None):
    """
    RMSE = Root Mean Squared Error = sqrt( mean( (act - pred)**2 ) )
    """
    if len(pred.shape) > 1:
        if pred.shape[1] == 2:
            pred = pred[:, 1]
        else:
            pred = pred.ravel()

    pred = pred.astype(np.float64, copy=False)
    d = act - pred
    sd = np.power(d, 2)
    if weight is not None:
        if weight.sum() == 0:
            return 0
        sd = sd * weight / weight.mean()
    mse = sd.mean()
    rmse = np.sqrt(mse)

    if np.isnan(rmse):
        return np.finfo(np.float64).max
    else:
        return rmse


def gamma_loss(act, pred, weight=None):
    """Gamma deviance"""
    eps = 0.001
    pred = np.maximum(pred, eps)  # ensure predictions are strictly positive
    act = np.maximum(act, eps)  # ensure actuals are strictly positive
    d = 2 * (-np.log(act / pred) + (act - pred) / pred)
    if weight is not None:
        d = d * weight / np.mean(weight)
    return np.mean(d)


def tweedie_loss(act, pred, weight=None, p=1.5):
    """tweedie deviance for p = 1.5 only"""

    if p <= 1 or p >= 2:
        raise ValueError('p equal to %s is not supported' % p)

    eps = 0.001
    pred = np.maximum(pred, eps)  # ensure predictions are strictly positive
    act = np.maximum(act, 0)  # ensure actuals are not negative
    d = (
        (act ** (2.0 - p)) / ((1 - p) * (2 - p))
        - (act * (pred ** (1 - p))) / (1 - p)
        + (pred ** (2 - p)) / (2 - p)
    )
    d = 2 * d
    if weight is not None:
        d = d * weight / np.mean(weight)
    return np.mean(d)


def poisson_loss(act, pred, weight=None):
    """
        Poisson Deviance = 2*(act*log(act/pred)-(act-pred))
        ONLY WORKS FOR POSITIVE RESPONSES
    """
    if len(pred.shape) > 1:
        pred = pred.ravel()
    pred = np.maximum(pred, 1e-8)  # ensure predictions are strictly positive
    act = np.maximum(act, 0)  # ensure actuals are non-negative
    d = np.zeros(len(act))
    d[act == 0] = pred[act == 0]
    cond = act > 0
    d[cond] = act[cond] * np.log(act[cond] / pred[cond]) - (act[cond] - pred[cond])
    d = d * 2
    if weight is not None:
        if weight.sum() == 0:
            return 0
        d = d * weight / weight.mean()
    return d.mean()

def plot_error(
    df: pd.DataFrame,
    project_id: str,
    feature: str,
    metric: callable,
    bins: int=10,
    showlegend: bool=False,
    height: int=500,
):
    """
    Plot a histogram of a feature with average predictions and actuals as overlayed scatter plots.
    
    Args:
        df (pd.DataFrame): Input dataframe containing the data to be plotted.
        project_id (str): DataRobot project ID.
        feature (str): Feature column name to be plotted.
        metric (callable): Error metric displayed in plot.
            Valid arguments are rmse, mae, log_loss, gamma_loss, poisson_loss, tweedie_loss, mape, and smape 
        bins (int, optional): Number of bins for the bar plot. Defaults to 10.
        showlegend (bool, optional): Whether to show the legend in the plot. Defaults to False.
        height (int, optional): Height of the plot. Defaults to 500.
        
    Returns:
        fig (plotly.graph_objects.Figure): A Plotly figure of the histogram with overlayed scatter plots.
    """
    # supported_metrics = {
    #     "rmse":rmse, 
    #     "mae":mae, 
    #     "log_loss":log_loss, 
    #     "gamma_loss":gamma_loss, 
    #     "poisson_loss":poisson_loss, 
    #     "tweedie_loss":tweedie_loss, 
    #     "mape":mape, 
    #     "smape":smape
    # }
    # assert metric in supported_metrics.values(), "Select a valid metric"

    # Pull DataRobot project
    project = dr.Project.get(project_id)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    target = project.target
    
    # Calculate average predictions per time period
    col = list(get_column_name_mappings(project_id))[0]
    
    bins = min(bins, len(df[feature].unique()))
        
    # Create bins for numerical features
    if np.issubdtype(df[feature].dtype, np.number):
        df["bins"] = pd.cut(df[feature], bins, duplicates='drop')
        df["bins"] = df["bins"].apply(lambda x: f"({x.left}, {x.right}]")
    # Create bins for categorical features
    else:
        unique_values = len(df[feature].unique())
        top_bins = df[feature].value_counts()[0:bins-1].index.values
        other_bin = df.loc[~df[feature].isin(top_bins), feature].unique()
        df["bins"] = ['OTHER' if i in other_bin else i for i in df[feature]]

    # Aggregate data and calculate mean and count for each bin
    df1 = (
        df
        .groupby(
            ["bins"]
        )[[feature, col,target]]
        .agg(['mean','count'])
        .reset_index()
    )

    df1.columns = df1.columns.map('_'.join)
    df1.rename({
        "bins_":"bins",
        f"{col}_mean":"Average Prediction",
        f"{target}_mean":"Average Actual",
    }, axis=1, inplace=True)

    df2 = (
        df
        .groupby(["bins"])[[feature, col,target]]
        .apply(lambda x: metric(x[target], x[col]))
        .reset_index()
        .rename({0:metric.__name__}, axis=1)
    )

    fig.add_trace(
        go.Bar(
            x=df1["bins"],
            y=df1[f"{target}_count"],
            name="Count",
            marker_color="lightblue",
            marker = dict(
                line=dict(color='rgba(0,0,0)', width=0.5)
            ),
            opacity=0.5,
        )
    )

    # Add the selected error metric as a scatter plot
    for trace, color, symbol in zip(
        [metric.__name__],
        ["black"],
        ["cross-open"],
    ):
        fig.add_trace(
            go.Scatter(
                x=df2["bins"],
                y=df2[trace],
                mode="lines+markers",
                name=trace,
                marker=dict(
                    size=5,
                    color=color,
                    symbol=symbol,
                    line=dict(
                        color=color,
                        width=2,
                    ),
                ),
                line=dict(
                    color=color,
                    width=2,
                ),
            ),
            secondary_y=True,
        )
    # Update y-axis and x-axis properties
    fig.update_yaxes(
        title="Frequency",
        showline=True, 
        linewidth=1, 
        linecolor='black',
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightgray',
    )
    fig.update_xaxes(
        title=feature,
        showline=True, 
        linewidth=1, 
        linecolor='black',
        tickwidth=1, 
        tickcolor='black', 
        ticklen=6,
    )
    fig.update_layout( 
        yaxis2={
            "title": f"{metric.__name__}",
            #"tickformat": ",.0%"
        },
        height=height,
        hoverlabel=DEFAULT_HOVER_LABEL,
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=showlegend,
        #legend_title=f"{class_type.capitalize()}",
        #font_family="Arial",
        font_color="black",
    )
    
    return fig