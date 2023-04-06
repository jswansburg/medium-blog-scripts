from typing import Tuple
import numpy as np
import pandas as pd
from matplotlib import colors
import matplotlib.pyplot as plt
import pandas.io.formats.style


def pivot_melted_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the melted DataFrame to convert it into a wide format.
    
    Args:
        df (pd.DataFrame): The input melted DataFrame.
        
    Returns:
        pd.DataFrame: The pivoted DataFrame with row_id as index, feature_name as columns, and strength as values.
    """
    return (
        df[["row_id", "feature_name", "strength"]]
        .pivot(index="row_id", columns="feature_name", values="strength")
        .fillna(0)
    )


def extract_colors(
    df: pd.DataFrame, 
    column: str, 
    cmap: str = "seismic"
) -> list:
    """
    Extract colors for the input DataFrame's specified column.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column for which to extract colors.
        cmap (str, optional): The colormap to use. Defaults to "seismic".
        
    Returns:
        list: List of colors for the specified column.
    """
    rng = df[column].max() - df[column].min()
    if rng == 0:
        return ["#FFFFFF" for _ in df[column].values]
    else:
        norm = colors.Normalize(-1 - (rng * 0), 1 + (rng * 0))
        normed = norm(df[column].values)
        return [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]


def prep_data_table(
    df: pd.DataFrame, 
    sample: int = 500
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Prepare the data table for plotting.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        sample (int, optional): Number of samples to include in the output. Defaults to 500.
        
    Returns:
        Tuple[pd.DataFrame, np.ndarray]: A tuple containing the prepared DataFrame and the gradient map array.
    """
    table_data = (
        df.drop_duplicates(["row_id"])
        .drop(columns=["feature_name", "strength"], axis=1)
        .sort_values(by="row_id")
        .reset_index(drop=True)
        .set_index("row_id")
    )
    bin_pivot = (
        pivot_melted_df(df)
        .reset_index()
        .sort_values(by="row_id")
        .reset_index(drop=True)
        .set_index("row_id")
    )

    for col in table_data.columns:
        if col not in bin_pivot.columns:
            bin_pivot[col] = 0
    bin_pivot = bin_pivot[table_data.columns]

    gmap = np.array([bin_pivot.iloc[0:sample][i] for i in bin_pivot.iloc[0:sample]]).T

    return table_data.iloc[0:sample], gmap


def plot_overlaid_prediction_explanations(
    df: pd.DataFrame, 
    sample: int = 50,
) -> pd.io.formats.style.Styler:
    """
    Plot the overlaid prediction explanations using a DataFrame with colors.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        sample (int, optional): Number of samples to include in the plot. Defaults to 500.
        
    Returns:
        pd.io.formats.style.Styler: The DataFrame Styler object with the overlaid prediction explanations.
    """
    table_data, gmap = prep_data_table(
        df,
        sample=min(df.shape[0], sample),
    )

    df_with_colors = (
        table_data.style.background_gradient(
            cmap="seismic", axis=None, vmin=-1, vmax=1, gmap=gmap
        )
    )

    return df_with_colors