from typing import Tuple
import numpy as np
import pandas as pd
from matplotlib import colors
import matplotlib.pyplot as plt
import pandas.io.formats.style
import re


def scale_between_minus_one_and_one(df, column_name):
    # Check if column_name exists in the dataframe
    if column_name in df.columns:
        # Calculate the min and max values
        min_val = df[column_name].min()
        max_val = df[column_name].max()
        
        # Perform the scaling
        df[column_name] = 2 * ((df[column_name] - min_val) / (max_val - min_val)) - 1
    else:
        print(f"Column '{column_name}' does not exist in the dataframe.")
    return df


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

    rows_to_keep = df['row_id'].sample(sample)

    for col in table_data.columns:
        if col not in bin_pivot.columns:
            bin_pivot[col] = 0
    
    txt = [c for c in df.columns if re.search('class_|prediction', c) is not None][0]
    cols_to_drop = ['explanation_number','actual_value','qualitative_strength',txt]
    bin_pivot = bin_pivot.loc[bin_pivot.index.isin(rows_to_keep), table_data.columns].drop(cols_to_drop, axis=1)
    #bin_pivot = bin_pivot[table_data.columns].drop(cols_to_drop, axis=1)

    gmap = np.array([bin_pivot.iloc[0:sample][i] for i in bin_pivot.iloc[0:sample]]).T

    # return table_data.iloc[0:sample], gmap
    return table_data.loc[table_data.index.isin(rows_to_keep), :], gmap


def plot_overlaid_prediction_explanations(
    df: pd.DataFrame, 
    sample: int = 50,
    scale: bool = True,
    vmin: float = None,
    vmax: float = None,
) -> pd.io.formats.style.Styler:
    """
    Plot the overlaid prediction explanations using a DataFrame with colors.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        sample (int, optional): Number of samples to include in the plot. Defaults to 500.
        
    Returns:
        pd.io.formats.style.Styler: The DataFrame Styler object with the overlaid prediction explanations.
    """
    if scale:
        df = scale_between_minus_one_and_one(df, 'strength')

    table_data, gmap = prep_data_table(
        df,
        sample=min(df.shape[0], sample),
    )
    table_data = table_data.drop(['explanation_number','actual_value','qualitative_strength'], axis=1)
    txt = [c for c in table_data.columns if re.search('class_|prediction', c) is not None][0]
    subset = list(set(table_data.columns) - set([txt]))

    if vmin is None:
        vmin = df['strength'].min()
    if vmax is None:
        vmax = df['strength'].max() 

    # print('vmin: ',vmin, '\n', 'vmax: ',vmax)
    df_with_colors = (
        table_data.style.background_gradient(
            cmap="seismic", axis=None, gmap=gmap, subset=subset, low=0.5, high=0.5, vmin=vmin, vmax=vmax,
        )
    )

    return df_with_colors