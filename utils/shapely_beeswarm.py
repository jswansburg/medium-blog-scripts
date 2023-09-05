import pandas as pd
import datarobot as dr
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

DEFAULT_HOVER_LABEL = dict(
    bgcolor="white", font_size=16, font_family="Rockwell", namelength=-1
)

def plot_shapely_beeswarm(df, sample=None):

    features = df['feature_name'].unique()[0:5]
    fig = make_subplots(rows=len(features), cols=1, vertical_spacing=0.075)

    min_strength = df['strength'].min()
    max_strength = df['strength'].max()
    
    # Unique values sorted in the order you desire
    unique_vals = [1, 0]

    # Discrete colorscale
    colorscale = ["#FB0D0D","blue"]
    colordict = {val:color for val, color in zip(unique_vals, colorscale)}

    for row, feature in enumerate(features):
        scatterplot_data = df.loc[df['feature_name']==feature, ['row_id','actual_value','strength']]
        scatterplot_data['actual_value'] = scatterplot_data['actual_value'].fillna('missing')
        scatterplot_data['y'] = np.random.normal(loc=0.5, scale=0.1, size=scatterplot_data.shape[0])
        
        if sample is not None:
            scatterplot_data['sample'] = np.where(scatterplot_data['row_id'].isin(sample), 1, 0)
            # Map the color_value column to the corresponding color
            scatterplot_data['color'] = scatterplot_data['sample'].map(colordict)
        else:
            marker_color = "blue"
        
        # Create subplot
        fig.append_trace(
            go.Scatter(
                x=scatterplot_data["strength"],
                y=scatterplot_data["y"],
                name=f"{feature}",
                mode="markers",
                marker=dict(
                    color=scatterplot_data['color'],
                    size=6,
                ),
                customdata=scatterplot_data[
                    ["row_id","actual_value","strength"]
                ],
                hovertemplate="<br>Row ID</b>: %{customdata[0]}"
                + "<br>Value</b>: %{customdata[1]}"
                + "<br>Strength</b>: %{customdata[2]: .2}"
                + "<extra></extra>",
                hoverlabel={
                    "align": "left",
                    "bgcolor": "white",
                    "font_size": 16,
                    "font_family": "Rockwell",
                },
            ),
            row=row+1,
            col=1,
        )
        fig.update_xaxes(
            title_text=f"{feature}", 
            range=[min_strength, max_strength], 
            row=row+1, 
            col=1,
            showline=True, 
            linewidth=1, 
            linecolor='black',
            ticks="inside", 
            tickwidth=1, 
            tickcolor='black', 
            ticklen=6,
            dtick = 0.25,
        )
        fig.update_yaxes(
            visible=False,
        )
        fig.update_layout(
            height=800,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            hoverlabel=DEFAULT_HOVER_LABEL
        )

    fig.show()
