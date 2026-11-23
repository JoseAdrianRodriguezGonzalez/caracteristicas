import numpy as np
import plotly.graph_objects as go
import pandas as pd
import sys
import os
url='./captured_data/'+sys.argv[1]

archivos=os.listdir(url)
csv_files= list(filter(lambda f: f.endswith("csv"),archivos))

for f in csv_files:
    print(f)
    df = pd.read_csv(os.path.join(url, f), header=None)
    df.columns=["x","y","z"]
    fig= go.Figure(
             data=[
                 go.Scatter3d(
                      x=df["x"],y=df["y"],z=df["z"],
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=df["z"],
                        colorscale="Viridis",
                        opacity=0.8
                    )
                )
            ]
         )
    fig.update_layout(
        scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
                ),
        margin=dict(l=0,r=0,b=0,t=0)
     )
    fig.show(renderer="browser")
