import dash
from dash import html, dcc, Input, Output
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn2
import io
import base64

# Sample DF
df = pd.DataFrame({
    'A': [True, False, True, False, True],
    'B': [True, True, False, False, True],
    'C': [False, True, True, False, True],
    'D': [True, False, False, True, False]
})

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Interactive Venn Diagram for Boolean Columns"),
    dcc.Checklist(
        id='venn-columns',
        options=[{'label': col, 'value': col} for col in df.columns],
        value=['A', 'B', 'C'],
        labelStyle={'display': 'block'},
        inline=False
    ),
    html.Div(id='warning-msg', style={'color': 'red'}),
    html.Img(id='venn-image')
])

@app.callback(
    Output('venn-image', 'src'),
    Output('warning-msg', 'children'),
    Input('venn-columns', 'value')
)
def update_venn(cols):
    if len(cols) > 3:
        return None, "Please select exactly 3 columns."
    
    # Compute sets
    sets = []
    for col in cols:
        sets.append(set(df.index[df[col]]))

    # Generate Venn
    fig, ax = plt.subplots()
    if len(cols) ==3:
        venn3(subsets=sets, set_labels=cols)
    else:
        venn2(subsets=sets, set_labels=cols)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    encoded = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"data:image/png;base64,{encoded}", ""

if __name__ == '__main__':
    app.run(debug=True)
