import plotly.graph_objs as go
import plotly.subplots as sp
import random
import time

# Create a figure
fig = sp.make_subplots(rows=1, cols=1)

# Create an initial scatter plot
x_data = []
y_data = []

scatter = go.Scatter(x=x_data, y=y_data, mode='lines+markers', name='Dynamic Data')
fig.add_trace(scatter)

# Set the layout
fig.update_layout(title='Dynamic Chart Example', xaxis=dict(title='X-axis'), yaxis=dict(title='Y-axis'))

# Create a function to update the data
def update_data(x_data, y_data):
    x_data.append(len(x_data))
    y_data.append(random.randint(0, 100))

# Start the while loop to continuously update the chart
while True:
    update_data(x_data, y_data)

    # Update the scatter plot trace with the new data
    fig.data[0].x = x_data
    fig.data[0].y = y_data

    # Update the layout (optional)
    fig.update_layout(title='Dynamic Chart Example', xaxis=dict(title='X-axis'), yaxis=dict(title='Y-axis'))

    # Update the chart
    fig.update()

    # Add a slight delay to control the update rate
    time.sleep(1)
