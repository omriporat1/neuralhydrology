import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# Load data
stations_df = pd.read_csv(r"C:\PhD\Data\IHS\stations_hydrographs\stations_time_periods.csv")
gauges_df = pd.read_csv(r"C:\PhD\Data\IMS\Data_by_station\Data_by_station_orig\gauge_time_periods.csv")
stations_gauges_df = pd.read_csv(r"C:\PhD\Data\IHS_IMS_wide\Stations_With_Gauges_Wide_Combined.csv")

# Convert dates to datetime format
stations_df['Start_Date'] = pd.to_datetime(stations_df['Start_Date'], utc=True)
stations_df['End_Date'] = pd.to_datetime(stations_df['End_Date'], utc=True)
gauges_df['Start_Date'] = pd.to_datetime(gauges_df['Start_Date'], utc=True)
gauges_df['End_Date'] = pd.to_datetime(gauges_df['End_Date'], utc=True)

# Mapping station ID to station names
station_id_to_name = dict(zip(stations_df['Station_ID'], stations_df['Station_name']))


def get_gantt_data(station_id):
    """Retrieve data for the selected hydrometric station and its associated rain gauges, ordered by distance."""
    station_data = stations_df[stations_df['Station_ID'] == station_id]
    station_name = station_id_to_name.get(station_id, 'Unknown')

    # Fetch the row for the given station ID
    station_row = stations_gauges_df[stations_gauges_df['station_id'] == f'il_{station_id}']

    if station_row.empty:
        return []

    # Extract relevant gauge IDs and distances
    gauge_info = []
    for i in range(1, 6):
        gauge_id_col = f'gauge_id_{i}'
        gauge_name_col = f'gauge_name_{i}'
        distance_col = f'distance_{i}'

        if not pd.isna(station_row[gauge_id_col].values[0]):
            gauge_info.append({
                'Gauge_ID': station_row[gauge_id_col].values[0],
                'Gauge_Name': station_row[gauge_name_col].values[0],
                'Distance': station_row[distance_col].values[0]
            })

    # Sort gauges by distance (ascending)
    gauge_info = sorted(gauge_info, key=lambda x: x['Distance'])

    # Filter gauge records
    gauge_ids = [g['Gauge_ID'] for g in gauge_info]
    gauge_records = gauges_df[gauges_df['Gauge_ID'].isin(gauge_ids)]

    # Construct Gantt data
    gantt_data = []

    # Add hydrometric station entry (always first)
    gantt_data.append({
        'Task': station_name,
        'Start': station_data['Start_Date'].values[0],
        'Finish': station_data['End_Date'].values[0],
        'Type': 'Hydrometric Station',
        'Order': 0
    })

    # Add rain gauges in sorted order
    for index, gauge in enumerate(gauge_info, start=1):
        gauge_row = gauge_records[gauge_records['Gauge_ID'] == gauge['Gauge_ID']]
        if not gauge_row.empty:
            gantt_data.append({
                'Task': f"{gauge['Gauge_Name']} - {gauge['Distance'] / 1000:.1f} km",
                'Start': gauge_row['Start_Date'].values[0],
                'Finish': gauge_row['End_Date'].values[0],
                'Type': 'Rain Gauge',
                'Order': index
            })

    return gantt_data


# Initialize Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Hydrometric Station and Rain Gauge Record Periods"),
    dcc.Dropdown(
        id='station-dropdown',
        options=[{'label': name, 'value': station_id} for station_id, name in station_id_to_name.items()],
        value=stations_df['Station_ID'].iloc[0],
        clearable=False
    ),
    dcc.Graph(id='gantt-chart')
])


@app.callback(
    Output('gantt-chart', 'figure'),
    [Input('station-dropdown', 'value')]
)
def update_gantt_chart(station_id):
    data = get_gantt_data(station_id)

    if not data:
        return go.Figure()

    df = pd.DataFrame(data)

    fig = px.timeline(df, x_start='Start', x_end='Finish', y='Task', color='Type',
                      title=f'Record Periods for {station_id_to_name.get(station_id, "Unknown")}',
                      color_discrete_map={'Hydrometric Station': 'blue', 'Rain Gauge': 'green'})

    fig.update_yaxes(categoryorder='array',
                     categoryarray=[row['Task'] for row in sorted(data, key=lambda x: x['Order'])])
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
