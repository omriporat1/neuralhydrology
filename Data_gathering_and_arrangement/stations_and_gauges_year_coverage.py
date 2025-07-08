import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pytz  # Ensure timezone support

def create_gauge_time_periods(gauges_file, data_folder):
    gauges_df = pd.read_csv(gauges_file)

    # Prepare output DataFrame
    summary = []

    # Iterate through station files
    for gauge_id in gauges_df["Station_ID"]:
        file_path = os.path.join(data_folder, f"{gauge_id}.csv")

        if os.path.exists(file_path):
            # Load gauge data
            df = pd.read_csv(file_path, parse_dates=["datetime"])

            # Extract start and end date
            start_date = df["datetime"].min()
            end_date = df["datetime"].max()

            # Append to summary list
            summary.append({"Station_ID": gauge_id, "Start_Date": start_date, "End_Date": end_date})
            print("Processed station", gauge_id)

    # Create DataFrame
    summary_df = pd.DataFrame(summary)

    # Merge with station names
    summary_df = summary_df.merge(gauges_df, on="Station_ID", how="left")

    # Save summary
    summary_df.to_csv(data_folder + r"\gauge_time_periods.csv", index=False)

    return summary_df


def create_stations_time_periods(stations_file, data_folder):
    stations_df = pd.read_csv(stations_file)

    # Prepare output DataFrame
    summary = []
    tz_info = pytz.timezone("Asia/Jerusalem")  # Replace with your timezone

    # Iterate through station files
    for station_id in stations_df["gauge_id"]:
        # remove the 'il' prefix:
        station_id = station_id[3:]

        file_path = os.path.join(data_folder, f"{station_id}.csv")

        if os.path.exists(file_path):
            # Load gauge data
            df = pd.read_csv(file_path, parse_dates=["Flow_sampling_time"])

            df["Flow_sampling_time"] = pd.to_datetime(df["Flow_sampling_time"], format='%Y-%m-%dT%H:%M:%S%z')  # convert date column to datetime
            df["Flow_sampling_time"] = df["Flow_sampling_time"].dt.tz_localize("UTC").dt.tz_convert(tz_info)  # Convert to local timezone


            # Extract start and end date
            start_date = df["Flow_sampling_time"].min().strftime("%Y-%m-%d %H:%M:%S%z")
            end_date = df["Flow_sampling_time"].max().strftime("%Y-%m-%d %H:%M:%S%z")

            # Append to summary list
            summary.append({"Station_ID": station_id, "Start_Date": start_date, "End_Date": end_date})
            print("Processed station", station_id)

    # Create DataFrame
    summary_df = pd.DataFrame(summary)

    stations_df.rename(columns={"gauge_id": "Station_ID", "gauge_lat": "Station_lat", "gauge_lon": "Station_lon", "gauge_name": "Station_name"}, inplace=True)
    stations_df["Station_ID"] = stations_df["Station_ID"].str.replace("il_", "", regex=False)

    # Merge with station names
    summary_df = summary_df.merge(stations_df, on="Station_ID", how="left")

    # Save summary
    summary_df.to_csv(data_folder + r"\stations_time_periods.csv", index=False)

    return summary_df


def plot_gantt_chart(df, title):
    fig = px.timeline(
        df,
        x_start="Start_Date",
        x_end="End_Date",
        y="Station_name",  # Use Station_name for better readability
        title=title,
        labels={"Station_name": "Station", "Start_Date": "Start Date", "End_Date": "End Date"},
    )

    fig.update_yaxes(categoryorder="category descending")  # Sort stations by name
    fig.update_layout(xaxis_title="Time Period", yaxis_title="Station / gauge")

    # Show the interactive chart
    fig.show()


def create_linked_gantt_chart(station_file, gauge_file, linkage_file, output_folder):
    """
    Creates a Gantt chart showing data coverage for hydrometric stations (green) and linked rain gauges (blue),
    with a dropdown filter for selecting specific stations.
    """

    # Load the time period data
    stations_df = pd.read_csv(os.path.join(output_folder, station_file))
    gauges_df = pd.read_csv(os.path.join(output_folder, gauge_file))
    linkage_df = pd.read_csv(linkage_file)

    # Ensure station names exist
    # if "station_name" not in stations_df.columns:
    #     raise KeyError("Column 'station_name' not found in stations_df")

    # Reshape linkage table (wide to long format for gauge stations)
    melted_df = linkage_df.melt(
        id_vars=["station_id", "station_name"],
        value_vars=["gauge_id_1", "gauge_id_2", "gauge_id_3", "gauge_id_4", "gauge_id_5"],
        var_name="Gauge_Index",
        value_name="Gauge_ID"
    ).dropna()

    # Merge station time periods
    stations_df["Type"] = "Hydrometric Station"
    gauges_df["Type"] = "Rain Gauge"

    merged_gauges = melted_df.merge(gauges_df, left_on="Gauge_ID", right_on="Station_ID", how="left")
    merged_stations = stations_df[["station_name", "Start_Date", "End_Date", "Type"]]

    # Combine both dataframes
    final_df = pd.concat([merged_stations, merged_gauges[["station_name", "Start_Date", "End_Date", "Type"]]])

    # Convert datetime columns
    final_df["Start_Date"] = pd.to_datetime(final_df["Start_Date"], utc=True)
    final_df["End_Date"] = pd.to_datetime(final_df["End_Date"], utc=True)

    # Create an initial figure with all data
    fig = px.timeline(
        final_df,
        x_start="Start_Date",
        x_end="End_Date",
        y="station_name",
        title="Hydrometric Stations & Linked Rain Gauges Coverage",
        labels={"station_name": "Station", "Start_Date": "Start Date", "End_Date": "End Date"},
        color="Type",
        category_orders={"Type": ["Hydrometric Station", "Rain Gauge"]},
        color_discrete_map={"Hydrometric Station": "green", "Rain Gauge": "blue"}
    )

    fig.update_yaxes(categoryorder="category ascending")
    fig.update_layout(xaxis_title="Time Period", yaxis_title="Station / Gauge")

    # Create dropdown menu
    unique_stations = final_df["station_name"].unique()
    dropdown_buttons = []

    # 'All Stations' button
    dropdown_buttons.append(
        {
            "label": "All Stations",
            "method": "update",
            "args": [
                [{"visible": True} for _ in fig.data],
                {"title": "Hydrometric Stations & Linked Rain Gauges Coverage"}
            ]
        }
    )

    # Buttons for individual stations
    for station in unique_stations:
        linked_gauges = melted_df[melted_df["station_name"] == station]["Gauge_ID"].values
        visibility = [
            (trace.name == station or trace.name in linked_gauges) for trace in fig.data
        ]

        dropdown_buttons.append(
            {
                "label": station,
                "method": "update",
                "args": [
                    [{"visible": v} for v in visibility],
                    {"title": f"Coverage for {station}"}
                ]
            }
        )

    fig.update_layout(
        updatemenus=[
            {
                "buttons": dropdown_buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.15,
                "y": 1.15,
                "xanchor": "left",
                "yanchor": "top",
                "type": "dropdown"
            }
        ]
    )

    # Show the interactive chart
    fig.show()


def main():
    gauges_file = r"C:\PhD\Data\IMS\Data_by_station\available_stations.csv"
    gauges_data_folder = r"C:\PhD\Data\IMS\Data_by_station\Data_by_station_orig"
    gauges_time_file_exists = os.path.exists(gauges_data_folder + r"\gauge_time_periods.csv")

    if not gauges_time_file_exists:
        gauge_summary_df = create_gauge_time_periods(gauges_file, gauges_data_folder)
    else:
        gauge_summary_df = pd.read_csv(gauges_data_folder + r"\gauge_time_periods.csv")

    stations_file = r"C:\PhD\Data\Caravan\attributes\il\attributes_other_il.csv"
    station_data_folder = r"C:\PhD\Data\IHS\stations_hydrographs"
    stations_time_file_exists = os.path.exists(station_data_folder + r"\stations_time_periods.csv")

    if not stations_time_file_exists:
        station_summary_df = create_stations_time_periods(stations_file, station_data_folder)
    else:
        station_summary_df = pd.read_csv(station_data_folder + r"\stations_time_periods.csv")

    # Plotting
    # plot_gantt_chart(gauge_summary_df, "Gauges Time Periods")
    # plot_gantt_chart(station_summary_df, "Stations Time Periods")

    gauges_file = gauges_data_folder + r"\gauge_time_periods.csv"
    stations_file = station_data_folder + r"\stations_time_periods.csv"
    linkage_file = r"C:\PhD\Data\IHS_IMS_wide\Stations_With_Gauges_Wide_Combined.csv"
    output_folder = r"C:\PhD\Data\IHS_IMS_wide"

    create_linked_gantt_chart(stations_file, gauges_file, linkage_file, output_folder)


if __name__ == '__main__':
    main()
