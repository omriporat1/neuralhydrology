import os
import pandas as pd
import plotly.express as px
from tqdm import tqdm


# === CONFIGURATION ===
combined_folder = r'C:\PhD\Data\IHS_IMS_wide\IHS_IMS_wide_unified_2025-04-08'  # folder with *_combined.csv files
output_html = "station_gantt_discontinuous.html"
assessment_file = r'C:\PhD\Data\IHS_IMS_wide\Stations_With_3_Gauges_Wide_according_to_assessment2.xlsx'



# === LOAD ASSESSMENT DATA ===
assessment_df = pd.read_excel(assessment_file)
assessment_df["station_label"] = assessment_df["station_id"] + " - " + assessment_df["station_name"]
assessment_lookup = assessment_df.set_index("station_id")[["station_label", "assessment"]]

# === COLLECT GANTT SEGMENTS FOR EACH STATION ===
records = []
all_files = [f for f in os.listdir(combined_folder) if f.endswith("_combined.csv")]

print(f"Processing {len(all_files)} station files...")

for file in tqdm(all_files, desc="Stations processed"):
    station_id = "il_" + file.split('_')[0]
    file_path = os.path.join(combined_folder, file)

    try:
        df = pd.read_csv(file_path, parse_dates=["datetime"])
    except Exception as e:
        print(f"Failed to read {file}: {e}")
        continue

    if df.empty or station_id not in assessment_lookup.index:
        continue

    df = df.sort_values("datetime")
    df["time_diff"] = df["datetime"].diff().dt.total_seconds().fillna(600)
    df["segment"] = (df["time_diff"] != 600).cumsum()

    segment_count = 0
    for seg_id, seg_df in df.groupby("segment"):
        if len(seg_df) >= 36:
            label = assessment_lookup.loc[station_id, "station_label"]
            assessment = assessment_lookup.loc[station_id, "assessment"]
            records.append({
                "Station": label,
                "Start": seg_df["datetime"].iloc[0],
                "Finish": seg_df["datetime"].iloc[-1],
                "Assessment": assessment
            })
            segment_count += 1

    print(f"{station_id}: {segment_count} valid segments")

# === CREATE GANTT CHART ===
gantt_df = pd.DataFrame(records)

if gantt_df.empty:
    print("No valid segments found.")
else:
    print(f"Found {len(gantt_df)} total valid segments.")

    # Color map for assessments
    color_map = {
        0: "green",
        1: "orange",
        2: "red",
        3: "gray"
    }

    # Create Gantt chart
    fig = px.timeline(
        gantt_df,
        x_start="Start",
        x_end="Finish",
        y="Station",
        color="Assessment",
        color_discrete_map=color_map
    )

    fig.update_layout(
        title="Valid Continuous Data Segments ≥6h per Station (Exact 10-min spacing)",
        showlegend=True,
        height=max(400, 20 * gantt_df['Station'].nunique())  # dynamic height
    )
    fig.update_yaxes(autorange="reversed")
    fig.write_html(output_html)
    fig.show()


