import requests
import json
import pandas as pd

all_stations_path = 'C:/PhD/Data/IMS/Data_by_station/available_stations.csv'
export_folder = 'C:/PhD/Data/IMS/Data_by_station/Data_by_station_orig/'

headers = {
    'Authorization': 'ApiToken 6d2dd889-3fcf-4987-986f-e4679d4b2400'
}

begin_time = '1999/10/01'  # Beginning of hydrological year 1999/2000
end_time = '2023/09/30'  # End of hydrological year 2022/2023

date_range = 'from=' + begin_time + '&to=' + end_time
# begin_time = datetime.datetime(2010, 10, 1, 00, 00, 00)
# end_time = datetime.datetime(2021, 9, 30, 23, 59, 59)

all_stations_df = pd.read_csv(all_stations_path)

for _, station_row in all_stations_df.iterrows():
    ID = str(station_row["Station_ID"])
    channel = str(station_row["Rain_channelID"])
    # url = "https://api.ims.gov.il/v1/Envista/stations/" + ID + '/data/' + channel + '/data' + date_range
    url = "https://api.ims.gov.il/v1/Envista/stations/" + ID + '/data/' + channel + '/?' + date_range
    response = requests.request("get", url, headers=headers)

    if response.text.strip():  # Check if the response content is not empty
        single_station_data = json.loads(response.text.encode('utf8'))
        filtered_station_data = [{"datetime": entry["datetime"], "Rain": entry["channels"][0]["value"]} for entry in
                                 single_station_data["data"]]
        df_filtered_station = pd.DataFrame(filtered_station_data)
        csv_filtered_export_destination = export_folder + ID + '.csv'
        df_filtered_station.to_csv(csv_filtered_export_destination, index=False)
        print("Station " + ID + " data saved to " + csv_filtered_export_destination)
    else:
        print("Empty response from the API.")


