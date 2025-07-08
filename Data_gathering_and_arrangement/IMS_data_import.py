import requests
import json
from pyproj import Transformer
import csv
def main():

    csv_location = 'C:/PhD/Data/IMS/Data_by_station/'
    csv_filename = 'available_stations.csv'
    full_path = csv_location + csv_filename

    url = "https://api.ims.gov.il/v1/Envista/stations"

    headers = {
        'Authorization': 'ApiToken 6d2dd889-3fcf-4987-986f-e4679d4b2400'
    }

    response = requests.request("Get", url, headers=headers)
    stations_data = json.loads(response.text.encode('utf8'))
    transformer = Transformer.from_crs("epsg:4326", "epsg:2039", always_xy=True)

    station_list = []

    for current_station in stations_data:
        is_station_active = current_station["active"]
        is_has_location = isinstance(current_station["location"]["longitude"], (int, float))
        is_has_rain = any("Rain" in monitor.get("name", "") for monitor in current_station.get("monitors", []))
        is_has_time = any("Time" in monitor.get("name", "") for monitor in current_station.get("monitors", []))
        if is_station_active and is_has_location and is_has_rain and is_has_time:

            itm_x, itm_y = transformer.transform(current_station["location"]["longitude"],
                                                 current_station["location"]["latitude"])
            row_data = {
                "Station_ID": current_station["stationId"],
                "Station_name": current_station["name"],
                "ITM_X": int(itm_x),
                "ITM_Y": int(itm_y),
            }
            for monitor in current_station["monitors"]:
                if monitor["name"] == "Rain":
                    row_data["Rain_channelID"] = monitor["channelId"]
                elif monitor["name"] == "Time":
                    row_data["Time_channelID"] = monitor["channelId"]
            # print(row_data)
            station_list.append(row_data)

    # print(station_list)
    with open(full_path, mode="w", newline="", encoding="utf-8") as csv_file:
        field_names = station_list[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(station_list)

    print(f"CSV file '{full_path}' created successfully.")


if __name__ == '__main__':
    main()

