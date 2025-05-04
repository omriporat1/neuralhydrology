import csv
import pandas as pd

def extract_unique_name_shd_pairs(file_path):
    unique_pairs = set()
    encodings_to_try = ['utf-8-sig', 'cp1255', 'iso-8859-8']

    for enc in encodings_to_try:
        try:
            with open(file_path, mode='r', newline='', encoding=enc) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    name = row.get("name")
                    shd_id = row.get("shd_id")
                    if name is not None and shd_id is not None:
                        unique_pairs.add((name, shd_id))
            break  # stop after successful read
        except UnicodeDecodeError as e:
            print(f"Encoding '{enc}' failed: {e}")
        except Exception as e:
            print(f"Unexpected error with encoding '{enc}': {e}")

    if not unique_pairs:
        raise ValueError("Failed to decode file with all tried encodings.")

    df = pd.DataFrame(list(unique_pairs), columns=["name", "shd_id"])
    return df

def export_to_csv(df, output_path):
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"CSV exported successfully to: {output_path}")


# Example usage
file_path = r"S:\hydrolab\home\Omri_Porat\PhD\Data\IHS_calibration_DB_from_Yoni\DataFromYoni05022024\Lachish - Sorek\Lachish - Sorek_ts.csv"  # Replace with your file path
output_csv = r'S:\hydrolab\home\Omri_Porat\PhD\Data\IHS_calibration_DB_from_Yoni\DataFromYoni05022024\Lachish - Sorek\unique_name_shd_id_pairs.csv'  # Output CSV path

df_unique = extract_unique_name_shd_pairs(file_path)
export_to_csv(df_unique, output_csv)

