'''
from collections import defaultdict
import os
import pandas as pd

folder = r"C:\PhD\Data\Caravan\attributes\il"


column_usage = defaultdict(list)

csv_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]


for path in csv_files:
    df = pd.read_csv(path)
    for col in df.columns:
        if col != 'gauge_id':  # ignore gauge_id, as it is expected
            column_usage[col].append(os.path.basename(path))

# Show repeated column names
repeated = {k: v for k, v in column_usage.items() if len(v) > 1}
if repeated:
    print("Columns appearing in more than one file:")
    for col, files in repeated.items():
        print(f"- {col} appears in: {', '.join(files)}")
else:
    print("All columns (except 'gauge_id') are uniquely assigned to a single file.")


all_ids = pd.Series(dtype=str)

for path in csv_files:
    df = pd.read_csv(path)
    if 'gauge_id' in df.columns:
        current_ids = df['gauge_id']
        duplicates = current_ids[current_ids.duplicated()]
        if not duplicates.empty:
            print(f"Duplicate gauge_ids within {os.path.basename(path)}: {duplicates.tolist()}")
        all_ids = pd.concat([all_ids, current_ids])

# Check cross-file duplicates
cross_duplicates = all_ids[all_ids.duplicated()]
if not cross_duplicates.empty:
    print(f"Gauge_ids appearing more than once across files: {cross_duplicates.unique().tolist()}")


for path in csv_files:
    df = pd.read_csv(path)
    if 'gauge_id' in df.columns:
        counts = df['gauge_id'].value_counts()
        dups = counts[counts > 1]
        if not dups.empty:
            print(f"{os.path.basename(path)} has duplicate rows for:")
            print(dups)
'''

import os
import pandas as pd

# Folder containing your static attribute CSV files
input_folder = r"C:\PhD\Data\Caravan\attributes\il"
output_path = os.path.join(input_folder, "static_attributes_merged.csv")

# Collect all CSV files in the folder
csv_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".csv")]

merged_df = None
used_columns = set()
first_merge = True

for path in csv_files:
    df = pd.read_csv(path)
    fname = os.path.basename(path)

    if 'gauge_id' not in df.columns:
        print(f"Skipping {fname}: missing 'gauge_id' column.")
        continue

    # Check for duplicate gauge_ids
    duplicate_ids = df['gauge_id'][df['gauge_id'].duplicated()]
    if not duplicate_ids.empty:
        print(f"Warning: {fname} contains duplicate gauge_ids:")
        print(duplicate_ids.unique().tolist())

    # Ensure unique column names (except 'gauge_id')
    new_columns = []
    for col in df.columns:
        if col == 'gauge_id':
            new_columns.append(col)
        elif col in used_columns:
            # Rename to avoid collision
            base_name = os.path.splitext(fname)[0]
            new_name = f"{col}_{base_name}"
            print(f"Renaming duplicate column '{col}' in {fname} to '{new_name}'")
            new_columns.append(new_name)
        else:
            new_columns.append(col)
            used_columns.add(col)

    df.columns = new_columns

    # Merge with existing DataFrame
    if first_merge:
        merged_df = df
        first_merge = False
    else:
        merged_df = pd.merge(merged_df, df, on='gauge_id', how='inner', validate="one_to_one")

# Final checks
if merged_df is not None:
    if merged_df.columns.duplicated().any():
        print("Error: Duplicate column names exist after merge.")
    elif merged_df['gauge_id'].duplicated().any():
        print("Error: Duplicate gauge_ids exist after merge.")
    else:
        merged_df.to_csv(output_path, index=False)
        print(f"\nStatic attributes merged successfully (inner join) → {output_path}")
        print(f"Total basins retained after merge: {len(merged_df)}")
else:
    print("No valid files were merged.")
