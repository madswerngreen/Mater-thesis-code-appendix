from pathlib import Path
import zipfile
import json
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm

# -----------------------
# Configuration
# -----------------------

# Base folder containing daily subfolders of zipped JSON data
base_folder = Path('../Data/06')

# Output folder for truck-specific CSVs (one file per truck)
output_folder = Path("../Data/toll_data_truck")

# TruckID mapping file (maps licence plate → TruckID)
truck_file = Path("../Data/trucks.parquet")


# -----------------------
# Helper Functions
# -----------------------

def decode_plate(hex_str: str) -> str:
    """
    Decode a hex-encoded licence plate into a string.
    """
    return bytes.fromhex(hex_str).decode("latin-1")


# Load existing TruckID mapping if it exists
if truck_file.exists():
    # Resume from previous run
    truck_map = pd.read_parquet(truck_file)
    plate_to_id = dict(zip(truck_map['licencePlateNumber'], truck_map['TruckID']))
    next_truck_id = max(plate_to_id.values()) + 1
else:
    # Start fresh
    plate_to_id = {}
    next_truck_id = 1


def get_truck_id(plate: str) -> int:
    """
    Get or assign a unique TruckID for a given licence plate string.
    """
    global next_truck_id
    if plate not in plate_to_id:
        plate_to_id[plate] = next_truck_id
        next_truck_id += 1
    return plate_to_id[plate]


def get_data(data: dict) -> tuple[pd.DataFrame, int]:
    """
    Parse one JSON record into a DataFrame of raw measurements for a truck.

    Returns:
        df: DataFrame with columns ['time', 'latitude', 'longitude', 'speed', 'heading',
                                    'weightLimits', 'CO2EmissionClass', 'TruckID']
        truck_id: Assigned TruckID for this truck
    """
    # Navigate JSON structure
    adu = data['InfoExchange']['InfoExchangeContent']['adus']['tollDeclarationAdus']['tollDeclarationAdu']
    charge_report = adu['chargeReport']
    raw_usage_data = charge_report['usageStatementList']['usageStatement']['listOfRawUsageData']
    vehicle_desc = raw_usage_data['VehicleDescription']

    # Decode licence plate and get TruckID
    licencePlateNumber = decode_plate(charge_report['vehicleLPNr']['LPN']['licencePlateNumber'])
    truck_id = get_truck_id(licencePlateNumber)

    # Extract measured raw data
    mrd_list = [raw['MeasuredRawData'] for raw in raw_usage_data['rawDataList']]

    # Keep simple types (avoid dtype casting inside loop)
    df = pd.DataFrame({
        "TruckID": truck_id,
        "time": pd.to_datetime([mrd['timeWhenMeasured'] for mrd in mrd_list],
                               format="%Y%m%d%H%M%SZ", utc=True),
        "latitude": [mrd['measuredPosition']['latitude'] for mrd in mrd_list],
        "longitude": [mrd['measuredPosition']['longitude'] for mrd in mrd_list],
        "speed": [mrd['additionalGnssData']['speed'] for mrd in mrd_list],
        "heading": [mrd['additionalGnssData']['heading'] for mrd in mrd_list],
        "weightLimits": vehicle_desc['weightLimits']['VehicleWeightLimits']['vehicleTechnicalMaxLadenWeight'],
        "CO2EmissionClass": vehicle_desc['specificCharacteristics']['environmentalCharacteristics']['euCO2EmissionClass']

    })

    return df, truck_id



# -----------------------
# Prepare output folder
# -----------------------
output_folder.mkdir(exist_ok=True)


# -----------------------
# Process ZIP files
# -----------------------

# List all subfolders (daily data)
subfolders = [sf for sf in base_folder.iterdir() if sf.is_dir()]
total_subfolders = len(subfolders)

for i, subfolder in enumerate(subfolders, start=1):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Processing day [{i}/{total_subfolders}] at {now}")

    if subfolder.is_dir():
        # Iterate over ZIP files in this subfolder with a progress bar
        for zip_path in tqdm(subfolder.glob('*.zip'), desc="ZIPs", miniters=1):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_name in zip_ref.namelist():
                    if file_name.endswith('.json'):
                        with zip_ref.open(file_name) as f:
                            # Parse JSON into DataFrame
                            data = json.load(f)
                            df_chunk, truck_id = get_data(data)
                            # Append chunk to the corresponding truck CSV
                            truck_file_path = output_folder / f"truck_{truck_id}.csv"
                            write_header = not truck_file_path.exists()
                            df_chunk.to_csv(truck_file_path, mode="a", header=write_header, index=False)


# -----------------------
# Save / update TruckID mapping
# -----------------------
truck_map = pd.DataFrame({
    "TruckID": list(plate_to_id.values()),
    "licencePlateNumber": list(plate_to_id.keys())
})
truck_map.to_parquet(truck_file, index=False)

print(f"Finished ingestion. One CSV per truck in {output_folder}, truck mapping in {truck_file}")


# -----------------------
# Finalization: Convert CSVs to Parquet
# -----------------------
csv_files = list(output_folder.glob("truck_*.csv"))

for csv_file in tqdm(csv_files, desc="Converting CSV to Parquet"):
    df = pd.read_csv(csv_file, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Downcast here (one-shot, faster overall)
    df["latitude"] = pd.to_numeric(df["latitude"], downcast="integer")
    df["longitude"] = pd.to_numeric(df["longitude"], downcast="integer")
    df["speed"] = pd.to_numeric(df["speed"], downcast="integer")
    df["heading"] = pd.to_numeric(df["heading"], downcast="integer")
    df["weightLimits"] = pd.to_numeric(df["weightLimits"], downcast="integer")
    df["CO2EmissionClass"] = pd.to_numeric(df["CO2EmissionClass"], downcast="integer")
    df["TruckID"] = pd.to_numeric(df["TruckID"], downcast="integer")

    parquet_file = csv_file.with_suffix(".parquet")
    df.to_parquet(parquet_file, index=False)

    csv_file.unlink()