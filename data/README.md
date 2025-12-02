# Data Directory

## Required Files

Place the following CSV files in this directory:

- `collision_2025.csv` - Main collision/accident records
- `vehicle_2025.csv` - Vehicle information for each accident
- `casualty_2025.csv` - Casualty (injury/death) records

## Data Source

**UK Department for Transport (DfT) - Road Casualty Statistics**

Download from: https://www.data.gov.uk/dataset/road-accidents-safety-data

Look for: "Road Safety Data - Provisional 2025"

## File Descriptions

### collision_2025.csv
- Each row = one traffic accident
- Key fields: date, time, location, weather, road conditions, severity
- Size: ~9 MB, 48,472 records

### vehicle_2025.csv
- Each row = one vehicle involved in an accident
- Key fields: vehicle type, driver age, manoeuvre
- Size: ~8.6 MB, 87,805 records

### casualty_2025.csv
- Each row = one casualty (injured/killed person)
- Key fields: severity, age, casualty type
- Size: ~4.7 MB, 60,991 records

## Note

CSV files are excluded from git via `.gitignore` to keep repository size small.
