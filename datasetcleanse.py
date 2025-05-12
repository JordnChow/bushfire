import pandas as pd
import os
from glob import glob

# Folder containing all your CSVs
folder_path = "./output_files2"  # Change this

# List to store the flattened data for each file
flattened_data = []
i = 0
# Loop through each CSV file

for filepath in glob(f"{folder_path}/*.csv"):
    i += 1
    try:
        df = pd.read_csv(filepath)
        print(f"Printing {filepath} #{i}")
        # Get Region and Date from the first row (the fire info row)
        region = df.loc[0, 'Region']
        date = pd.to_datetime(df.loc[0, 'Date'])

        # Extract weather rows (where 'Parameter' is not NaN)
        weather_df = df[df['Parameter'].notna()]

        # Flatten weather data
        weather_flat = {}
        for _, row in weather_df.iterrows():
            param = row['Parameter'].strip()
            mean_val = row['mean()']
            weather_flat[f"{param}_mean"] = mean_val

        # Optional: Extract fire-related metrics from the top row too
        fire_info = {
            'Estimated_fire_area': df.loc[0, 'Estimated_fire_area'],
            'Mean_fire_brightness': df.loc[0, 'Mean_estimated_fire_brightness'],
            'Mean_frp': df.loc[0, 'Mean_estimated_fire_radiative_power'],
            'Mean_confidence': df.loc[0, 'Mean_confidence'],
            'Hotspot_count': df.loc[0, 'Count'],
        }

        # Combine all into one dictionary
        final_row = {
            'Region': region,
            'Date': date,
            **fire_info,
            **weather_flat
        }

        flattened_data.append(final_row)

    except Exception as e:
        print(f"Error processing {filepath}: {e}")

# Create a DataFrame of all flattened rows
final_df = pd.DataFrame(flattened_data)

# Save to a new CSV
final_df.to_csv("flattened_wildfire_data.csv", index=False)
