import requests
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np

"""
Fetch all the locations with a hotspot in australia
"""
def fetch_hotspots_australia(bbox=[140.999,-37.505,153.638,-28.157]):
    print("Started1")
    # Default bounding box is for NSW

    MAP_KEY = "6c8a65ccb7fd9e0e7caef47ad2c3fb49"

    # MODIS near real time data, last 24 hours, CSV output
    # https:/firms.modaps.eosdis.nasa.gov/api/area/csv/MAP_KEY/SATELLITE or SENSOR/BOUNDINGBOX/DAYS
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/MODIS_NRT/{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}/1"

    # Send request
    response = requests.get(url)
    # Check for successful request
    if response.status_code == 200:
        # Load into DataFrame
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
    else:
        print(f"Failed to fetch data: {response.status_code}")

    url = "https://api.open-meteo.com/v1/forecast"
    results_df = pd.DataFrame()
    i=0

    # Loop through top FIRMS high-confidence hotspots
    for idx,row in enumerate(df.head(1).iterrows()):
        i+=1
        params = {
            "latitude": row[1].latitude,
            "longitude": row[1].longitude,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,soil_moisture_3_9cm,shortwave_radiation,windspeed_10m",
            "timezone": "auto"
        }
        try:
            response = requests.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                hourly = data.get("hourly", {})
                weather_data = {
                    "FIRMS_confidence": row[1].confidence,
                    "latitude": row[1].latitude,
                    "longitude": row[1].longitude,
                    "brightness": row[1].brightness,
                    "precipitation_mm": f"{hourly['precipitation'][0]}",
                    "relative_humidity": hourly["relative_humidity_2m"][0],
                    "soil_water_content_m3_m3": f"{hourly['soil_moisture_3_9cm'][0]}",
                    "solar_radiation_w_m2": hourly["shortwave_radiation"][0],
                    "temperature_c": hourly["temperature_2m"][0],
                    "wind_speed_m_s": hourly["windspeed_10m"][0]
                }

                new_row = pd.DataFrame([weather_data])
                results_df = pd.concat([results_df, new_row], ignore_index=True)
            else:
                print("Failed to fetch data. Status code:", response.status_code)
        except Exception as e:
            print(f"Request failed at row {idx}: {e}")

    """
    Class for ML model
    """
    class RegressionNN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.model(x)
        
    data_df = pd.read_csv("./flattened_wildfire_data.csv")

    X = data_df.drop(["Mean_confidence", "Std_confidence"], axis=1).values
    input_dim = X.shape[1] 
    model = RegressionNN(input_dim)
    model.load_state_dict(torch.load("wildfire_nn_weights1.pth"))
    model.eval()

    #Scale the data
    scaler_X = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(data_df["Mean_confidence"].values.reshape(-1, 1))
    confidence = []

    for row in results_df.iterrows():
        info = row[1]
        print(info)
        new_input = [[x for x in info[3:]]]
        new_input_scaled = scaler_X.transform(new_input)
        
        # Convert to tensor
        new_input_tensor = torch.tensor(new_input_scaled, dtype=torch.float32)
        with torch.no_grad():
            prediction_scaled = model(new_input_tensor)
            prediction_unscaled = scaler_y.inverse_transform(prediction_scaled.numpy())
            prediction_capped = min(100, prediction_unscaled[0][0])
        adjusted_confidence = (prediction_capped*3 + row[1].FIRMS_confidence) / 4
        confidence.append(adjusted_confidence)
    results_df["confidence"] = confidence
    return results_df