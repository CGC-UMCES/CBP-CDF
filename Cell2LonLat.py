#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:23:42 2024

@author: xiaoxug
"""
import pandas as pd
from pyproj import Proj, Transformer

def read_fortran_file(filename, nsbp):
    # Lists to store the read values
    ihy = []
    jhy = []
    nbox = []
    nwcbox = []

    with open(filename, 'r') as file:
        for _ in range(nsbp):
            line = file.readline()   
            ihy_val = int(line[:3])
            jhy_val = int(line[3:6]) 
            nbox_val = int(line[6:9])
            av = line[9:] 
            additional_values = [int(num) for num in av.split() if num.isdigit()]         
            ihy.append(ihy_val)
            jhy.append(jhy_val)
            nbox.append(nbox_val)
            nwcbox.append(additional_values)
                
    return ihy, jhy, nbox, nwcbox

def find_all_values(start, end, ihy, jhy, nwcbox):
    results = []
    
    for i, nwcbox_vals in enumerate(nwcbox):
        for pos, val in enumerate(nwcbox_vals):
            if start <= val <= end:
                results.append({
                    'nwcbox_value': val,
                    'Layer#': pos + 1,
                    'I': ihy[i],
                    'J': jhy[i]
                })
    
    return results

def utm_to_latlon(utm_easting: float, utm_northing: float, utm_zone: int, northern_hemisphere: bool = True) -> tuple:
    """Convert UTM coordinates to latitude and longitude."""
    utm_proj = Proj(proj='utm', zone=utm_zone, datum='WGS84', south=not northern_hemisphere)
    wgs84_proj = Proj(proj='latlong', datum='WGS84')
    transformer = Transformer.from_proj(utm_proj, wgs84_proj)
    return transformer.transform(utm_easting, utm_northing)

# Read the data
filename = 'col_cbay_56920.dat'  # Replace with your actual file path
nsbp = 11064  # Replace with the actual number of lines to read

# Read the data
ihy, jhy, nbox, nwcbox = read_fortran_file(filename, nsbp)

# Define the range of values to find
start_value = 1
end_value = 56920

# Find all values in the specified range
results = find_all_values(start_value, end_value, ihy, jhy, nwcbox)

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Read the second CSV file (Cell_ID, I, J, X, Y)
file2 = 'col_cbay_56920xy.csv'  # Replace with the actual file path
df2 = pd.read_csv(file2)

# Step 1: Ensure columns do not have leading/trailing spaces
df2.columns = df2.columns.str.strip()

# Step 2: Merge the DataFrames on 'I' and 'J' columns
merged_df = pd.merge(results_df, df2, on=['I', 'J'], how='inner')

# Step 3: Convert UTM (X, Y) to latitude and longitude
utm_zone = 18  # Replace with your actual UTM zone if different
latitudes = []
longitudes = []

for index, row in merged_df.iterrows():
    lon, lat = utm_to_latlon(row['X'], row['Y'], utm_zone)  # Convert X and Y to lon and lat
    latitudes.append(lat)
    longitudes.append(lon)

merged_df['Latitude'] = latitudes  # Add Latitude column
merged_df['Longitude'] = longitudes  # Add Longitude column

output_file = 'Cell2LonLat.csv'  # Desired output file name
merged_df.to_csv(output_file, index=False)


print("Cellid2LonLat.csv files created successfully.")
print(merged_df)  # Display the merged DataFrame
