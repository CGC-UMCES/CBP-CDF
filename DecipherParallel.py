
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:07:38 2024

@author: xiaoxug
"""
import pandas as pd
import time
from datetime import datetime
from joblib import Parallel, delayed
import numpy as np  # Ensure numpy is imported
import psutil  # Import psutil for memory checking
from utilsV2 import (
    load_config,
    add_days,
    format_time_string,
    read_known_fortran_binary_file,
    read_remaining_fortran_binary_file,
    compute_values,
    create_netcdf_file,
    create_netcdf_file_update
)

# Function to check and print memory usage
def print_memory_usage(stage):
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"{stage} - Memory usage: {mem_info.rss / (1024 ** 2):.2f} MB")

# Load configuration
config = load_config('/data/xguo/merrill/Run/config.json')

# Extract constants
nsbp = config["constants"]["nsbp"]
nbp = config["constants"]["nbp"]
chunk_size = config["constants"]["chunk_size"]
endian = config["constants"]["endian"]
output_dir = config["constants"]['output_directory']

# Extract simulation parameters
start_jday = config["simulation_parameters"]["start_jday"]
end_jday = config["simulation_parameters"]["end_jday"]
utm_zone = config["simulation_parameters"]["utm_zone"]
NAC = config["simulation_parameters"]["NAC"]
nwcbox = config["simulation_parameters"]["nwcbox"]

var_names = config["simulation_parameters"]["var_names"]
var_units = config["simulation_parameters"]["var_units"]
calc_var_names = config["simulation_parameters"]["calc_var_names"]
calc_var_units = config["simulation_parameters"]["calc_var_units"]

start_year = config["simulation_parameters"]["start_year"]
start_month = config["simulation_parameters"]["start_month"]
start_day = config["simulation_parameters"]["start_day"]
binary_filename = config["simulation_parameters"]["binary_filename"]
csv_file = config["simulation_parameters"]["csv_file"]
area_file=config["simulation_parameters"]["area_file"]

# Load area data from the Excel file
area_data = pd.read_excel(area_file)
area_data['Cell_ID'] = area_data['Cell_ID'].astype(int) - 1
cell_id_to_area = area_data.set_index('Cell_ID')['Area'].to_dict()


cell_ids = list(range(1, nsbp + 1))  # nsbp from config
layer_ns = [
    1.067 if layernumber == 1 else (layernumber - 2) * 1.524 + 2.134 + 0.762
    for layernumber in range(1, 20)  # Assuming 19 layers
]
# Generate time values based on start and end dates
#start_date = np.datetime64(f"{start_year:04d}-{start_month:02d}-{start_day:02d}")
start_date = datetime.fromisoformat(str(np.datetime64(f"{start_year:04d}-{start_month:02d}-{start_day:02d}")))
end_date = start_date + np.timedelta64(end_jday - start_jday, 'D')  # End date based on jday difference
# Generate the time range
time_values = np.arange(start_date, end_date + np.timedelta64(1, 'D'), dtype='datetime64[D]')
# Read binary file data
data_Header = read_known_fortran_binary_file(binary_filename)

# Print the data_Header results
RED = "\033[31m"
RESET = "\033[0m"
# Print data header in red
print(f"{RED}Data from data_Header:{RESET}")
for key, value in data_Header.items():
    if 'mystery' not in key and 'placeholder' not in key:
        print(f"{RED}{key}: {value}{RESET}")

# Load file name from config
file_name = config["constants"]["file_name"]

BLUE = "\033[34m"
RESET = "\033[0m"
print(f"{BLUE}Provided nwcbox: {nwcbox}{RESET}")

# NetCDF setup
dimensions = ('Cell_ID', 'Layer_N', 'Time')

# Initialize NetCDF file
#netcdf_file = create_netcdf_file(var_names, calc_var_names, var_units, calc_var_units, dimensions, nsbp, output_dir)

# Initialize NetCDF file
netcdf_file=create_netcdf_file_update(var_names, calc_var_names, var_units, calc_var_units, cell_ids, layer_ns, time_values, output_dir)

# Read CSV data
df = pd.read_csv(csv_file)

# Initialize arrays to store values for bulk writing
n_days = end_jday - start_jday + 1
n_cells = np.max(df['Cell_ID'])  # Assuming Cell_ID is sequential
n_layers = np.max(df['Layer#'])   # Assuming Layer# is sequential

# Create arrays for bulk assignment
calc_values = {var_name: np.full((n_cells, n_layers, n_days), np.nan) for var_name in calc_var_names}
var_values = {var_name: np.full((n_cells, n_layers, n_days), np.nan) for var_name in var_names}
latitudes = np.full((n_cells, n_layers), np.nan)
longitudes = np.full((n_cells, n_layers), np.nan)
depths = np.full((n_layers), np.nan)
nwcbox_vals = np.full((n_cells, n_layers), np.nan)
dates = np.full(n_days, '', dtype='<U10')  # Assuming date is in string format
area_vals = np.full((n_cells, n_layers), np.nan)  # Added for Area


def process_nwcbox(jday, nwcbox_val):
    # Read remaining binary data
    data_real = read_remaining_fortran_binary_file(binary_filename, jday=jday)
    c1_array = data_real['c1'].reshape(NAC, nwcbox)

    # Each nwcbox processing here
    matching_rows = df[df['nwcbox_value'] == nwcbox_val]
    layernumber, cell_id = matching_rows['Layer#'].values[0], matching_rows['Cell_ID'].values[0]
    Depth = 1.067 if layernumber == 1 else (layernumber - 2) * 1.524 + 2.134 + 0.762
    lon, lat = matching_rows['Longitude'].values[0], matching_rows['Latitude'].values[0]
    area_value = cell_id_to_area.get(cell_id - 1, np.nan)  # Get the area value for this cell_id


    # Compute values
    c1_array_nwcbox = c1_array[:, nwcbox_val - 1]
    computed_values = compute_values(
        c1_array=c1_array_nwcbox,
        ke=data_real['ke'][nwcbox_val - 1],
        cchl1=data_real['cchl1'][nwcbox_val - 1],
        cchl2=data_real['cchl2'][nwcbox_val - 1],
        cchl3=data_real['cchl3'][nwcbox_val - 1],
        anc1=data_Header['anc1'], anc2=data_Header['anc2'], anc3=data_Header['anc3'],
        ancsz=0, anclz=0,
        kadpo4=data_Header['kadpo4'],
        apc1=data_Header['apc1'], apc2=data_Header['apc2'], apc3=data_Header['apc3'],
        apcsz=0, apclz=0,
        fi1=data_real['fi1'][nwcbox_val - 1], fi2=data_real['fi2'][nwcbox_val - 1], fi3=data_real['fi3'][nwcbox_val - 1],
        nl1=data_real['nl1'][nwcbox_val - 1], nl2=data_real['nl2'][nwcbox_val - 1], nl3=data_real['nl3'][nwcbox_val - 1],
        pl1=data_real['pl1'][nwcbox_val - 1], pl2=data_real['pl2'][nwcbox_val - 1], pl3=data_real['pl3'][nwcbox_val - 1],
        nwcbox_val=nwcbox_val, nsbp=nsbp
    )

    return cell_id, layernumber, jday, computed_values, lon, lat, Depth, nwcbox_val, c1_array_nwcbox, area_value

# Memory usage before processing starts
print_memory_usage("Before processing")

# Processing loop
for jday in range(start_jday, end_jday + 1):
    # Calculate date and time string
    days_to_add = jday - 1
    new_date = add_days(start_date, days_to_add)
    time_string = format_time_string(new_date)

    print(f"Processing Date: {time_string}")
    start_time = time.time()
    # Parallel processing for each nwcbox using 16 cores
    results = Parallel(n_jobs=30)(
        delayed(process_nwcbox)(jday, nwcbox_val) for nwcbox_val in range(1, nwcbox + 1)
    )

    elapsed_time = time.time() - start_time
    print(f"Time taken for parallel {time_string}: {elapsed_time:.2f} seconds")
    
    
    # Write data to the single NetCDF file
    for result in results:
        cell_id, layernumber, jday, computed_values, lon, lat, Depth, nwcbox_val, c1_array_nwcbox, area_value = result
        
        # Update the temporary arrays for bulk assignment
        for var_name in calc_var_names:
            calc_values[var_name][cell_id - 1, layernumber - 1, jday - start_jday] = computed_values['selected_values'][calc_var_names.index(var_name)]

        for var_name in var_names:
            # Check if the value is less than 0.1e-8 and set it accordingly
            value_to_write = c1_array_nwcbox[var_names.index(var_name)]
            if value_to_write < 0.1e-8:
                value_to_write = 0.1e-8
            var_values[var_name][cell_id - 1, layernumber - 1, jday - start_jday] = value_to_write

        latitudes[cell_id - 1, layernumber - 1] = lat
        longitudes[cell_id - 1, layernumber - 1] = lon
        depths[layernumber - 1] = Depth
        nwcbox_vals[cell_id - 1, layernumber - 1] = nwcbox_val
        
        if nwcbox_val == -9999:
            area_value = np.nan  # Assign NaN for invalid nwcbox values
        # Store area value in the array
        area_vals[cell_id - 1, layernumber - 1] = area_value
        
    # Assign dates after processing all results for the day
    dates[jday - start_jday] = time_string

    elapsed_time = time.time() - start_time
    print(f"Time taken for Date {time_string}: {elapsed_time:.2f} seconds")

    # Check memory usage after processing each day
    print_memory_usage(f"After processing day {jday}")
save_time=time.time()

# Before writing nwcbox_vals, check for NaN values
nwcbox_vals[np.isnan(nwcbox_vals)] = -1  # Replace NaN with a default value (e.g., -1)

# Ensure that nwcbox_vals matches the NetCDF variable type
nwcbox_vals = nwcbox_vals.astype(np.int32)  # Change the data type if necessary

# Now write to the NetCDF file
netcdf_file.variables['nwcbox'][:, :] = nwcbox_vals


# Write the accumulated data to the NetCDF file
for var_name, values in calc_values.items():
    netcdf_file.variables[var_name][:] = values

for var_name, values in var_values.items():
    netcdf_file.variables[var_name][:] = values

netcdf_file.variables['Latitude'][:, :] = latitudes
netcdf_file.variables['Longitude'][:, :] = longitudes
netcdf_file.variables['Depth'][:] = depths
netcdf_file.variables['nwcbox'][:, :] = nwcbox_vals
netcdf_file.variables['Date'][:] = dates
# Bulk assign Area to the NetCDF variable
netcdf_file.variables['Area'][:, :] = area_vals

# Close NetCDF file after writing all data
netcdf_file.close()

elapsed_time = time.time() - save_time
print(f"Time taken for saving to netcdf {time_string}: {elapsed_time:.2f} seconds")

print("Finished writing to NetCDF.")
