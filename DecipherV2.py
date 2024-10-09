#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:55:17 2024

@author: xiaoxug
"""

import pandas as pd
import time
from datetime import datetime
from utilsV2 import (
    load_config,
    add_days,
    format_time_string,
    read_known_fortran_binary_file,
    read_remaining_fortran_binary_file,
    compute_values,
    create_netcdf_file
)

# Load configuration
config = load_config('/Users/xiaoxug/CBP-Decipher/config.json')

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

start_date = datetime(start_year, start_month, start_day)
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
netcdf_file = create_netcdf_file(var_names, calc_var_names, var_units, calc_var_units, dimensions, nsbp, output_dir)

# Processing loop
for jday in range(start_jday, end_jday + 1):
    # Calculate date and time string
    days_to_add = jday - 1
    new_date = add_days(start_date, days_to_add)
    time_string = format_time_string(new_date)
   
    print(f"Processing Date: {time_string}")
    start_time = time.time()

    # Read remaining binary data
    data_real = read_remaining_fortran_binary_file(binary_filename, jday=jday)
    c1_array = data_real['c1'].reshape(NAC, nwcbox)

    # Read CSV data
    df = pd.read_csv(csv_file)

    for nwcbox_val in range(1, nwcbox + 1):
        start_time_l = time.time()
        matching_rows = df[df['nwcbox_value'] == nwcbox_val]
        layernumber, cell_id = matching_rows['Layer#'].values[0], matching_rows['Cell_ID'].values[0]
        Depth = 1.067 if layernumber == 1 else (layernumber - 2) * 1.524 + 2.134 + 0.762
        lon, lat = matching_rows['Longitude'].values[0], matching_rows['Latitude'].values[0]

        # Compute values using helper function
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

        # Write data to the single NetCDF file
        for var_name in calc_var_names:
            netcdf_file.variables[var_name][cell_id - 1, layernumber - 1, jday - start_jday] = computed_values['selected_values'][calc_var_names.index(var_name)]

        for var_name in var_names:
        # Check if the value is less than 0.1e-8 and set it accordingly
           value_to_write = c1_array_nwcbox[var_names.index(var_name)]
           if value_to_write < 0.1e-8:
               value_to_write = 0.1e-8
           netcdf_file.variables[var_name][cell_id - 1, layernumber - 1, jday - start_jday] = value_to_write

        netcdf_file.variables['Latitude'][cell_id - 1, layernumber - 1] = lat
        netcdf_file.variables['Longitude'][cell_id - 1, layernumber - 1] = lon
        netcdf_file.variables['Date'][jday - start_jday] = time_string
        netcdf_file.variables['Depth'][cell_id - 1, layernumber - 1, jday - start_jday] = Depth
        netcdf_file.variables['nwcbox'][cell_id - 1, layernumber - 1, jday - start_jday] = nwcbox_val

        if nwcbox_val % 1000 == 0:
            print(f"Processed # of nwcbox: {nwcbox_val}, please wait...")

    elapsed_time = time.time() - start_time
    print(f"Time taken for Date {time_string}: {elapsed_time:.2f} seconds")

print("All data written successfully!")

# Close the NetCDF file
netcdf_file.close()


