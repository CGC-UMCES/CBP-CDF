#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 16:34:17 2024

@author: xiaoxug
"""
import netCDF4 as nc
import numpy as np

def create_variable_with_fillvalue(output_file, var_name, var):
    fill_value = var._FillValue  # Use the defined fill value from the source variable
    return output_file.createVariable(var_name, var.datatype, var.dimensions, fill_value=fill_value)

def copy_variable_attributes(source_variable, target_variable):
    for attr_name in source_variable.ncattrs():
        if attr_name != '_FillValue':  # Skip the _FillValue attribute
            target_variable.setncattr(attr_name, source_variable.getncattr(attr_name))

# Open the two source NetCDF files
file1 = nc.Dataset("/Users/xiaoxug/CBP-Decipher/netCDF_Output/All_Var1.nc", "r")
file2 = nc.Dataset("/Users/xiaoxug/CBP-Decipher/netCDF_Output/All_Var2.nc", "r")

# Create the output NetCDF file
output_file = nc.Dataset("/Users/xiaoxug/CBP-Decipher/netCDF_Output/combine.nc", "w", format="NETCDF4")

# Create dimensions from the first file
for dim_name, dim in file1.dimensions.items():
    output_file.createDimension(dim_name, len(dim) if not dim.isunlimited() else None)

# Handle 'Time' dimension (assuming you want to combine using 'Date' as the time variable)
time_size1 = len(file1.variables['Date'])
time_size2 = len(file2.variables['Date'])
total_time_size = time_size1 + time_size2

# Create the 'Date' variable in the output file
time_var_out = output_file.createVariable('Date', file1.variables['Date'].datatype, ('Time',))
time_var_out[:time_size1] = file1.variables['Date'][:]  # Copy time values from file1
time_var_out[time_size1:total_time_size] = file2.variables['Date'][:]  # Append time values from file2

# Create variables from the first file
for var_name, variable in file1.variables.items():
    if var_name != 'Date':  # Skip the Date variable
        out_var = create_variable_with_fillvalue(output_file, var_name, variable)
        
        if variable.ndim == 3:  # For 3D variables
            out_var[:, :, :time_size1] = variable[:]  # Copy data
        elif variable.ndim == 2:  # For 2D variables
            out_var[:, :] = variable[:]  # Copy data
       
            
        copy_variable_attributes(variable, out_var)

# Now handle the second file's variables
for var_name, variable in file2.variables.items():
    if var_name != 'Date':  # Skip the Date variable
        if var_name in output_file.variables:
            existing_var = output_file.variables[var_name]
            new_data = variable[:]  # Get all data for the variable

            if variable.ndim == 3:  # For 3D variables
                existing_var[:, :, time_size1:total_time_size] = new_data[:, :, :]  # Append data
            elif variable.ndim == 2:  # For 2D variables
                existing_var[:,:] = new_data[:, :]  # Append data
 
        else:
            out_var = create_variable_with_fillvalue(output_file, var_name, variable)
            
            if variable.ndim == 3:  # For 3D variables
                out_var[:, :, time_size1:total_time_size] = variable[:]  # Copy data
            elif variable.ndim == 2:  # For 2D variables
                out_var[:,:] = variable[:]  # Copy data
      
            
            copy_variable_attributes(variable, out_var)

# Close the files
file1.close()
file2.close()
output_file.close()

print("Combined NetCDF files successfully!")
