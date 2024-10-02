#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:01:37 2024

@author: xiaoxug
"""
import numpy as np
from netCDF4 import Dataset

def print_nc_structure(file_path):
    # Open the NetCDF file
    with Dataset(file_path, 'r') as nc:
        print("NetCDF File Structure:")
        
        # Print dimensions
        print("\nDimensions:")
        for dim_name, dim in nc.dimensions.items():
            print(f"  {dim_name}: size={len(dim)}")
        
        # Print variables and their attributes
        print("\nVariables:")
        for var_name, var in nc.variables.items():
            dimensions = var.dimensions
            dtype = var.dtype
            attrs = var.ncattrs()
            print(f"  {var_name}:")
            print(f"    Dimensions: {dimensions}")
            print(f"    Data type: {dtype}")
            if attrs:
                print("    Attributes:")
                for attr_name in attrs:
                    attr_value = getattr(var, attr_name)
                    print(f"      {attr_name}: {attr_value}")


def print_data_for_user_input(file_path, cell_id_index, jday_index, layer_index):
    
    # ANSI escape code for red text
    RED = "\033[91m"
    RESET = "\033[0m"
    
    
    # Open the NetCDF file
    with Dataset(file_path, 'r') as nc:
        # Print NetCDF structure
        print_nc_structure(file_path)
        
        # Retrieve dimensions: Cell_ID, Jday, and Layernumber
        dim_cell_size = nc.dimensions['Cell_ID'].size
        jday_size = nc.dimensions['time'].size
        layernumber_size = nc.dimensions['Layer#'].size
        
        # Retrieve all variable names
        var_names = list(nc.variables.keys())
        
        # Check if indices are within bounds
        if (cell_id_index >= dim_cell_size or 
            jday_index >= jday_size or 
            layer_index >= layernumber_size):
            print(f"Indices are out of range.")
            return
        
        # Retrieve and print the data at the specified indices
        print(f"{RED}\nData points for input indices (Cell_ID={cell_id_index+1}, ctime={jday_index+1}, Layernumber={layer_index+1}):")
        
        # Print all variables at the specified indices
        for var_name in var_names:
            # Handle different dimensions for each variable
            try:
                var_data = nc.variables[var_name]
                # Determine the number of dimensions for indexing
                num_dims = var_data.ndim
                
                # Prepare indexing based on dimensions
                if num_dims == 3:
                    # 3D variable: Index using all three indices
                    value = var_data[cell_id_index, layer_index,jday_index]
                elif num_dims == 2:
                    # 2D variable: Index using Cell_ID and Jday
                    value = var_data[cell_id_index, layer_index]
                elif num_dims == 1:
                    # 1D variable: Index using the time dimension
                    value = var_data[jday_index]  # Assuming 'time' dimension corresponds to jday_index
                else:
                    # If variable is scalar (0D), simply retrieve it
                    value = var_data
                
                # Get units if available
                units = var_data.units if 'units' in var_data.ncattrs() else 'No units'
                print(f"{var_name}: {value} ({units})")
                
            except IndexError:
                print(f"{var_name}: Unable to access data with provided indices.")
            except KeyError:
                print(f"{var_name}: Variable not found in the dataset.")


def main():
    file_path = '/Users/xiaoxug/CBP-Decipher/netCDF_Output_old/S.nc'  # Replace with your actual NetCDF file path
    
    # Ask the user for input indices
    cell_id_index = int(input("Enter the Cell_ID index you are looking for: "))-1
    layer_index = int(input("Enter the Layernumber index you are looking for: "))-1
    jday_index = int(input("Enter the Jday index you are looking for: "))-1

    print_data_for_user_input(file_path, cell_id_index, jday_index, layer_index)
    
if __name__ == "__main__":
    main()
