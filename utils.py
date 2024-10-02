
"""
Created on Tue Sep 24 15:07:53 2024

@author: xiaoxug
"""
import json
import numpy as np
from netCDF4 import Dataset
from typing import Dict, Any,List
from pyproj import Proj, Transformer
from datetime import datetime, timedelta
import logging
import os

# Load your JSON configuration file
with open('config.json') as f:
    config = json.load(f)



def load_config(file_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config


def add_days(start_date: datetime, days_to_add: int) -> datetime:
    """Add days to a given start date."""
    return start_date + timedelta(days=days_to_add)

def format_time_string(date: datetime) -> str:
    """Format the time as 'year-month-day-hour'."""
    return f"{date.year}-{date.month:02d}-{date.day:02d}-{date.hour:02d}"

def utm_to_latlon(utm_easting: float, utm_northing: float, utm_zone: int, northern_hemisphere: bool = True) -> tuple:
    """Convert UTM coordinates to latitude and longitude."""
    utm_proj = Proj(proj='utm', zone=utm_zone, datum='WGS84', south=not northern_hemisphere)
    wgs84_proj = Proj(proj='latlong', datum='WGS84')
    transformer = Transformer.from_proj(utm_proj, wgs84_proj)
    return transformer.transform(utm_easting, utm_northing)

def read_known_fortran_binary_file(filename: str, config_file: str = 'config.json') -> dict:
    """
    Function to read a known portion of a binary file using Fortran format.
    Args:
    - filename (str): The path to the binary file.
    - config_file (str): Path to the JSON configuration file.
    
    Returns:
    - dict: Dictionary containing the parsed data.
    """
    # Load configuration from JSON
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file {config_file} not found.")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {config_file}.")
        raise

    constants = config['constants']
    dtype_known = config['dtype_known']

    # Create a numpy dtype from the dtype_known definitions
    np_dtype = []
    for item in dtype_known:
        if item['type'] == 'nsbpf4':
            array_size = constants['nsbp']
            np_dtype.append((item['name'], f'{array_size}f4'))
        elif item['type'] == 'nwcboxf4':
            array_size = constants['nwcbox']
            np_dtype.append((item['name'], f'{array_size}f4'))
        else:
            np_dtype.append((item['name'], item['type']))

    np_dtype = np.dtype(np_dtype)

    # Apply endianness
    if constants['endian'] == 'big':
        np_dtype = np_dtype.newbyteorder('>')
    elif constants['endian'] == 'little':
        np_dtype = np_dtype.newbyteorder('<')
    else:
        raise ValueError("Endianness must be 'big' or 'little'")

    known_data = []
    try:
        with open(filename, 'rb') as file:
            chunk = file.read(constants['chunk_size'])
            logging.info("Start loading")
            # Convert the chunk to a numpy array
            chunk_data = np.frombuffer(chunk, dtype=np_dtype)
            logging.info("Finish loading")
            
            # Process known portion of the data
            if chunk_data.size > 0:
                known_data.extend(chunk_data)
    except FileNotFoundError:
        logging.error(f"Binary file {filename} not found.")
        raise
    except Exception as e:
        logging.error(f"Error reading binary file: {e}")
        raise

    # Extract the first record if it exists
    if known_data:
        record_known = known_data[0]
    else:
        record_known = {item['name']: None for item in dtype_known}

       
    # Prepare the result dictionary
    result = {}
    if record_known is not None:
        for name in record_known.dtype.names:
            result[name] = record_known[name]
    
    # Convert logical values to boolean
    for key in ['quality_diag', 'sediment_diag', 'sav']:
        if key in result and result[key] is not None:
            result[key] = bool(result[key])

    return result

def read_remaining_fortran_binary_file(filename: str, config_file: str = 'config.json', jday: int = 1) -> Dict[str, Any]:
    """
    Function to read the remaining portion of a binary file using Fortran format after known chunk size.

    Args:
    - filename (str): The path to the binary file.
    - config_file (str): Path to the JSON configuration file.
    - jday (int): Julian day to compute chunk size for remaining data.

    Returns:
    - dict: Dictionary containing the parsed data.
    """
    # Load configuration from JSON
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading config: {e}")
        raise

    constants = config['constants']
    
    # Calculate chunk size based on Julian day
    chunk_size = 493896 + (jday - 1) * (12856748 + 4 * 11064)

    # Define the remaining portion of the data types and shapes from the config
    dtype_remaining = config['dtype_remaining']
    
    np_dtype_remaining = []
    for item in dtype_remaining:
        if ' * ' in item['type']:  # Handle cases like "nwcbox * 28f4"
            base_type, multiplier = item['type'].split(' * ')
            
            # Extract the numeric part and the type part after the first letter
            for i, char in enumerate(multiplier):
                if char.isalpha():
                    num_part = multiplier[:i]  # Numeric part (e.g., '28')
                    type_part = multiplier[i:]  # Type part (e.g., 'f4')
                    break
            else:
                num_part = multiplier
                type_part = ''
            
            if base_type == 'nwcbox':
                array_size = constants['nwcbox']
                total_size = int(num_part) * array_size
                np_dtype_remaining.append((item['name'], f'{total_size}{type_part}'))  # e.g., 'f4'
            elif base_type == 'nsbp':
                array_size = constants['nsbp']
                total_size = int(num_part) * array_size
                np_dtype_remaining.append((item['name'], f'{total_size}{type_part}'))  # e.g., 'f4'
            else:
                # Handle other cases as needed
                np_dtype_remaining.append((item['name'], item['type']))
        else:
            # Handle regular types directly
            if item['type'] == 'nsbpf4':
                array_size = constants['nsbp']
                np_dtype_remaining.append((item['name'], f'{array_size}f4'))
            elif item['type'] == 'nwcboxf4':
                array_size = constants['nwcbox']
                np_dtype_remaining.append((item['name'], f'{array_size}f4'))
            else:
                np_dtype_remaining.append((item['name'], item['type']))
    
    np_dtype_remaining = np.dtype(np_dtype_remaining)
      # Apply endianness from the config
    if constants['endian'] == 'big':
        np_dtype_remaining = np_dtype_remaining.newbyteorder('>')
    elif constants['endian'] == 'little':
        np_dtype_remaining = np_dtype_remaining.newbyteorder('<')
    else:
        raise ValueError("Endianness must be 'big' or 'little'")
    
    remaining_data = []
    
    # Read the binary file
    with open(filename, 'rb') as file:
        file.seek(chunk_size)  # Use constants['chunk_size'] to skip initial data
        chunk = file.read(12856748)  # Replace with the appropriate chunk size if needed
        chunk_data = np.frombuffer(chunk, dtype=np_dtype_remaining)
    
        if chunk_data.size > 0:
            remaining_data.extend(chunk_data)
    
    # Extract the first record if it exists
    if remaining_data:
        record_remaining = remaining_data[0]
    else:
        return {}
    
    # Prepare the result dictionary using the record directly
    result = {name: record_remaining[name] for name in record_remaining.dtype.names}
    
    return result


def extend_c1_array(c1_array, new_dim=36):
    """Extend the c1_array to the specified new dimension, filling in missing values."""
    extended_array = np.zeros(new_dim)  # Initialize extended array
    indices = config['data_dimensions']['index_mapping_c1']  # Get index mapping from config
    for i in range(len(indices)):
        if i < len(c1_array):
            extended_array[indices[i]] = c1_array[i]
    return extended_array

def compute_values(c1_array, ke, cchl1, cchl2, cchl3, anc1, anc2, anc3, ancsz, anclz, kadpo4, apc1, apc2, apc3, apcsz, apclz, fi1, fi2, fi3, nl1, nl2, nl3, pl1, pl2, pl3,nwcbox_val,nsbp):
  
    # Extend c1_array to 36 dimensions
   c1_array_extended = extend_c1_array(c1_array, new_dim=36)
   
   val = np.zeros(17)  # Only store the values for 17 variables
   LFactor = ""  # LFactor is now a single value instead of a list
   # Compute various values using the extended array
   val[0] = max(0.0, c1_array_extended[26])  # Example variable
   val[1] = max(0.0, c1_array_extended[0])  # Temperature
   val[2] = max(0.0, c1_array_extended[1])  # Salinity
   
   val[3] = max(0.0, ke) if nwcbox_val <= nsbp else 0.0

   # Chlorophyll
   val[4] = max(0.0, 1000.0 * (c1_array_extended[3] / cchl1 + c1_array_extended[4] / cchl2 + c1_array_extended[5] / cchl3))

   # Ammonium
   val[5] = max(0.0, c1_array_extended[12])
   # Nitrate
   val[6] = max(0.0, c1_array_extended[13])
   # Organic N
   val[7] = (anc1 * c1_array_extended[3] + anc2 * c1_array_extended[4] + anc3 * c1_array_extended[5] + c1_array_extended[17]
             + c1_array_extended[18] + c1_array_extended[30] + ancsz * c1_array_extended[6] + anclz * c1_array_extended[7] + c1_array_extended[15] + c1_array_extended[16])
   # Total Nitrogen
   val[8] = val[5] + val[6] + val[7]
   
   # Dissolved Inorganic Phosphorus
   df = 1.0 / (1.0 + kadpo4 * c1_array_extended[2])
   val[9] = df * c1_array_extended[19]
   
   # Particulate Organic Phosphorus
   val[10] = (c1_array_extended[22] + c1_array_extended[23] + c1_array_extended[31] + apc1 * c1_array_extended[3] + apc2 * c1_array_extended[4]
              + apc3 * c1_array_extended[5] + apcsz * c1_array_extended[6] + apclz * c1_array_extended[7] + c1_array_extended[20] + c1_array_extended[21])
   
   # Particulate Inorganic Phosphorus
   pf = kadpo4 * c1_array_extended[2] / (1.0 + kadpo4 * c1_array_extended[2])
   val[11] = pf * c1_array_extended[19] + c1_array_extended[24]
   
   # Total Phosphorus
   val[12] = val[11] + val[10] + val[9]
   
   # Total Solids
   val[13] = (c1_array_extended[32] + c1_array_extended[33] + c1_array_extended[34] + c1_array_extended[35]
              + 2.5 * (c1_array_extended[3] + c1_array_extended[4] + c1_array_extended[5] + c1_array_extended[6] + c1_array_extended[7] + c1_array_extended[10] + c1_array_extended[11] + c1_array_extended[29]))
   
   # Light Limitation
   val[14] = (fi1 * c1_array_extended[3] + fi2 * c1_array_extended[4] + fi3 * c1_array_extended[5]) / (c1_array_extended[3] + c1_array_extended[4] + c1_array_extended[5])
   
   # Nitrogen Limitation
   val[15] = (nl1 * c1_array_extended[3] + nl2 * c1_array_extended[4] + nl3 * c1_array_extended[5]) / (c1_array_extended[3] + c1_array_extended[4] + c1_array_extended[5])
   
   # Phosphorus Limitation
   val[16] = (pl1 * c1_array_extended[3] + pl2 * c1_array_extended[4] + pl3 * c1_array_extended[5]) / (c1_array_extended[3] + c1_array_extended[4] + c1_array_extended[5])
   
   # Determine LFactor based on computed values
   if val[15] > 0.833 and val[16] > 0.833:
       LFactor = "NR"
   elif val[15] <= 0.5 and val[16] < 0.5:
       LFactor = "NP"
   elif val[15] < val[16]:
       LFactor = "N"
   else:
       LFactor = "P"
   
   # Indices of values to write
   indices_to_write = [4, 8, 12, 7, 10, 13, 14, 15, 16]
   selected_values = val[indices_to_write]  # Select specific values
   
   return {'selected_values': selected_values, 'LFactor': LFactor}

def create_netcdf_file(var_names: List[str], var_units: List[str], dimensions: tuple, nsbp: int,output_dir:str) -> Dict[str, Dataset]:
    """Create NetCDF files for given variable names and initialize dimensions and variables."""
    
    # Check if the output directory exists; create if it doesn't
    if not os.path.exists(output_dir):
       os.makedirs(output_dir)
   
   # Check for existing files in the directory
    for var_name in var_names:
       file_path = os.path.join(output_dir, f'{var_name}.nc')
       if os.path.isfile(file_path):
           print(f"\033[91mFound existing file: {file_path}. Please change the output directory!! EXIT.\033[0m")
           exit()

   # Create NetCDF files for each variable
    netcdf_files = {}
    for var_name in var_names:
       netcdf_files[var_name] = Dataset(os.path.join(output_dir, f'{var_name}.nc'), 'w', format='NETCDF4')
    
    
    for var_name, file in netcdf_files.items():
        # Create dimensions
        file.createDimension('Cell_ID', nsbp)
        file.createDimension('Layer#', None)  # Unlimited for dynamic sizing
        file.createDimension('time', None)  # Use time as a dimension
        
        # Create fixed variables
        lat = file.createVariable('Latitude', 'f4', dimensions, zlib=True, fill_value=np.nan)
        lat.units = 'degrees_north'
        
        lon = file.createVariable('Longitude', 'f4', dimensions, zlib=True, fill_value=np.nan)
        lon.units = 'degrees_east'
        
        date = file.createVariable('Date', 'S19', dimensions, fill_value=np.nan)
        date.units = 'days y-m-d'  # Adjust to your reference date
        
        depth = file.createVariable('Depth', 'f4', dimensions, zlib=True, fill_value=np.nan)
        depth.units = 'meters'
        
        nwcbox = file.createVariable('nwcbox', 'f4', dimensions, zlib=True, fill_value=np.nan)
        nwcbox.units = '#'
        
        # Create a single variable corresponding to the file's name
        var = file.createVariable(var_name, 'f4', dimensions, zlib=True, fill_value=np.nan)
        var.units = var_units[var_names.index(var_name)]  # Set the units for the specific variable

    return netcdf_files