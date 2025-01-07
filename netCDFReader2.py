
import xarray as xr

# Open the NetCDF dataset
ds = xr.open_dataset("/data/xguo/merrill/Run/netCDF_Output/basetest/Year1.nc")

# Print the dataset without truncation
with xr.set_options(display_width=120):
    print(ds)

# Print variable values at slice numbers Cell_ID=694, Layer_N=3, Time=1
print("\nVariable values at slice numbers (Cell_ID=694, Layer_N=3, Time=1):")
for var_name in ds.data_vars:
    try:
        # Dynamically slice based on variable dimensions
        variable = ds[var_name]
        indexers = {}
        if 'Cell_ID' in variable.dims:
            indexers['Cell_ID'] = 694
        if 'Layer_N' in variable.dims:
            indexers['Layer_N'] = 3
        if 'Time' in variable.dims:
            indexers['Time'] = 1

        value = variable.isel(**indexers).values
        unit = variable.attrs.get('units', 'No unit specified')  # Get the unit attribute, default if not present
        print(f"  {var_name}: {value} ({unit})")
    except Exception as e:
        print(f"  {var_name}: Skipped due to error ({e})")
