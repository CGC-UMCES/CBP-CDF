#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 16:23:26 2024

@author: xiaoxug
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib.backend_bases import MouseButton
import numpy as np
from geopy.distance import geodesic
import netCDF4
from datetime import datetime

# Load observation data from CSV file and convert 'SampleDate' to datetime format
obs_data = pd.read_csv('wq_do_chla_po4.csv')  # Adjust to your file path
obs_data['SampleDate'] = pd.to_datetime(obs_data['SampleDate'], errors='coerce')  # Ensure proper datetime format

# Load Cell2LonLat data
cell_data = pd.read_csv('Cell2LonLat.csv')

# Extract unique parameters from the observation data
available_parameters = obs_data['Parameter'].unique()
# Define the parameter mapping
param_mapping = {
    'CHLA': 'Chl',
    'PO4F': 'PO4'
}
# Replace parameters in available_parameters based on param_mapping
available_parameters = [
    param_mapping.get(param, param) for param in available_parameters
]


# Convert Layer# to Depth using the given equation
def layer_to_depth(layer):
    return 1.067 if layer == 1 else (layer - 2) * 1.524 + 2.134 + 0.762

cell_data['Depth'] = cell_data['Layer#'].apply(layer_to_depth)

# Function to plot observation data scatter plot
def plot_obs_data(ax):
    # Compute the total number of records at each (X, Y) coordinate
    record_counts = obs_data.groupby(['Longitude', 'Latitude']).size().reset_index(name='count')

    # Scatter plot for observation data
    sc_obs = ax.scatter(record_counts['Longitude'], record_counts['Latitude'],
                        c=record_counts['count'], cmap='viridis', s=50, alpha=0.9)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Observation Data')

    # Set color bar
    cbar_obs = plt.colorbar(sc_obs, ax=ax)
    cbar_obs.set_label('Total Records')

    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, record_counts, ax))

# Function to show global analysis in a new window
#def show_global_analysis(event):
    # Create a new figure for global analysis
    #global_fig, ax_global = plt.subplots(figsize=(8, 6))

    # Show available depths and their record counts
    #depth_counts = obs_data['Depth'].value_counts().sort_index()

    # Prepare the text output for depths
    #depth_text = "\n".join([f"Depth: {depth}, Records: {count}" for depth, count in depth_counts.items()])
    
    # Display depth counts
    #ax_global.text(0.5, 0.5, depth_text, fontsize=12, ha='center', va='center')
    #ax_global.axis('off')  # Hide axes

    # Show the global analysis figure
    #plt.show()

# Function to find the 7 nearest grid points from the clicked location
def find_nearest_grids(lon, lat, depth, num_points=15):
    def calculate_distance(row):
        # Calculate horizontal distance (in meters) using geodesic distance for lat/lon
        horizontal_distance = geodesic((lat, lon), (row['Latitude'], row['Longitude'])).meters
        # Calculate depth difference (in meters)
        depth_difference = abs(depth - row['Depth'])
        # Combine horizontal distance and depth difference to get 3D distance
        return np.sqrt(horizontal_distance**2 + depth_difference**2)
    
    # Apply the distance function to the entire cell_data
    cell_data['distance'] = cell_data.apply(calculate_distance, axis=1)
    
    # Sort by distance and return the nearest 15 points
    nearest_points = cell_data.nsmallest(num_points, 'distance')
    return nearest_points[['nwcbox_value', 'Layer#', 'Depth', 'Cell_ID', 'Latitude', 'Longitude','distance']]


# Function to retrieve time series data from the NetCDF file
def retrieve_time_series_data(nc_file, nearest_grids):
    with netCDF4.Dataset(nc_file) as dataset:
        # Retrieve available parameters
        time_series_data = {}
        
        # Directly retrieve the Date dimension
        date_data = dataset.variables['Date'][:]  # Retrieve date data
        
        # Convert date data to a string format
        date_data = [date for date in date_data]  # If needed, adjust the format

        # Loop through all nearest grid points
        for idx, row in nearest_grids.iterrows():
            cell_id = int(row['Cell_ID'])  # Convert to integer
            layer_n = int(row['Layer#'])  # Convert to integer
            
            # Store time series data for all available parameters
            for param in available_parameters:
                if param in dataset.variables.keys():  # Check if the parameter exists
                    if param not in time_series_data:
                        time_series_data[param] = []  # Initialize if not already present
                    
                    # Append data for the specific grid point
                    time_series_data[param].append(dataset.variables[param][cell_id, layer_n, :])

        # Convert lists to arrays for easier plotting
        for param in time_series_data:
            time_series_data[param] = np.array(time_series_data[param])

        # Add the date data to the time series data
        time_series_data['Date'] = date_data
        
        return time_series_data

def plot_time_series_data(time_series_data, nearest_grids, param_units, param, entered_depth_data=None):
    # Ensure 'Date' is in datetime format for time_series_data
    time_series_data['Date'] = pd.to_datetime(time_series_data['Date'], errors='coerce')

    # Retrieve unit for the current parameter
    unit = param_units.get(param, 'Unit not found')  # Default message if unit not found

    # Map the parameter to its corresponding value in time_series_data
    mapped_param = param_mapping.get(param, param)  # Get the mapped parameter or use original

    fig, ax = plt.subplots(figsize=(10, 5))

   
    ax.set_title(f'Time Series of {mapped_param} (Unit: {unit})')
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{mapped_param} ({unit})')

    # Plot each nearest grid point's data with a unique color
    for idx in range(len(nearest_grids)):
        if idx < time_series_data[mapped_param].shape[0]:  # Ensure idx is within bounds
            ax.plot(time_series_data['Date'], time_series_data[mapped_param][idx], 
                    label=f'Cell_ID: {int(nearest_grids.iloc[idx]["Cell_ID"])}, Layer: {int(nearest_grids.iloc[idx]["Layer#"])}',
                    linestyle='-', marker='')

    # Plot the entered depth data if provided, filtering by the current parameter
    if entered_depth_data is not None:
        # Ensure 'SampleDate' is in datetime format for entered_depth_data
        entered_depth_data['SampleDate'] = pd.to_datetime(entered_depth_data['SampleDate'], errors='coerce')

        # Filter the entered_depth_data for the current parameter
        filtered_depth_data = entered_depth_data[entered_depth_data['Parameter'] == param]

        if not filtered_depth_data.empty:
            ax.plot(filtered_depth_data['SampleDate'], filtered_depth_data['MeasureValue'], 
                    marker='o', color='red', linewidth=2, 
                    label=f'Entered Depth Data: {filtered_depth_data["Depth"].iloc[0]} m')

    # Display the legend and rotate x-axis labels for readability
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show(block=False)


def show_nearest_grids(lon, lat, depth, filtered_data, entered_depth_data, cell_data, param_units):
    nearest_grids = find_nearest_grids(lon, lat, depth)

    # Sort nearest grids by distance (assuming the distance is in the last column)
    nearest_grids = nearest_grids.sort_values(by='distance')

    # Create a new figure with specified subplots
    fig = plt.figure(figsize=(12, 8))
    ax_info = fig.add_subplot(221)  # Top left
    ax_table = fig.add_subplot(223)  # Bottom left
    ax_3d = fig.add_subplot(122, projection='3d')  # Right side

    # Prepare details of the selected observation point
    selected_details = f"Selected Point Details:\nLongitude: {lon}\nLatitude: {lat}\nDepth: {depth} m"
    
    # Display the selected point details in the first subplot
    ax_info.text(0.5, 0.5, selected_details, fontsize=12, ha='center', va='center', 
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgrey'))
    ax_info.axis('off')  # Hide the axes for the info subplot

    # Extract nearest grid data for the table
    grid_table_data = nearest_grids.values.tolist()
    col_labels = ['nwcbox_value', 'Layer#', 'Depth (m)', 'Cell_ID', 'Latitude', 'Longitude', 'Distance (m)']

    # Set the title of the table with the monitoring station name and coordinates
    station_name = filtered_data['MonitoringStation'].unique()[0]  # Assuming one station
    table_title = f"The Simulation Grid Information Near the Monitoring Station: {station_name} ({lon}, {lat})"
    ax_table.set_title(table_title, fontsize=10, pad=10)

    # Create a table of nearest grids
    ax_table.axis('tight')
    ax_table.axis('off')
    table = ax_table.table(cellText=grid_table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)  # Make table font size smaller
    table.scale(1, 1)  # Keep the table size moderate

    # Set the background color for each row based on the nearest point color
    num_points = nearest_grids.shape[0]
    colors = np.random.rand(num_points, 3)  # RGB colors
    for i in range(num_points):
        for j in range(len(col_labels)):
            table[(i + 1, j)].set_facecolor(colors[i])  # Set background color for all cells in the row

    # 3D Plot for all Cell2LonLat points and nearest grid points
    cell_points = ax_3d.scatter(cell_data['Longitude'], cell_data['Latitude'], -cell_data['Depth'], 
                                color='gray', alpha=0.2, label='CBP Simulation Grids')
    

    # Scatter plot for nearest grid points with corresponding colors
    for i in range(num_points):
        ax_3d.scatter(nearest_grids['Longitude'].iloc[i], nearest_grids['Latitude'].iloc[i], 
                      -nearest_grids['Depth'].iloc[i], color=colors[i], label=f'Near Simulation Grid {i+1}')

    # Highlight the selected point in red
    ax_3d.scatter(lon, lat, -depth, color='red', s=100, label='Monitoring Station')

    # Set axis limits based on the range of the data
    ax_3d.set_xlim([cell_data['Longitude'].min(), cell_data['Longitude'].max()])
    ax_3d.set_ylim([cell_data['Latitude'].min(), cell_data['Latitude'].max()])
    ax_3d.set_zlim([-cell_data['Depth'].max(), -cell_data['Depth'].min()])  # High to low for Depth

    ax_3d.set_xlabel('Longitude')
    ax_3d.set_ylabel('Latitude')
    ax_3d.set_zlabel('Depth (m)')  # Add unit to Depth
    ax_3d.set_title('3D Visualization of Points, Please Rotate')
    ax_3d.legend()

    # Button to toggle visibility of Cell2LonLat points
    button_ax = plt.axes([0.82, 0.95, 0.15, 0.05])
    toggle_button = Button(button_ax, 'Toggle All Simulation Grids')

    show_cell_points = True
    def toggle_points(event):
        nonlocal show_cell_points
        show_cell_points = not show_cell_points
        cell_points.set_alpha(0.2 if show_cell_points else 0.0)
        plt.draw()

    toggle_button.on_clicked(toggle_points)

    # Button to show layers near selected points
    button_layers_ax = plt.axes([0.82, 0.88, 0.15, 0.05])
    layers_button = Button(button_layers_ax, 'Show Simulation Layers Only Near the Obs Data')

    def show_nearest_layers(event):
        # Hide all Cell2LonLat points
        cell_points.set_alpha(0.0)
        
        # Find the layers in nearest_grids
        nearest_layers = nearest_grids['Layer#'].unique()
    
        # Filter Cell2LonLat data to keep only points from the same layers
        filtered_cell_data = cell_data[cell_data['Layer#'].isin(nearest_layers)]
    
        # Draw the filtered points in gray
        ax_3d.scatter(filtered_cell_data['Longitude'], filtered_cell_data['Latitude'], 
                      -filtered_cell_data['Depth'], color='gray', alpha=0.005, label='Same Layer Points')
    
        plt.draw()

    layers_button.on_clicked(show_nearest_layers)

    # Button to retrieve time series data
    button_ts_ax = plt.axes([0.82, 0.81, 0.15, 0.05])
    time_series_button = Button(button_ts_ax, 'Plot Time Series Data of the Obs and Near Simulation Grids')

    def show_time_series_data(event):
        # Load NetCDF file
        nc_file = '/Users/xiaoxug/CBP-Decipher/netCDF_output/All_Var.nc'
        
        # Retrieve time series data for all nearest grid points
        time_series_data = retrieve_time_series_data(nc_file, nearest_grids)
    
        # Ensure that entered_depth_data is accessible
        if entered_depth_data is not None and not entered_depth_data.empty:  # Check if it's available and not empty
            # Get the unique parameters from entered_depth_data
            unique_params = entered_depth_data['Parameter'].unique()
    
            # Loop through each unique parameter and plot in separate windows
            for param in unique_params:
                
                
                # Filter entered_depth_data for the current parameter
                param_depth_data = entered_depth_data[entered_depth_data['Parameter'] == param]
    
                if not param_depth_data.empty:
                    
                    # Plot the time series data for the specific parameter with corresponding entered_depth_data
                    plot_time_series_data(time_series_data, nearest_grids, param_units, param, param_depth_data)
                else:
                    print(f"No data available for parameter {param}.")
        else:
            print("Entered depth data is not available.")


    time_series_button.on_clicked(show_time_series_data)

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.8, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()



# Function to handle click events on the scatter plot
def on_click(event, record_counts, ax):
    if event.inaxes == ax and event.button is MouseButton.LEFT:
        # Get the clicked coordinates
        x_click, y_click = event.xdata, event.ydata
        
        # Get the closest point
        closest_idx = np.argmin(np.sqrt((record_counts['Longitude'] - x_click) ** 2 + 
                                          (record_counts['Latitude'] - y_click) ** 2))
        closest_point = record_counts.iloc[closest_idx]

        # Prepare details for the pop-up
        lon, lat = closest_point['Longitude'], closest_point['Latitude']
        count = closest_point['count']
        
        # Find the station name in obs_data based on the clicked longitude and latitude
        matching_station = obs_data[(obs_data['Longitude'] == lon) & (obs_data['Latitude'] == lat)]
    
        if not matching_station.empty:
            station_name = matching_station['MonitoringStation'].iloc[0]  # Assuming one matching station
        else:
            station_name = 'Unknown Station'
    
    

        # Get data for the selected point
        filtered_data = obs_data[(obs_data['Longitude'] == lon) & (obs_data['Latitude'] == lat)]
        time_range = f"{filtered_data['SampleDate'].min().strftime('%Y-%m-%d')} to {filtered_data['SampleDate'].max().strftime('%Y-%m-%d')}"

        # Count records per parameter and depth
        depth_parameter_counts = filtered_data.groupby(['Depth', 'Parameter']).size().unstack(fill_value=0)

        # Create a new figure for the clicked point
        detail_fig = plt.figure(figsize=(12, 8))
        
        # Upper subplot for text information
        ax_text = detail_fig.add_subplot(211)
        ax_text.axis('off')  # Turn off axes for the text plot

        ax_text.text(0.5, 0.9, f"Station: {station_name}\nLongitude: {lon}\nLatitude: {lat}\nRecords: {count}\nTime Range: {time_range}",
                     fontsize=12, ha='center')

        # Plot available parameters and their counts
        parameter_counts = filtered_data['Parameter'].value_counts()
        ax_text.text(0.5, 0.5, f"Available Parameters and Counts:\n" + 
                     "\n".join([f"{param}: {parameter_counts[param]}" for param in parameter_counts.index]),
                     fontsize=12, ha='center')

        # Lower subplot for the stacked bar plot
        ax_plot = detail_fig.add_subplot(212)

        # Plot stacked bar chart
        if not depth_parameter_counts.empty:
            depth_parameter_counts.plot(kind='bar', stacked=True, ax=ax_plot, 
                                        color=plt.cm.viridis(np.linspace(0, 1, depth_parameter_counts.shape[1])))
            ax_plot.set_title(f'Record Counts by Parameter at Each Depth - Monitoring Station:{station_name} ({lon}, {lat})')
            ax_plot.set_xlabel('Depth')
            ax_plot.set_ylabel('Total Record Counts')
            ax_plot.legend(title='Parameters', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax_plot.set_ylim(0, depth_parameter_counts.values.sum(axis=1).max() * 1.1)

        # Adjust spacing between text and plot
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

        # Create a box and button for entering depth below the plot
        ax_depth_input = plt.axes([0.15, 0.05, 0.5, 0.05])
        depth_input = TextBox(ax_depth_input, 'Enter depth for detailed analysis:', initial='')

        # Adjust the button placement
        button_ax = plt.axes([0.7, 0.05, 0.15, 0.05])
        detail_button = Button(button_ax, 'Go To Detailed Plots at this Monitoring Station')

        grid_button_ax = plt.axes([0.7, 0.01, 0.15, 0.05])  # Adjust the second button below the first
        grid_button = Button(grid_button_ax, 'Show Nearest Simulation Grids Near This Monitoring Station')

        # Initialize variable to hold entered depth data
        entered_depth_data = None
        param_units = filtered_data[['Parameter', 'Unit']].drop_duplicates().set_index('Parameter')['Unit'].to_dict()

        # Update entered_depth_data based on user input when the button is clicked
        def update_entered_depth_data():
            nonlocal entered_depth_data
            entered_depth = depth_input.text
            try:
                entered_depth = float(entered_depth)
                entered_depth_data = filtered_data[filtered_data['Depth'] == entered_depth]
                if not entered_depth_data.empty:
                    entered_depth_data.sort_values(by='SampleDate', inplace=True)
                    print(f"Entered Depth Data loaded for depth {entered_depth}.")
                else:
                    print(f"No data found for depth {entered_depth}")
            except ValueError:
                print("Invalid depth entered")

        # Call the update function when the input changes
        depth_input.on_submit(lambda _: update_entered_depth_data())

        # Connect button click event to update depth data
        detail_button.on_clicked(update_entered_depth_data)

        # Function to show line plots for specific depth
        def show_details(event):
            if entered_depth_data is not None and not entered_depth_data.empty:
                line_fig, axes = plt.subplots(nrows=len(entered_depth_data['Parameter'].unique()), 
                                               ncols=1, figsize=(12, 8), sharex=True)
                parameters = entered_depth_data['Parameter'].unique()
                colors = plt.cm.viridis(np.linspace(0, 1, len(parameters)))

                for i, param in enumerate(parameters):
                    param_data = entered_depth_data[entered_depth_data['Parameter'] == param]
                    ax = axes[i]
                    ax.plot(param_data['SampleDate'], param_data['MeasureValue'], 
                             marker='o', color=colors[i], label=f'{param} (Unit: {param_data["Unit"].iloc[0]})')
                    ax.set_title(f'{param} vs SampleDate at Depth {depth_input.text} m')
                    ax.set_xlabel('SampleDate')
                    ax.set_ylabel('MeasureValue')
                    ax.legend(loc='upper right')

                plt.show()

        # Connect detail_button to show_details function
        detail_button.on_clicked(show_details)

        # Connect the grid button to display nearest grids
        grid_button.on_clicked(lambda event: show_nearest_grids(lon, lat, float(depth_input.text), filtered_data, entered_depth_data, cell_data, param_units))

        plt.show()


# Main function to create the scatter plot and interface
def main():
    global fig
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot observation data on scatter plot
    plot_obs_data(ax)

    # Add a button for global analysis
    #ax_button = plt.axes([0.8, 0.02, 0.1, 0.05])
    #global_button = Button(ax_button, 'Global Analysis')
    #global_button.on_clicked(show_global_analysis)

    plt.show()

# Call the main function to run the application
main()
