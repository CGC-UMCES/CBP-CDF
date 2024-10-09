#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 12:03:45 2024

@author: xiaoxug
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 09:46:46 2024

@author: xiaoxug
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import TextBox
from scipy.spatial import distance
import netCDF4 as nc
import textwrap

# Load your data from CSV
data = pd.read_csv('Cell2LonLat.csv')

# Check if the necessary columns are present
if not all(col in data.columns for col in ['Latitude', 'Longitude', 'Cell_ID', 'Layer#']):
    raise ValueError("CSV file must contain 'Latitude', 'Longitude', 'Cell_ID', and 'Layer#' columns.")

x = data['Longitude']
y = data['Latitude']
cell_id = data['Cell_ID']
layer = data['Layer#']

# Compute the total number of layers at each (X, Y) coordinate
layer_counts = data.groupby(['Longitude', 'Latitude']).size()

# Load NetCDF file
nc_file = '/Users/xiaoxug/CBP-Decipher/netCDF_output/All_Var.nc'
nc_data = nc.Dataset(nc_file, 'r')

# Identify variables to exclude
excluded_vars = {'PosX', 'PosY', 'Depth', 'nwcbox'}

# Identify the variables for time and the other variable
time_var = None
data_vars = []  # Change to a list to hold multiple data variables

for var_name in nc_data.variables:
    if var_name not in excluded_vars:
        if 'Date' in var_name:
            time_var = var_name           
        else:
            data_vars.append(var_name)  # Append to the list of data variables


if not time_var or not data_vars:
    raise ValueError("Time variable or data variable not found in NetCDF file.")



def get_nearest_points_within_layer(xy, current_layer, num_points=6):
    # Filter data to only include points from the same layer
    layer_data = data[data['Layer#'] == current_layer]
    
    # Extract coordinates for the current layer
    x_layer = layer_data['Longitude'].values
    y_layer = layer_data['Latitude'].values
    
    # Compute distances within the same layer
    points = np.column_stack((x_layer, y_layer))
    distances = distance.cdist([xy], points)[0]
    
    # Get indices of the nearest points within the layer
    nearest_indices = np.argsort(distances)[:num_points]
    
    # Exclude the selected point if it is among the nearest points
    if xy in points[nearest_indices]:
        nearest_indices = nearest_indices[nearest_indices != np.where((points == xy).all(axis=1))[0][0]]
    
    return layer_data.iloc[nearest_indices]

def get_nc_variable_info_at_point(x, y, layer, jday_start, jday_end):
    time_data = nc_data.variables[f'{time_var}'][:]
    default_vars = ['T', 'S', 'DO']
    
    # Prompt user for additional variables to plot
    additional_vars = input("Enter additional variables to plot (comma-separated, e.g., 'O2, Chl'): ").split(',')
    additional_vars = [var.strip() for var in additional_vars if var.strip()]  # Clean input

    # Combine default and additional variables
    variables_to_plot = default_vars + additional_vars
    
    num_vars = len(variables_to_plot)
    num_cols = 4  # You can set this to the desired number of columns per row
    num_rows = (num_vars + num_cols - 1) // num_cols  # Calculate required number of rows

    # Add 1 row for scatter plot and coordinate info
    total_rows = num_rows + 1
    
    fig, axs = plt.subplots(total_rows, num_cols, figsize=(5 * num_cols, 4 * total_rows))
    axs = axs.flatten()  # Flatten axes array for easy indexing

    # Find nearest points within the specified layer
    nearest_points = get_nearest_points_within_layer((x, y), layer)
    
    # Color mapping for nearest points
    line_colors = plt.cm.jet(np.linspace(0, 1, len(nearest_points)))
    
    # Plot data for each variable
    for idx, data_var in enumerate(variables_to_plot):
        if data_var not in nc_data.variables:
            print(f"Variable '{data_var}' not found in NetCDF file.")
            continue

        data_var_data = nc_data.variables[data_var][:]
        selected_cell_index = selected_cell_id

        if selected_cell_index.size > 0:
            if 0 <= jday_start < len(time_data) and 0 <= jday_end < len(time_data):
                data_values = data_var_data[selected_cell_index, selected_layer, jday_start:jday_end+1]
                time_values = time_data[jday_start:jday_end+1]
                data_var_unit = nc_data.variables[f'{data_var}'].units

                ax = axs[idx]
                ax.plot(time_values, data_values, marker='o', color='red', label=f'Selected Point')

                # Annotate selected point
                for i, value in enumerate(data_values):
                    ax.annotate(f'{value:.2f}', (time_values[i], data_values[i]), textcoords="offset points", xytext=(0, 5), ha='center')

                # Plot nearest points without (Lon, Lat) in the legend
                for point_idx, (_, point) in enumerate(nearest_points.iterrows()):
                    cell_index = point['Cell_ID']
                    if cell_index.size > 0:
                        cell_index = int(cell_index)
                        point_data_values = data_var_data[cell_index, layer, jday_start:jday_end+1]
                        ax.plot(time_values, point_data_values, marker='x', linestyle='--', color=line_colors[point_idx], label=f'Near {point_idx + 1}')

                        # Annotate nearest points
                        for i, value in enumerate(point_data_values):
                            ax.annotate(f'{value:.2f}', (time_values[i], point_data_values[i]), textcoords="offset points", xytext=(0, 5), ha='center')

                ax.set_xlabel('Date')
                ax.set_ylabel(f'{data_var} ({data_var_unit})')
                ax.set_title(f'{data_var} vs Date')
                ax.grid(True)
                ax.legend()

            else:
                print("Jday range is out of bounds.")
        else:
            print("Cell_ID not found in NetCDF file.")

    # Add placeholders for empty subplots if any
    for empty_idx in range(num_vars, len(axs) - 2):  # Reserve the last two subplots for scatter plot and coordinates
        axs[empty_idx].set_visible(True)
        axs[empty_idx].text(0.5, 0.5, 'Placeholder', ha='center', va='center', fontsize=12, color='gray')
        axs[empty_idx].set_xticks([])
        axs[empty_idx].set_yticks([])

    # Scatter plot on the last row of the grid (second last subplot)
    ax_scatter = axs[-2]  # Use the second last subplot for the scatter plot
    scatter_colors = [line_colors[i] for i in range(len(nearest_points))]
    ax_scatter.scatter(nearest_points['Longitude'], nearest_points['Latitude'], c=scatter_colors)
    ax_scatter.scatter(x, y, color='red', label='Selected Point')
    
    for i, (ix, iy) in enumerate(zip(nearest_points['Longitude'], nearest_points['Latitude'])):
        ax_scatter.text(ix, iy, f'Near {i + 1}', fontsize=9, ha='center', va='bottom', color=scatter_colors[i])
    
    ax_scatter.set_xlabel('Longitude')
    ax_scatter.set_ylabel('Latitude')
    ax_scatter.set_title('Scatter Plot of Nearest Points and Selected Point')
    ax_scatter.grid(True)

    # Display Longitude and Latitude info in the last subplot
    ax_coords = axs[-1]  # Use the last subplot for the (Lon, Lat) information
    ax_coords.axis('off')  # Turn off axis for this subplot
    
    # Create coordinate info with matching colors
    coord_info = ""
    for i, (point, color) in enumerate(zip(nearest_points.iterrows(), scatter_colors)):
        row = point[1]
        coord_info += f"\nNear {i + 1}: Lon {row['Longitude']}, Lat {row['Latitude']}"
        ax_coords.text(0.5, 1 - (i + 1) * 0.05, f'Near {i + 1}: Lon {row["Longitude"]:.6f}, Lat {row["Latitude"]:.6f}', 
                       ha='center', va='center', fontsize=10, color=color)
    
    # Add selected point information
    ax_coords.text(0.5, -0.05, f'Selected Point: Lon {x}, Lat {y}', ha='center', va='center', fontsize=10, color='red')
    
    plt.tight_layout()
    plt.show()



def update_layer(layer_index):
    global sc, cbar

    # Clear the left panel and reinitialize
    ax_left.clear()

    # Filter data for the current layer
    layer_data = data[data['Layer#'] == layer_index]

    # Replot the scatter plot for the current layer
    sc = ax_left.scatter(layer_data['Longitude'], layer_data['Latitude'],
                         c=[layer_counts.get((xi, yi), 0) for xi, yi in zip(layer_data['Longitude'], layer_data['Latitude'])],
                         cmap='viridis', s=10, alpha=0.7)
    ax_left.set_xlabel('Longitude')
    ax_left.set_ylabel('Latitude')
    ax_left.set_title(f'Chespeake Bay Simulation Cell for Layer {layer_index}')

    # Create or update colorbar
    if cbar is None:
        cbar = plt.colorbar(sc, ax=ax_left)
        cbar.set_label('Total Number of Layers')
    else:
        cbar.update_normal(sc)

    # Update the info panel for the current layer
    info_panel.clear()
    info_panel.axis('off')

    fig.canvas.draw_idle()

def on_click(event):
    global selected_x, selected_y, selected_layer, selected_cell_id

    if event.inaxes == ax_left:
        distances = np.sqrt((x - event.xdata) ** 2 + (y - event.ydata) ** 2)
        nearest_idx = distances.idxmin()
        
        selected_x = x.iloc[nearest_idx]
        selected_y = y.iloc[nearest_idx]
        selected_layer = layer.iloc[nearest_idx]
        selected_cell_id = cell_id.iloc[nearest_idx]

        selected_data = data[(data['Longitude'] == selected_x) & (data['Latitude'] == selected_y)]
        nearest_points = get_nearest_points_within_layer((selected_x, selected_y), selected_layer)
        
        info_panel.clear()
        info_panel.axis('off')

       # Wrap the text to fit within two or three lines
        wrapped_text = textwrap.fill(f'All Layers at Selected (Longitude, Latitude): ({selected_x}, {selected_y})', width=40)
        info_panel.text(0.05, 0.95, wrapped_text, fontsize=12, weight='bold', va='top')



        #info_panel.text(0.05, 0.95, f'All Layers at Selected (Longitude-UTM, Latitude-UTM): ({selected_x}, {selected_y})', fontsize=12, weight='bold')
        y_offset = 0.8
        for i, row in selected_data.iterrows():
            info_panel.text(0.05, y_offset, f'Layer: {row["Layer#"]}, Cell ID: {row["Cell_ID"]}', fontsize=10)
            y_offset -= 0.05
        
        info_panel.text(0.55, 0.95, f'Nearest 6 Points in Layer {selected_layer}:', fontsize=12, weight='bold')
        y_offset = 0.90
        for i, row in nearest_points.iterrows():
            info_panel.text(0.55, y_offset, f'Longitude: {row["Longitude"]}, Latitude: {row["Latitude"]}, Layer: {row["Layer#"]}, Cell ID: {row["Cell_ID"]}', fontsize=10)
            y_offset -= 0.05

        try:
            jday_range = input("Enter Jday range (start,end): ")
            jday_start, jday_end = map(int, jday_range.split(','))
            get_nc_variable_info_at_point(selected_x, selected_y, selected_layer, jday_start, jday_end)
        except ValueError as e:
            print(f"Error: {e}")

        info_panel.set_xlim(0, 1)
        info_panel.set_ylim(0, 1)
        fig.canvas.draw_idle()

def submit_layer(text):
    try:
        layer_index = int(text)
        if layer_index in layer.unique():
            update_layer(layer_index)
        else:
            print(f"Layer {layer_index} not found.")
    except ValueError:
        print("Invalid input. Please enter a valid integer.")



fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 3, width_ratios=[4, 1, 2], height_ratios=[1, 4])

ax_left = fig.add_subplot(gs[1, 0])
info_panel = fig.add_subplot(gs[1, 1:])
info_panel.axis('off')

text_box_ax = plt.axes([0.35, 0.02, 0.15, 0.05])
text_box = TextBox(text_box_ax, 'Enter Layer: ')

sc = None
cbar = None

text_box.on_submit(submit_layer)

update_layer(layer.unique()[0])

fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
