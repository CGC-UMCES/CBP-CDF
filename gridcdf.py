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
nc_file = '/Users/xiaoxug/CBP-Decipher/netCDF_output/S.nc'
nc_data = nc.Dataset(nc_file, 'r')

# Identify variables to exclude
excluded_vars = {'PosX', 'PosY', 'Depth', 'nwcbox'}

# Identify the variables for time and the other variable
time_var = None
data_var = None

for var_name in nc_data.variables:
   
    if var_name not in excluded_vars:
        if 'Date' in var_name:
            time_var = var_name           
        else:
            data_var = var_name

if not time_var or not data_var:
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
    data_var_data = nc_data.variables[f'{data_var}'][:]
   
    # Fetch the data for the selected point and Jday range
    selected_cell_index = selected_cell_id
    
    if selected_cell_index.size > 0:
        selected_cell_index = selected_cell_index
        if 0 <= jday_start < len(time_data) and 0 <= jday_end < len(time_data):
            data_values = data_var_data[selected_cell_index,selected_layer ,jday_start:jday_end+1]
            time_values = time_data[selected_cell_index, selected_layer,jday_start:jday_end+1]
           
            data_var_unit = nc_data.variables[f'{data_var}'].units
            
            # Get nearest points
            nearest_points = get_nearest_points_within_layer((x, y), layer)
            
            # Plot for the selected point
            plt.figure(figsize=(15, 6))
            
            # Time series plot
            ax1 = plt.subplot(121)
            line_colors = plt.cm.jet(np.linspace(0, 1, len(nearest_points)))
            plt.plot(time_values, data_values, marker='o', color='red', label=f'Selected Point ({x}, {y})')

            for i, value in enumerate(data_values):
                plt.annotate(f'{value:.2f}', (time_values[i], data_values[i]), textcoords="offset points", xytext=(0, 5), ha='center')

            for idx, (_, point) in enumerate(nearest_points.iterrows()):
                cell_index = point['Cell_ID']
                if cell_index.size > 0:
                    cell_index = int(cell_index)
                    point_data_values = data_var_data[cell_index,layer, jday_start:jday_end+1]
                    plt.plot(time_values, point_data_values, marker='x', linestyle='--', color=line_colors[idx], label=f'Near Point {idx + 1} ({point["Longitude"]}, {point["Latitude"]})')

                    # Annotate the data points for nearest points
                    for i, value in enumerate(point_data_values):
                        plt.annotate(f'{value:.2f}', (time_values[i], point_data_values[i]), textcoords="offset points", xytext=(0, 5), ha='center')

            plt.xlabel('Date')
            plt.ylabel(f'{data_var} ({data_var_unit})')
            plt.title(f'{data_var} vs Date for Selected and Nearest Points')
            plt.grid(True)
            plt.legend()

            # Scatter plot
            ax2 = plt.subplot(122)
            scatter_colors = [line_colors[i] for i in range(len(nearest_points))]
            ax2.scatter(nearest_points['Longitude'], nearest_points['Latitude'], c=scatter_colors)
            ax2.scatter(x, y, color='red', label='Selected Point')

            for i, (ix, iy) in enumerate(zip(nearest_points['X'], nearest_points['Y'])):
                ax2.text(ix, iy+0.5, f'Near Point {i + 1} ({ix:.2f}, {iy:.2f})', fontsize=9, ha='center',verticalalignment='bottom', color=scatter_colors[i])
            
            ax2.text(x, y, f'Selected Point ({x:.2f}, {y:.2f})', fontsize=9, ha='center',verticalalignment='bottom', color='red')
            ax2.set_xlabel('Longitude')
            ax2.set_ylabel('Latitude')
            ax2.set_title('Scatter Plot of Nearest Points and Selected Point')

            plt.tight_layout()
            plt.show()
        else:
            print("Jday range is out of bounds.")
    else:
        print("Cell_ID not found in NetCDF file.")





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
