import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import networkx as nx
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

import plotly.graph_objects as go
import plotly.express as px

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
# Path to the HDF5 file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define a relative path based on the current script's directory
file_path = os.path.join(current_dir, "WinNet_data.h5")
#file_path = '/home/ryan/University/Honours/Semester_2/Project/WinNet/runs/x-ray_burst/WinNet_data.h5'


with h5py.File(file_path, 'r') as hdf:
    # Access the 'snapshots' group
    snapshots_group = hdf['snapshots']

    # Extract the time, Z (atomic number), and Y (abundance) datasets
    time_data = snapshots_group['time'][:]
    A_data = snapshots_group['A'][:]
    N_data = snapshots_group['N'][:]
    Y_data = snapshots_group['Y'][:]
    Z_data = snapshots_group['Z'][:]

# List of element symbols up to Z=118
element_symbols = [
    'n',  # Z=0 (neutron)
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',      # Z=1-10
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',   # Z=11-20
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', # Z=21-30
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', # Z=31-40
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',# Z=41-50
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', # Z=51-60
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',# Z=61-70
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', # Z=71-80
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',# Z=81-90
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', # Z=91-100
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',# Z=101-110
    'Rg', 'Cn', 'Fl', 'Lv', 'Ts', 'Og'                          # Z=111-118
]

def get_element_symbol(Z):
    if Z >= 0 and Z < len(element_symbols):
        return element_symbols[int(Z)]
    else:
        return 'Unknown'

# Read data from HDF5 file (assuming you've already loaded the datasets)
# If not, include the code to load Z_data, N_data, A_data, Y_data, time_data

# Ensure that Y_data is a 2D array
if Y_data.ndim == 1:
    Y_data = Y_data.reshape(-1, len(Z_data))

# Get the abundances at the final time step
Y_final = Y_data[-1, :]  # Shape: (number_of_nuclides,)

# Create a DataFrame with the isotope information, including 'Index'
data = pd.DataFrame({
    'Index': np.arange(len(Z_data)),  # Include the index
    'Z': Z_data,
    'N': N_data,
    'A': A_data,
    'Abundance': Y_final
})

# Add element symbols and isotope labels
data['Element'] = data['Z'].apply(get_element_symbol)
data['Isotope'] = data['Element'] + '-' + data['A'].astype(str)

# Sort the data by abundance in descending order
data_sorted = data.sort_values(by='Abundance', ascending=False)







##TODO: Which Nuclides change when?

# Define the time limit and threshold for significant change
time_limit = 250  # Seconds
significant_change_threshold = 0.05  # 10% change

# Filter data to include only the time points before the defined time limit
indices_within_time = time_data <= time_limit
time_filtered = time_data[indices_within_time]
Y_filtered = Y_data[indices_within_time, :]  # Shape: (num_filtered_time_steps, num_nuclides)

# Initialize lists to store nuclides that change and do not change
changing_nuclides = []
stable_nuclides = []

# Iterate over each nuclide (each column in Y_filtered)
for i in range(Y_filtered.shape[1]):
    abundance_over_time = Y_filtered[:, i]

    # Calculate the relative change from the first to the last time point within the time limit
    initial_abundance = abundance_over_time[0]
    final_abundance = abundance_over_time[-1]
    
    # Avoid division by zero; if the initial abundance is 0, we use absolute change
    if initial_abundance == 0:
        change = np.abs(final_abundance - initial_abundance)
    else:
        change = np.abs((final_abundance - initial_abundance) / initial_abundance)

    # Check if the change is significant
    if change >= significant_change_threshold:
        changing_nuclides.append(data.iloc[i]['Isotope'])
    else:
        stable_nuclides.append(data.iloc[i]['Isotope'])

# Convert lists to DataFrames for better visualization
changing_nuclides_df = pd.DataFrame(changing_nuclides, columns=['Nuclide'])
stable_nuclides_df = pd.DataFrame(stable_nuclides, columns=['Nuclide'])

# Print results
print("Nuclides that change significantly before 300 seconds:")
print(changing_nuclides_df)

print("\nNuclides that do not change significantly before 300 seconds:")
print(stable_nuclides_df)





##Clusters??##

# Ensure Y_data is a 2D array with shape (num_time_steps, num_nuclides)
if Y_data.ndim == 1:
    Y_data = Y_data.reshape(-1, len(Z_data))

# Remove nuclides with abundance zero throughout the entire time period
non_zero_indices = [i for i in range(Y_data.shape[1]) if not np.all(Y_data[:, i] == 0)]
Y_data = Y_data[:, non_zero_indices]

# Create a DataFrame with nuclide information
data = pd.DataFrame({
    'Index': np.arange(len(non_zero_indices)),
    'Z': Z_data[non_zero_indices],
    'N': N_data[non_zero_indices],
    'A': A_data[non_zero_indices]
})

# Add element symbols and isotope labels
element_symbols = [
    'n', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]

def get_element_symbol(Z):
    if 0 <= Z < len(element_symbols):
        return element_symbols[int(Z)]
    else:
        return 'Unknown'

data['Element'] = data['Z'].apply(get_element_symbol)
data['Isotope'] = data['Element'] + '-' + data['A'].astype(str)



# Use log-transformed abundances to handle wide range of values
Y_log = np.log10(Y_data + 1e-20)  # Add small value to avoid log(0)

# Transpose Y_log to have nuclides as samples (rows) and time points as features (columns)
# Shape: (num_nuclides, num_time_steps)
Y_log_T = Y_log.T

# Standardize the data (mean=0, variance=1)
scaler = StandardScaler()
Y_scaled = scaler.fit_transform(Y_log_T)

# Determine the optimal number of clusters using the Silhouette score
silhouette_scores = []
K_range = range(2, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(Y_scaled)
    score = silhouette_score(Y_scaled, cluster_labels)
    silhouette_scores.append(score)

# Plot Silhouette scores to choose k
plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis For Optimal k')
plt.show()

# Choose k based on the highest silhouette score
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters: {optimal_k}")

#TODO:choose K
#TODO:choose K
optimal_k = 8

# Perform K-Means clustering with optimal_k
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(Y_scaled)

# Add cluster labels to the DataFrame
data['Cluster'] = clusters

# Plot abundances of nuclides in each cluster
from itertools import cycle

# Define a color map with a set number of colors
base_colors = plt.cm.get_cmap('tab20').colors  # 'tab20' provides 20 distinct colors
color_cycle = cycle(base_colors)  # Use cycle to allow repeating of colors

for cluster_id in range(optimal_k):
    cluster_indices = data[data['Cluster'] == cluster_id]['Index'].values
    num_nuclides = len(cluster_indices)

    plt.figure(figsize=(14, 11.5))

    # Use the color cycle to assign colors
    for idx, nuclide_idx in enumerate(cluster_indices):
        color = next(color_cycle)
        plt.plot(time_data, Y_data[:, nuclide_idx], label=data.iloc[nuclide_idx]['Isotope'], color=color)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Abundance (Y)')
    plt.title(f'Nuclide Abundances in Cluster {cluster_id+1}')
    plt.yscale('log')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize='small', ncol=3)
    plt.tight_layout()
    plt.show()


#TODO: interactive plots
# Plot abundances of nuclides in each cluster interactively
# for cluster_id in range(optimal_k):
#     cluster_indices = data[data['Cluster'] == cluster_id]['Index'].values
#     fig = go.Figure()
#     for idx in cluster_indices:
#         color = cmap(idx)
#         fig.add_trace(go.Scatter(x=time_data, y=Y_data[:, idx], mode='lines', name=data.iloc[idx]['Isotope']),color = color)
#     fig.update_layout(
#         title=f'Nuclide Abundances in Cluster {cluster_id}',
#         xaxis_title='Time (s)',
#         yaxis_title='Abundance (Y)',
#         yaxis_type='log',
#         legend_title='Isotopes'
#     )
#     fig.show()
for cluster_id in range(optimal_k):
    cluster_indices = data[data['Cluster'] == cluster_id]['Index'].values
    num_nuclides = len(cluster_indices)
    cmap = plt.cm.get_cmap('tab20', num_nuclides)  # Repeat colors if needed

    fig = go.Figure()
    for idx, nuclide_idx in enumerate(cluster_indices):
        color = cmap(idx)  # Get the color from the colormap
        rgba_color = f'rgba({color[0]*255}, {color[1]*255}, {color[2]*255}, {color[3]})'  # Convert to rgba for Plotly

        fig.add_trace(go.Scatter(
            x=time_data,
            y=Y_data[:, nuclide_idx],
            mode='lines',
            name=data.iloc[nuclide_idx]['Isotope'],
            line=dict(color=rgba_color)
        ))

    fig.update_layout(
        title=f'Nuclide Abundances in Cluster {cluster_id+1}',
        xaxis_title='Time (s)',
        yaxis_title='Abundance (Y)',
        yaxis_type='log',
        legend_title='Isotopes'
    )
    fig.show()

# TODO:Perform hierarchical clustering
linkage_matrix = linkage(Y_scaled, method='ward')

# Plot dendrogram for hierarchical clustering
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, labels=data['Isotope'].values, leaf_rotation=90, leaf_font_size=8)
plt.xlabel('Nuclides')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.tight_layout()
plt.show()


optimal_k = 8
# Perform Agglomerative Clustering with optimal_k
agg_clustering = AgglomerativeClustering(n_clusters=optimal_k, affinity='euclidean', linkage='ward')
agg_clusters = agg_clustering.fit_predict(Y_scaled)

# Add Agglomerative Clustering labels to the DataFrame
data['Agg_Cluster'] = agg_clusters


# Plot abundances of nuclides in each Agglomerative cluster
for cluster_id in range(optimal_k):
    cluster_indices = data[data['Agg_Cluster'] == cluster_id]['Index'].values
    num_nuclides = len(cluster_indices)
    cmap = plt.cm.get_cmap('tab20', num_nuclides) 

    plt.figure(figsize=(14, 12))
    for idx in cluster_indices:
        color = cmap(idx)
        plt.plot(time_data, Y_data[:, idx], label=data.iloc[idx]['Isotope'],color=color)
    plt.xlabel('Time (s)')
    plt.ylabel('Abundance (Y)')
    plt.title(f'Nuclide Abundances in Agglomerative Cluster {cluster_id+1}')
    plt.yscale('log')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize='small', ncol=3)
    plt.tight_layout()
    plt.show()


#TODO: interactive plots
# Plot abundances of nuclides in each Agglomerative cluster interactively
for cluster_id in range(optimal_k):
    cluster_indices = data[data['Agg_Cluster'] == cluster_id]['Index'].values
    num_nuclides = len(cluster_indices)
    cmap = plt.cm.get_cmap('tab20', num_nuclides)  # Use the same color map for all plots

    fig = go.Figure()
    for idx in cluster_indices:
        color = cmap(idx)  # Get the color from the colormap
        rgba_color = f'rgba({color[0]*255}, {color[1]*255}, {color[2]*255}, {color[3]})'  # Convert to rgba for Plotly

        fig.add_trace(go.Scatter(
            x=time_data,
            y=Y_data[:, idx],
            mode='lines',
            name=data.iloc[idx]['Isotope'],
            line=dict(color=rgba_color)  # Set the color in Plotly
        ))

    fig.update_layout(
        title=f'Nuclide Abundances in Agglomerative Cluster {cluster_id+1}',
        xaxis_title='Time (s)',
        yaxis_title='Abundance (Y)',
        yaxis_type='log',
        legend_title='Isotopes'
    )
    fig.show()




##Reaction Graph Network##

# Compute time differences
delta_time = np.diff(time_data)  # Shape: (num_time_steps - 1,)

# Compute abundance differences
delta_Y = np.diff(Y_data, axis=0)  # Shape: (num_time_steps - 1, num_nuclides)

# Avoid division by zero
delta_time[delta_time == 0] = 1e-20

# Compute approximate time derivatives of abundances
dY_dt = delta_Y / delta_time[:, np.newaxis]  # Shape: (num_time_steps - 1, num_nuclides)

# Create a DataFrame for nuclide information
nuclide_df = pd.DataFrame({
    'Index': np.arange(len(Z_data)),
    'Z': Z_data,
    'N': N_data,
    'A': A_data
})

# Add element symbols and isotope labels
element_symbols = [
    'n', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Fl', 'Lv', 'Ts', 'Og'
]

def get_element_symbol(Z):
    if 0 <= Z < len(element_symbols):
        return element_symbols[int(Z)]
    else:
        return 'Unknown'

nuclide_df['Element'] = nuclide_df['Z'].apply(get_element_symbol)
nuclide_df['Isotope'] = nuclide_df['Element'] + '-' + nuclide_df['A'].astype(str)

# Function to find the index of a nuclide given Z and N
def find_nuclide_index(Z, N):
    matches = nuclide_df[(nuclide_df['Z'] == Z) & (nuclide_df['N'] == N)]
    if not matches.empty:
        return matches['Index'].values[0]
    else:
        return None

# Initialize the graph
G = nx.DiGraph()

# Filter nuclides with final abundance above a threshold
abundance_threshold = 1e-5
final_abundances = Y_data[-1, :]
selected_indices = np.where(final_abundances > abundance_threshold)[0]

# Add nodes for selected nuclides
for idx in selected_indices:
    row = nuclide_df.iloc[idx]
    G.add_node(row['Index'], label=row['Isotope'], Z=row['Z'], N=row['N'], A=row['A'])

# Estimate reaction flows
num_time_steps = dY_dt.shape[0]

# Initialize a dictionary to accumulate flows
flow_dict = {}

for t in range(num_time_steps):
    for idx in selected_indices:
        Z = Z_data[idx]
        N = N_data[idx]
        Y_dot = dY_dt[t, idx]  # Net change in abundance for nuclide idx at time t

        # Skip if net change is negligible
        if np.abs(Y_dot) < 1e-20:
            continue

        # Consider possible reactions
        reactions = [
            {'dZ': 1, 'dN': -1},  # Beta-minus decay
            {'dZ': -1, 'dN': 1},  # Beta-plus decay
            {'dZ': 0, 'dN': 1},   # Neutron capture
            {'dZ': 1, 'dN': 0},   # Proton capture
            {'dZ': -2, 'dN': -2}  # Alpha decay
        ]

        for reaction in reactions:
            daughter_Z = Z + reaction['dZ']
            daughter_N = N + reaction['dN']
            daughter_idx = find_nuclide_index(daughter_Z, daughter_N)
            if daughter_idx is not None and daughter_idx in selected_indices:
                key = (idx, daughter_idx)
                flow = Y_dot * delta_time[t]
                flow_dict[key] = flow_dict.get(key, 0) + flow

# Add edges to the graph
for (parent_idx, daughter_idx), flow in flow_dict.items():
    if flow > 0:
        G.add_edge(parent_idx, daughter_idx, weight=flow)

# Map node indices to labels
node_labels = {idx: data['label'] for idx, data in G.nodes(data=True)}

# Node sizes based on final abundances
node_sizes = []
for idx in G.nodes():
    abundance = Y_data[-1, idx]
    node_sizes.append(abundance * 1e6)  # Adjust scale as needed

# Edge widths based on accumulated flows
edge_widths = []
for u, v, data in G.edges(data=True):
    edge_widths.append(data['weight'] * 1e6)  # Adjust scale as needed

plt.figure(figsize=(14, 12))

# Use a layout for better visualization
pos = nx.spring_layout(G, k=0.5, iterations=50)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue')

# Draw edges
nx.draw_networkx_edges(G, pos, width=edge_widths, arrows=True, arrowstyle='->', arrowsize=10)

# Draw labels
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

plt.title('Reaction Flow Network Diagram')
plt.axis('off')
plt.tight_layout()
plt.show()






###ANIMATION###
# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Find the max Z and N to set the axis limits
max_Z = np.max(Z_data)
max_N = np.max(N_data)

# Create a grid for the nuclide chart, initialized to zeros (log scale of abundances)
grid = np.zeros((max_Z + 1, max_N + 1))

# Define the imshow object outside of the update function
im = ax.imshow(grid, aspect='equal', cmap='plasma', norm=LogNorm(vmin=1e-12, vmax=1))

# Set the aspect ratio of the axis to be equal
ax.set_aspect('equal')

# Add the colorbar only once
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Abundance (Y)')



# Add neutron, proton driplines, and valley of stability
def add_driplines():
    # Neutron dripline (approximate values)
    neutron_dripline_N = [
        1, 2, 3, 4, 6, 8, 10, 14, 16, 20, 24, 28, 32, 36, 40, 44, 50,
        54, 58, 64, 70, 76, 82, 90, 100, 110, 120, 130, 140, 150
    ]
    neutron_dripline_Z = [
        1, 2, 3, 4, 5, 6, 7, 8, 8, 10, 10, 12, 12, 14, 16, 16, 20,
        20, 22, 24, 26, 28, 30, 34, 36, 38, 42, 44, 48, 50
    ]
    
    # Proton dripline (approximate values)
    proton_dripline_N = [
        1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 22, 26, 30, 34, 38,
        42, 48, 54, 60, 68, 76, 84, 92, 100, 110, 120, 130, 140
    ]
    proton_dripline_Z = [
        1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28,
        30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54
    ]

    # Valley of stability (approximate values)
    valley_stability_N = [
        1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 26, 28, 32, 36,
        40, 44, 50, 54, 58, 64, 70, 76, 82, 90, 100, 110, 120, 130
    ]
    valley_stability_Z = [
        1, 2, 3, 4, 6, 8, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 30,
        34, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84
    ]

    # Plot neutron dripline
    ax.plot(neutron_dripline_N, neutron_dripline_Z, 'b-', label='Neutron Dripline')
    
    # Plot proton dripline
    ax.plot(proton_dripline_N, proton_dripline_Z, 'r-', label='Proton Dripline')

    # Plot valley of stability
    ax.plot(valley_stability_N, valley_stability_Z, 'g--', label='Valley of Stability')

    # Add the legend for driplines and valley of stability
    ax.legend()



# Plot initial state (time 0)
def init():
    ax.set_xlim(0, max_N + 1)
    ax.set_ylim(0, max_Z + 1)
    ax.set_xlabel('Neutron Number (N)')
    ax.set_ylabel('Proton Number (Z)')
    ax.set_title('Nuclide Chart: Abundances Over Time')
    add_driplines()

    return im,  

# Calculate the number of nuclides per time step
# num_nuclides_per_step = len(Z_data) // len(time_data)

def update(frame):
    current_Z = Z_data  # Use the entire Z_data array
    current_N = N_data  # Use the entire N_data array
    current_Y = Y_data[frame, :]  # Get abundances for the current time frame

    # Reset grid
    grid.fill(0)

    # Update grid with current abundances
    for i in range(len(current_Z)):
        z = current_Z[i]
        n = current_N[i]
        abundance = current_Y[i]

        # Set grid value to the abundance at (Z, N)
        grid[z, n] = abundance

    # Update the image data in the imshow object
    im.set_data(grid)
    
    # Update the title with the current time
    ax.set_title(f'Nuclide Chart: Time = {time_data[frame]:.4f} s')

    return im,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(time_data), init_func=init, blit=False, repeat=False)

# Save or show the animation
ani.save('nuclide_chart_animation.mp4', writer='ffmpeg', fps=30) #TODO: do one with more fps

# Display the plot
plt.show()
