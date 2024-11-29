import networkx as nx
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os







# # Load the data using the provided column names
column_names = ["time", "temp", "dens", "nin", "zin", "yin", "nout", "zout", "yout", "flow"]
directory_path = 'flow'
# df_raw = pd.read_csv(file_path, sep=r'\s+', names=column_names, skip_blank_lines=False)

# filtered_rows = []
# for index, row in df_raw.iterrows():
#     try:
#         # Attempt to convert all elements to numeric values
#         numeric_row = pd.to_numeric(row, errors='raise')
#         filtered_rows.append(numeric_row)
#     except ValueError:
#         # Skip rows that cannot be converted to numeric values (i.e., headers or invalid rows)
#         continue

# # Create a new dataframe from the filtered rows
# df_filtered = pd.DataFrame(filtered_rows, columns=column_names)

# # Convert 'nin' and 'nout' columns to integers without converting other columns
# # Use floor or rounding as needed to handle floating-point values in 'nin' and 'nout'
# df_filtered['nin'] = df_filtered['nin'].astype(int, errors='ignore')
# df_filtered['nout'] = df_filtered['nout'].astype(int, errors='ignore')

# # Create a graph to represent the reactions
# G = nx.DiGraph()

# # Add edges to the graph based on the reactions data
# for _, row in df_filtered.iterrows():
#     nin = row['nin']
#     nout = row['nout']
#     flow = row['flow']
    
#     # Add an edge from 'nin' to 'nout' with the flow as the weight
#     G.add_edge(nin, nout, weight=flow)

# # Draw the graph
# pos = nx.spring_layout(G)  # Use spring layout for better visualization
# edge_labels = nx.get_edge_attributes(G, 'weight')

# plt.figure(figsize=(10, 8))
# try:
#     nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=15, font_weight='bold', arrows=True, arrowstyle='-|>', arrowsize=20)
# except Exception as e:
#     print(f"Error while drawing the graph: {e}")
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

# plt.title('Reaction Flow Graph Network')
# plt.show()


# Initialize a list to store reactions with flow > 1e-2
# Iterate over all files in the directory
# Initialize a list to store reactions with flow > 1e-2
# Column names for each file
column_names = ["time", "temp", "dens", "nin", "zin", "yin", "nout", "zout", "yout", "flow"]

# Initialize a list to store reactions with flow > 1e-5
significant_reactions = []

# Iterate over all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.dat'):
        file_path = os.path.join(directory_path, filename)
        
        # Load the data, skip potential header rows, and ensure data consistency
        try:
            df = pd.read_csv(file_path, sep=r'\s+', names=column_names, comment='#', skip_blank_lines=True)
        except Exception as e:
            print(f"Error loading file {filename}: {e}")
            continue

        # Drop rows with NaN values in crucial columns (assumed last column is 'flow')
        df_filtered = df.dropna(subset=[column_names[-1]]).reset_index(drop=True)

        # Ensure flow column is numeric and filter for significant flows
        df_filtered[column_names[-1]] = pd.to_numeric(df_filtered[column_names[-1]], errors='coerce')
        df_filtered = df_filtered[df_filtered[column_names[-1]] > 1e-5]

        # Append relevant information (filename, time, nin, zin, nout, zout, flow) to the list
        for _, row in df_filtered.iterrows():
            significant_reactions.append((
                filename, row[column_names[0]], row[column_names[3]], row[column_names[4]], row[column_names[6]], row[column_names[7]], row[column_names[-1]]
            ))

# Display the significant reactions
if significant_reactions:
    for reaction in significant_reactions:
        print(f"File: {reaction[0]}, Time: {reaction[1]}, Nin: {reaction[2]}, Zin: {reaction[3]}, Nout: {reaction[4]}, Zout: {reaction[5]}, Flow: {reaction[6]}")
else:
    print("No significant reactions found with flow greater than 1e-5.")



print(significant_reactions)