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

    
print("Z DATA:")
print (Z_data)

print("Y DATA:")
print(Y_data)

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

def plot_h_he():
    
    print("A: ",A_data)
    print("N: ",N_data)
    print(time_data)
    print("Z: ",Z_data)
    print(Y_data)
    # Filter the time data for the first second
    indices_within_time = time_data <= 1.0

    # Initialize arrays to hold Hydrogen and Helium abundances
    hydrogen_abundance = []
    helium_abundance = []

    # Iterate over each time point within the first second
    for i, within_time in enumerate(indices_within_time):
        if within_time:
            current_Z = Z_data[i]
            current_Y = Y_data[i]
            
            # Sum the abundances for Hydrogen (Z = 1) and Helium (Z = 2)
            hydrogen_abund = np.sum(current_Y[current_Z == 1])
            helium_abund = np.sum(current_Y[current_Z == 2])
            print(hydrogen_abund)
            hydrogen_abundance.append(hydrogen_abund)
            helium_abundance.append(helium_abund)

    # Convert lists to numpy arrays
    hydrogen_abundance = np.array(hydrogen_abundance)
    helium_abundance = np.array(helium_abundance)

    time_data_filtered = time_data[indices_within_time]

    # Plotting Hydrogen and Helium abundances vs time (within the first second)
    plt.figure(figsize=(10, 6))
    plt.plot(time_data_filtered, hydrogen_abundance, label='Hydrogen Abundance', color='blue')
    plt.plot(time_data_filtered, helium_abundance, label='Helium Abundance', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Abundance')
    plt.title('Hydrogen and Helium Abundances vs Time (First Second)')
    plt.legend()
    plt.grid(True)
    plt.show()



#Done
#Done
def He4_H1_abundances_plot():
    import numpy as np
    import matplotlib.pyplot as plt

    # Initialize indices to None
    H1_index = None
    He4_index = None

    # Loop over indices to find H1 and He4
    for i in range(len(Z_data)):
        if (Z_data[i] == 1) and (A_data[i] == 1):
            H1_index = i
        if (Z_data[i] == 2) and (A_data[i] == 4):
            He4_index = i

    # Check if indices were found
    if H1_index is None:
        print("Error: H1 not found in the data.")
        return
    if He4_index is None:
        print("Error: He4 not found in the data.")
        return

    # Extract abundance data directly using NumPy slicing
    Y_H1 = Y_data[:, H1_index]
    Y_He4 = Y_data[:, He4_index]

    # Plotting
    plt.plot(time_data, Y_H1, label='H1')
    plt.plot(time_data, Y_He4, label='He4')

    plt.xlabel('Time (s)')
    plt.ylabel('Abundance')
    plt.title('Abundances of H1 and He4 vs Time')
    plt.legend()
    plt.yscale('log')  # Use a logarithmic scale if abundances vary greatly
    plt.grid(True)
    plt.show()

    

He4_H1_abundances_plot()


##Plotting Energy Generation##

def plot_4th_column(directory_path):
    # File paths for the CSV files
    file_1 = os.path.join(directory_path, 'tcontr_converted.csv')
    file_2 = os.path.join(directory_path, 'tcontr-1_converted.csv')

    # Load the CSV data
    data_1 = pd.read_csv(file_1)
    data_2 = pd.read_csv(file_2)

    # Extract time and 4th column from both files
    time_1 = data_1.iloc[:, 0]  # Assuming time is in the first column
    values_1 = data_1.iloc[:, 3]  # 4th column

    time_2 = data_2.iloc[:, 0]
    values_2 = data_2.iloc[:, 3]



    # Plotting both columns with respect to time on the same plot (log scale)
    plt.figure(figsize=(10, 6))
    plt.plot(time_1, values_1, label='Run 1 ')
    plt.plot(time_2, values_2, label='Run 2 ')

    plt.xlabel('Time')
    plt.ylabel('Nuclear Energy Generation [erg/g/s]')
    plt.yscale('log')
    plt.title('Nuclear Energy Generation vs Time (Log Scale)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
plot_4th_column('/home/ryan/University/Honours/Semester_2/Project/WinNet/Traces/')


def P_T_Entropy_plot():
    file_path = os.path.join(current_dir,'mainout.dat')

    # Step 1: Read the data (assuming the file is space-delimited)
    try:
        # Skip the first row of headers and define column names explicitly
        column_names = ['iteration', 'time', 'temperature', 'rho', 'Ye', 
                        'R', 'Y_n', 'Y_p', 'Y_alpha', 'Y_lights', 
                        'Y_heavies', '<Z>', '<A>', 'entropy', 'Sn']
        data = pd.read_csv(file_path, delim_whitespace=True, skiprows=3, names=column_names)
    
        # Display the first few rows to verify the data structure
        print(data.head())

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return  # Exit the function if data cannot be read

    # Step 2: Plot temperature, pressure, and entropy vs time
    try:
       
    
        # Plot Temperature vs Time
        
        plt.plot(data['time'], data['temperature'], color='r')
        plt.title('Temperature vs Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Temperature [GK]')
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=15))
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins=15))
        plt.show()
        # Plot Entropy vs Time
        
        plt.plot(data['time'], data['entropy'], color='b')
        plt.title('Entropy vs Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Entropy [kB/baryon]')
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=15))
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins=15))
        plt.show()

        plt.plot(data['time'], data['rho'], color='g')  # Assuming 'rho' represents pressure
        plt.title('Density vs Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Density [units]')
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=15))
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins=15))

        plt.show()
        # Adjust layout and show plot
        
        

    except KeyError as e:
        print(f"One of the columns was not found in the data: {e}")

# Call the function to execute the plot
P_T_Entropy_plot()

def PT_comparison_plot():
    # File paths for both mainout.dat files
    file_path_1 = os.path.join(os.getcwd(), 'mainout.dat')
    file_path_2 = '/home/ryan/University/Honours/Semester_2/Project/WinNet/runs/x-ray_burst/mainout.dat'

    try:
        # Define column names explicitly
        column_names = ['iteration', 'time', 'temperature', 'rho', 'Ye', 
                        'R', 'Y_n', 'Y_p', 'Y_alpha', 'Y_lights', 
                        'Y_heavies', '<Z>', '<A>', 'entropy', 'Sn']

        # Read the data from both files
        data_1 = pd.read_csv(file_path_1, delim_whitespace=True, skiprows=3, names=column_names)
        data_2 = pd.read_csv(file_path_2, delim_whitespace=True, skiprows=3, names=column_names)

        # Plot Temperature vs Time for both datasets
        plt.plot(data_1['time'], data_1['temperature'], color='r', label='Run 2')
        plt.plot(data_2['time'], data_2['temperature'], color='b', label='Run 1')
        plt.title('Temperature Comparison vs Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Temperature [GK]')
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=15))
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins=15))
        plt.legend()
        plt.show()

        # Plot Density vs Time for both datasets
        plt.plot(data_1['time'], data_1['rho'], color='r', label='Run 2')
        plt.plot(data_2['time'], data_2['rho'], color='b', label='Run 1')
        plt.title('Density Comparison vs Time')
        plt.xlabel('Time [s]')
        plt.ylabel(r'Density [$\mathrm{g/cm^3}$]')
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=15))
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins=15))
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"An error occurred while reading the files: {e}")

PT_comparison_plot()


 
def nuclide_abundances_plot(nuclides):
    """
    Plot the abundance of multiple nuclides given their atomic numbers and mass numbers.
    
    Parameters:
    nuclides (list of tuples): List of (Z, A) tuples for the nuclides.

    """
    for Z, A in nuclides:
        nuclide_index = -1
        
        # Find the index of the nuclide with the given Z and A
        for i in range(len(Z_data)):
            if Z_data[i] == Z and A_data[i] == A:
                nuclide_index = i
                break

        if nuclide_index == -1:
            print(f"Nuclide with Z={Z} and A={A} not found.")
            continue

        # Extract abundance data for the nuclide
        Y_nuclide = []
        for i in range(len(time_data)):
            Y_nuclide.append(Y_data[i, nuclide_index])

        # Get the element symbol from the atomic number
        element_symbol = element_symbols[Z] if Z < len(element_symbols) else f'Z={Z}'

        # Plotting the abundance vs time
        plt.plot(time_data, Y_nuclide, label=f'{element_symbol}-{A}')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Abundance')
    plt.title('Abundance of Nuclides vs Time')
    plt.legend()
    plt.yscale('log')  # Use a logarithmic scale if abundances vary greatly
    plt.grid(True)
    plt.show()


# Example usage
nuclide_abundances_plot([(28, 56), (32, 64), (34, 68), (36, 72), (50, 104)])



def plot_all_abundances():
        # Find all unique Z values
  # Assume that Z_data and Y_data are structured such that they need to be split by time step
    num_nuclides_per_step = len(Z_data) // len(time_data)  # Assuming equal number of nuclides per time step

        # Prepare a dictionary to store abundance over time for each Z
    abundance_dict = {}

        # Loop through each time step
    for i in range(len(time_data)):
            # Extract data for the current time step
        start_index = i * num_nuclides_per_step
        end_index = start_index + num_nuclides_per_step
            
        current_Z = Z_data[start_index:end_index]
        current_Y = Y_data[start_index:end_index]
            
            # Loop through each unique Z value and sum the abundances
        for Z in np.unique(current_Z):
            
            if Z not in abundance_dict:
            
                abundance_dict[Z] = []
            
            abundance = np.sum(current_Y[current_Z == Z])
            abundance_dict[Z].append(abundance)

        # Convert lists to numpy arrays for plotting
    for Z in abundance_dict:
        
        abundance_dict[Z] = np.array(abundance_dict[Z])

    # Plotting
    plt.figure(figsize=(12, 8))
    for Z in abundance_dict:
        plt.plot(time_data, abundance_dict[Z], label=f'Z = {Z}')

    plt.xlabel('Time (s)')
    plt.ylabel('Abundance')
    plt.title('Abundances vs Time for All Z Numbers')
    plt.legend()
    plt.yscale('log')  # Use a logarithmic scale if abundances vary greatly
    plt.grid(True)
    plt.show()





#This method is getting the SUM_ABUNDANCES (summing all isotopes for each A value)
#
#
def extract_abundances(file_path):
    # Read the file and skip the header line
    data = pd.read_csv(file_path, delim_whitespace=True, skiprows=1, names=["A", "Y(A)", "X(A)"])
    
    # Return the DataFrame
    return data





# Example usage
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path_abundances = os.path.join(current_dir, "finabsum.dat")
#file_path = '/home/ryan/University/Honours/Semester_2/Project/WinNet/runs/x-ray_burst/finabsum.dat'

abundance_df = extract_abundances(file_path_abundances)

# Display the resulting table
print(abundance_df)

# If you want to save it to a CSV file
#abundance_df.to_csv('abundances.csv', index=False)







def plot_abundances_Sn_Sb_Te(time_data, Z_data, Y_data):
    # Atomic numbers for Sn, Sb, and Te
    atomic_numbers = {
        'Sn': 50,
        'Sb': 51,
        'Te': 52
    }

    #These arrays hold the indices of the position of the corresponding Z numbers in the Z list
    Sn_indices = []
    Sb_indices = []
    Te_indices = []

    index = 0 #variable tracks index number
    for i in Z_data:
        
        if i == 50:
            Sn_indices.append(index)
        elif i == 51:
            Sb_indices.append(index)
        elif i == 52:
            Te_indices.append(index)
        
        index += 1

    Y_Sn = []
    Y_Sb = []
    Y_Te = []
    
    
    
    #This loop is going throught the Y_2D array and summing all the Y values for the corresponding Z values and putting them into seperate lists/arrays
    for i in range(0,len(time_data)):

        sum_Sn = 0
        for j in Sn_indices:
            sum_Sn += Y_data[i,j]
        Y_Sn.append(sum_Sn)

        sum_Sb = 0
        for j in Sb_indices:
            sum_Sb += Y_data[i,j]
        Y_Sb.append(sum_Sb)
        
        sum_Te = 0j
        for j in Te_indices:
            sum_Te += Y_data[i,j]

        Y_Te.append(sum_Te)
        
    
    plt.plot(time_data,Y_Te,label ='Te')
    plt.plot(time_data,Y_Sn,label = 'Sn')
    plt.plot(time_data,Y_Sb, label = 'Sb')

    plt.xlabel('Time (s)')
    plt.ylabel('Abundance')
    plt.title('Abundances of Sn, Sb, and Te vs Time')
    plt.legend()
    plt.yscale('log')  # Use a logarithmic scale if abundances vary greatly
    plt.grid(True)
    plt.show()


# Example usage
plot_abundances_Sn_Sb_Te(time_data, Z_data, Y_data)


# Example usage


# plot_abundances_Sn_Sb_Te(file_path)

#plot_all_abundances()
#plot_h_he()


def P_T_Entropy_plot():
    file_path = '/home/ryan/University/Honours/Semester_2/Project/WinNet/runs/x-ray_burst/mainout.dat'

    # Step 1: Read the data (assuming the file is space-delimited)
    try:
        # Skip the first row of headers and define column names explicitly
        column_names = ['iteration', 'time', 'temperature', 'rho', 'Ye', 
                        'R', 'Y_n', 'Y_p', 'Y_alpha', 'Y_lights', 
                        'Y_heavies', '<Z>', '<A>', 'entropy', 'Sn']
        data = pd.read_csv(file_path, delim_whitespace=True, skiprows=3, names=column_names)
    
        # Display the first few rows to verify the data structure
        print(data.head())

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return  # Exit the function if data cannot be read

    # Step 2: Plot temperature, pressure, and entropy vs time
    try:
       
    
        # Plot Temperature vs Time
        
        plt.plot(data['time'], data['temperature'], color='r')
        plt.title('Temperature vs Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Temperature [GK]')
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=15))
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins=15))
        plt.show()
        # Plot Entropy vs Time
        
        plt.plot(data['time'], data['entropy'], color='b')
        plt.title('Entropy vs Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Entropy [kB/baryon]')
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=15))
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins=15))
        plt.show()

        plt.plot(data['time'], data['rho'], color='g')  # Assuming 'rho' represents pressure
        plt.title('Pressure vs Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Pressure [units]')
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=15))
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins=15))

        plt.show()
        # Adjust layout and show plot
        
        

    except KeyError as e:
        print(f"One of the columns was not found in the data: {e}")

# Call the function to execute the plot
P_T_Entropy_plot()




##Most Prevalent Nuclides##

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

# Select the top 25 most abundant isotopes
top_n = 25
top_n_data = data_sorted.head(top_n)

# Reorder the columns for better presentation
top_n_display = top_n_data[['Isotope', 'Z', 'N', 'A', 'Abundance']]

# Format the abundance for better readability
pd.options.display.float_format = '{:.3e}'.format

### TABLE ###






# Display the top 25 isotopes as a table
print(top_n_display.to_string(index=False))

# Extract the indices for the top 25 isotopes
top_n_indices = top_n_data['Index'].values

# Get the isotope labels
isotope_labels = top_n_data['Isotope'].values

# Extract the abundances over time for the top 25 isotopes
abundances_over_time = Y_data[:, top_n_indices]  # Shape: (num_time_steps, top_n)

# Extract hydrogen and helium-4 abundances
# Extract abundance of Hydrogen-1 (Z=1, A=1) over time
hydrogen1_abundance = Y_data[:, (Z_data == 1) & (A_data == 1)]

# Extract abundance of Helium-3 (Z=2, A=3) over time
helium3_abundance = Y_data[:, (Z_data == 2) & (A_data == 3)]

# Extract abundance of Helium-4 (Z=2, A=4) over time
helium4_abundance = Y_data[:, (Z_data == 2) & (A_data == 4)]

# Define the number of nuclides to plot
num_nuclides = len(top_n_indices) + 3

# Create a colormap with enough distinct colors
cmap = plt.cm.get_cmap('tab20', num_nuclides)

# Plotting the abundances over time
plt.figure(figsize=(12, 8))

for i in range(len(top_n_indices)):
    color = cmap(i)
    plt.plot(time_data, abundances_over_time[:, i], label=isotope_labels[i], color=color)

# Add hydrogen and helium-4 to the plot
plt.plot(time_data, hydrogen1_abundance, label='H-1', color='red')
plt.plot(time_data, helium4_abundance, label='He-4', color='black')
plt.plot(time_data, helium3_abundance, label='He-3', color='purple')

plt.xlabel('Time (s)')
plt.ylabel('Abundance (Y)')
plt.title('Abundances of Top 25 Isotopes Over Time (Including H-1, He-3 and He-4)')
plt.legend(loc='lower right', fontsize='small', ncol=2)
plt.yscale('log')
plt.grid(True)
plt.tight_layout()
plt.show()







##ALL NUCLIDES##TODO

# Extract the indices for all isotopes
all_indices = data['Index'].values

# Extract the indices for all isotopes with non-zero abundance
y_max = np.max(Y_data, axis=0)
non_zero_indices = np.where(y_max > 0)[0]
all_indices = data['Index'].values[non_zero_indices]

# Get the isotope labels for non-zero isotopes
isotope_labels = data['Isotope'].values[non_zero_indices]

# Extract the abundances over time for non-zero isotopes
abundances_over_time = Y_data[:, all_indices]  # Shape: (num_time_steps, number_of_nuclides)

# Define the number of nuclides to plot
num_nuclides = len(all_indices)

# Create a colormap with enough distinct colors
cmap = plt.cm.get_cmap('tab20', 30)

# Plotting the abundances over time
plt.figure(figsize=(14, 11.5))

for i in range(num_nuclides):
    color = cmap(i % 30)  # Reuse colors after 30 are exhausted
    abundance = abundances_over_time[:, i]
    plt.plot(time_data, abundance, label=isotope_labels[i], color=color)

plt.xlabel('Time (s)')
plt.ylabel('Abundance (Y)')
plt.title('Abundances of All Isotopes Over Time')
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize='small', ncol=4)
plt.yscale('log')
plt.ylim(1e-25, 1e+0)
plt.grid(True)
plt.tight_layout()
plt.show()




###INTERACTIVE VERSION###


import plotly.graph_objs as go
import plotly.subplots as sp

# Creating an interactive plot using Plotly
fig = sp.make_subplots(rows=1, cols=1, subplot_titles=('Abundances of All Isotopes Over Time',))

for i in range(num_nuclides):
    color = cmap(i % 30)  # Reuse colors after 30 are exhausted
    abundance = abundances_over_time[:, i]
    fig.add_trace(go.Scatter(x=time_data, y=abundance, mode='lines', name=isotope_labels[i],
                             line=dict(color='rgba' + str(color), width=2)))

fig.update_xaxes(title_text='Time (s)')
fig.update_yaxes(title_text='Abundance (Y)', type='log', range=[-25, 0])
fig.update_layout(title_text='Abundances of All Isotopes Over Time', legend=dict(x=1.1, y=1.1), autosize=True, margin=dict(l=50, r=50, b=100, t=100, pad=4),
                  template='plotly_white')

fig.show()
