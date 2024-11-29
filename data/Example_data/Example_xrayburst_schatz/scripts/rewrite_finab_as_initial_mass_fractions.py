# Input mass fractions in the form of a list of tuples
# (A, Z, N, Yi, Xi) -> You can replace this with your actual data reading mechanism
mass_fractions = [
    (1, 1, 0, 4.53470521998598E-14, 4.53470521998598E-14),
    (4, 2, 2, 1.34796149528653E-03, 5.39184598114614E-03),
    (12, 6, 6, 3.39653449492245E-04, 4.07584139390694E-03),
    # Add all the data here
]

# Isotope symbols dictionary, extend this for all needed elements
element_symbols = {
    1: 'h', 2: 'he', 3: 'li', 4: 'be', 5: 'b', 6: 'c', 7: 'n', 8: 'o', 9: 'f', 10: 'ne', 11: 'na',
    12: 'mg', 13: 'al', 14: 'si', 15: 'p', 16: 's', 17: 'cl', 18: 'ar', 19: 'k', 20: 'ca',
    21: 'sc', 22: 'ti', 23: 'v', 24: 'cr', 25: 'mn', 26: 'fe', 27: 'co', 28: 'ni', 29: 'cu',
    30: 'zn', 31: 'ga', 32: 'ge', 33: 'as', 34: 'se', 35: 'br', 36: 'kr', 37: 'rb', 38: 'sr',
    39: 'y', 40: 'zr', 41: 'nb', 42: 'mo', 43: 'tc', 44: 'ru', 45: 'rh', 46: 'pd', 47: 'ag',
    48: 'cd', 49: 'in', 50: 'sn', 51: 'sb', 52: 'te', 53: 'i', 54: 'xe'
}

# Function to write mass fractions to a file
def write_mass_fractions_to_file(filename, mass_fractions, element_symbols):
    with open(filename, 'w') as file:
        for A, Z, N, Yi, Xi in mass_fractions:
            element_symbol = element_symbols.get(Z, 'unknown')  # Default to 'unknown' if element not in dictionary
            isotope_name = f"{element_symbol}{A - Z}"  # Format like 'he4' for helium-4
            file.write(f"  {isotope_name:<6}  {Xi:.6e}\n")

# Write mass fractions to file
output_file = 'mass_fractions_output.txt'
write_mass_fractions_to_file(output_file, mass_fractions, element_symbols)

print(f"Mass fractions written to {output_file}")
