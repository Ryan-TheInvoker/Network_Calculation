import pandas as pd
import glob
import matplotlib.pyplot as plt
from io import StringIO
import matplotlib.animation as animation
import networkx as nx

species_names = {
    (1, 0): 'H',       # Hydrogen-1
    (1, 1): 'D',       # Deuterium
    (1, 2): 'T',       # Tritium
    (2, 1): 'He3',     # Helium-3
    (2, 2): 'He4',     # Helium-4
    (3, 3): 'Li6',     # Lithium-6
    (3, 4): 'Li7',     # Lithium-7
    (4, 3): 'Be7',     # Beryllium-7
    (4, 4): 'Be8',     # Beryllium-8
    (5, 5): 'B10',     # Boron-10
    (5, 6): 'B11',     # Boron-11
    (6, 6): 'C12',     # Carbon-12
    (6, 7): 'C13',     # Carbon-13
    (6, 8): 'C14',     # Carbon-14
    (7, 7): 'N14',     # Nitrogen-14
    (7, 8): 'N15',     # Nitrogen-15
    (8, 8): 'O16',     # Oxygen-16
    (8, 9): 'O17',     # Oxygen-17
    (8, 10): 'O18',    # Oxygen-18
    (9, 10): 'F19',    # Fluorine-19
    (10, 10): 'Ne20',  # Neon-20
    (10, 11): 'Ne21',  # Neon-21
    (10, 12): 'Ne22',  # Neon-22
    (11, 12): 'Na23',  # Sodium-23
    (12, 12): 'Mg24',  # Magnesium-24
    (12, 13): 'Mg25',  # Magnesium-25
    (12, 14): 'Mg26',  # Magnesium-26
    (13, 14): 'Al27',  # Aluminum-27
    (14, 14): 'Si28',  # Silicon-28
    (14, 15): 'Si29',  # Silicon-29
    (14, 16): 'Si30',  # Silicon-30
    (15, 16): 'P31',   # Phosphorus-31
    (16, 16): 'S32',   # Sulfur-32
    (16, 17): 'S33',   # Sulfur-33
    (16, 18): 'S34',   # Sulfur-34
    (16, 20): 'S36',   # Sulfur-36
    (17, 18): 'Cl35',  # Chlorine-35
    (17, 20): 'Cl37',  # Chlorine-37
    (18, 22): 'Ar40',  # Argon-40
    (18, 21): 'Ar39',  # Argon-39
    (18, 20): 'Ar38',  # Argon-38
    (19, 20): 'K39',   # Potassium-39
    (19, 21): 'K40',   # Potassium-40
    (19, 22): 'K41',   # Potassium-41
    (20, 20): 'Ca40',  # Calcium-40
    (20, 21): 'Ca41',  # Calcium-41
    (20, 22): 'Ca42',  # Calcium-42
    (20, 24): 'Ca44',  # Calcium-44
    (20, 25): 'Ca45',  # Calcium-45
    (20, 26): 'Ca46',  # Calcium-46
    (20, 28): 'Ca48',  # Calcium-48
    (21, 24): 'Sc45',  # Scandium-45
    (22, 26): 'Ti48',  # Titanium-48
    (22, 28): 'Ti50',  # Titanium-50
    (23, 28): 'V51',   # Vanadium-51
    (24, 28): 'Cr52',  # Chromium-52
    (24, 30): 'Cr54',  # Chromium-54
    (25, 30): 'Mn55',  # Manganese-55
    (26, 30): 'Fe56',  # Iron-56
    (26, 32): 'Fe58',  # Iron-58
    (27, 32): 'Co59',  # Cobalt-59
    (28, 31): 'Ni59',  # Nickel-59
    (28, 32): 'Ni60',  # Nickel-60
    (29, 34): 'Cu63',  # Copper-63
    (29, 36): 'Cu65',  # Copper-65
    (30, 34): 'Zn64',  # Zinc-64
    (30, 36): 'Zn66',  # Zinc-66
    (31, 38): 'Ga69',  # Gallium-69
    (31, 40): 'Ga71',  # Gallium-71
    (32, 38): 'Ge70',  # Germanium-70
    (32, 40): 'Ge72',  # Germanium-72
    (33, 42): 'As75',  # Arsenic-75
    (34, 45): 'Se79',  # Selenium-79
    (34, 46): 'Se80',  # Selenium-80
    (35, 44): 'Br79',  # Bromine-79
    (35, 46): 'Br81',  # Bromine-81
    (36, 48): 'Kr84',  # Krypton-84
    (36, 50): 'Kr86',  # Krypton-86
    (37, 48): 'Rb85',  # Rubidium-85
    (37, 50): 'Rb87',  # Rubidium-87
    (38, 50): 'Sr88',  # Strontium-88
    (39, 50): 'Y89',   # Yttrium-89
    (40, 51): 'Zr91',  # Zirconium-91
    (40, 54): 'Zr94',  # Zirconium-94
    (41, 52): 'Nb93',  # Niobium-93
    (42, 54): 'Mo96',  # Molybdenum-96
    (42, 58): 'Mo100', # Molybdenum-100
    (43, 56): 'Tc99',  # Technetium-99
    (44, 57): 'Ru101', # Ruthenium-101
    (44, 58): 'Ru102', # Ruthenium-102
    (45, 58): 'Rh103', # Rhodium-103
    (46, 60): 'Pd106', # Palladium-106
    (46, 62): 'Pd108', # Palladium-108
    (47, 60): 'Ag107', # Silver-107
    (47, 62): 'Ag109', # Silver-109
    (48, 64): 'Cd112', # Cadmium-112
    (49, 66): 'In115', # Indium-115
    (50, 68): 'Sn118', # Tin-118
    (50, 70): 'Sn120', # Tin-120
    (9, 9): 'F18',     # Species with n=9.0, z=9.0
    (11, 9): 'F20',    # Species with n=9.0, z=11.0
    (12, 9): 'F21',    # Species with n=9.0, z=12.0
    (13, 10): 'Ne23',  # Species with n=10.0, z=13.0
    (13, 11): 'Na24',  # Species with n=11.0, z=13.0
    (14, 10): 'Ne24',  # Species with n=10.0, z=14.0
    (14, 11): 'Na25',  # Species with n=11.0, z=14.0
    (15, 11): 'Na26',  # Species with n=11.0, z=15.0
    (15, 12): 'Mg27',  # Species with n=12.0, z=15.0
    (15, 13): 'Al28',  # Species with n=13.0, z=15.0
    (16, 11): 'Na27',  # Species with n=11.0, z=16.0
    (16, 12): 'Mg28',  # Species with n=12.0, z=16.0
    (16, 13): 'Al29',  # Species with n=13.0, z=16.0
    (17, 14): 'Si31',  # Species with n=14.0, z=17.0
    (17, 15): 'P32',   # Species with n=15.0, z=17.0
    (18, 15): 'P33',   # Species with n=15.0, z=18.0
    (17, 17): 'Cl34',  # Species with n=17.0, z=17.0
    (19, 17): 'Cl36',  # Species with n=17.0, z=19.0
    (19, 18): 'Ar37',  # Species with n=18.0, z=19.0
    (26, 26): 'Fe52',  # Species with n=26.0, z=26.0
    (27, 26): 'Fe53',  # Species with n=26.0, z=27.0
    (26, 27): 'Co53',  # Species with n=27.0, z=26.0
    (27, 27): 'Co54',  # Species with n=27.0, z=27.0
    (28, 26): 'Fe54',  # Species with n=26.0, z=28.0
    (28, 27): 'Co55',  # Species with n=27.0, z=28.0
    (29, 27): 'Co56',  # Species with n=27.0, z=29.0
    (19, 19): 'K38',   # Species with n=19.0, z=19.0
    (19, 20): 'K39',   # Species with n=20.0, z=19.0
    (21, 21): 'Sc42',  # Species with n=21.0, z=21.0
    (22, 21): 'Sc43',  # Species with n=21.0, z=22.0
    (23, 21): 'Sc44',  # Species with n=21.0, z=23.0
    (22, 22): 'Ti44',  # Species with n=22.0, z=22.0
    (23, 22): 'Ti45',  # Species with n=22.0, z=23.0
    (24, 22): 'Ti46',  # Species with n=22.0, z=24.0
    (23, 23): 'V46',   # Species with n=23.0, z=23.0
    (24, 23): 'V47',   # Species with n=23.0, z=24.0
    (25, 22): 'Ti47',  # Species with n=22.0, z=25.0
    (25, 23): 'V48',   # Species with n=23.0, z=25.0
    (24, 24): 'Cr48',  # Species with n=24.0, z=24.0
    (25, 24): 'Cr49',  # Species with n=24.0, z=25.0
    (26, 23): 'V49',   # Species with n=23.0, z=26.0
    (26, 24): 'Cr50',  # Species with n=24.0, z=26.0
    (25, 25): 'Mn50',  # Species with n=25.0, z=25.0
    (26, 25): 'Mn51',  # Species with n=25.0, z=26.0
    (25, 26): 'Fe51',  # Species with n=26.0, z=25.0
    (27, 24): 'Cr51',  # Species with n=24.0, z=27.0
    (27, 25): 'Mn52',  # Species with n=25.0, z=27.0
    (28, 25): 'Mn53',  # Species with n=25.0, z=28.0
    (27, 28): 'Ni55',  # Species with n=28.0, z=27.0
    (28, 28): 'Ni56',  # Species with n=28.0, z=28.0
    (29, 28): 'Ni57',  # Species with n=28.0, z=29.0
    (30, 28): 'Ni58',  # Species with n=28.0, z=30.0
    (29, 29): 'Cu58',  # Species with n=29.0, z=29.0
    (30, 29): 'Cu59',  # Species with n=29.0, z=30.0
    (29, 30): 'Zn59',  # Species with n=30.0, z=29.0
    (30, 30): 'Zn60',  # Species with n=30.0, z=30.0
    (34, 34): 'Se68',  # Species with n=34.0, z=34.0
    (33, 35): 'Br68',  # Species with n=35.0, z=33.0
    (36, 36): 'Kr72',  # Species with n=36.0, z=36.0
    (35, 37): 'Rb72',  # Species with n=37.0, z=35.0
    (38, 38): 'Sr76',  # Species with n=38.0, z=38.0
    (37, 39): 'Y76',   # Species with n=39.0, z=37.0
    (40, 40): 'Zr80',  # Species with n=40.0, z=40.0
    (39, 41): 'Nb80',  # Species with n=41.0, z=39.0
    (42, 42): 'Mo84',  # Species with n=42.0
        (9, 11): 'Na20',    # Species with n=9.0, z=11.0
    (9, 12): 'Mg21',    # Species with n=9.0, z=12.0
    (10, 13): 'Al23',   # Species with n=10.0, z=13.0
    (11, 13): 'Si24',   # Species with n=11.0, z=13.0
    (10, 14): 'Si24',   # Species with n=10.0, z=14.0
    (11, 14): 'P25',    # Species with n=11.0, z=14.0
    (11, 15): 'P26',    # Species with n=11.0, z=15.0
    (12, 15): 'S27',    # Species with n=12.0, z=15.0
    (13, 15): 'S28',    # Species with n=13.0, z=15.0
    (11, 16): 'S27',    # Species with n=11.0, z=16.0
    (12, 16): 'Cl28',   # Species with n=12.0, z=16.0
    (13, 16): 'Cl29',   # Species with n=13.0, z=16.0
    (14, 17): 'Cl31',   # Species with n=14.0, z=17.0
    (15, 17): 'Cl32',   # Species with n=15.0, z=17.0
    (15, 18): 'Ar33',   # Species with n=15.0, z=18.0
    (17, 19): 'K36',    # Species with n=17.0, z=19.0
    (18, 19): 'K37',    # Species with n=18.0, z=19.0
    (26, 28): 'Fe54',   # Species with n=26.0, z=28.0
    (27, 29): 'Co56',   # Species with n=27.0, z=29.0
    (20, 19): 'Ca39',   # Species with n=20.0, z=19.0
    (21, 22): 'Sc43',   # Species with n=21.0, z=22.0
    (21, 23): 'V44',    # Species with n=21.0, z=23.0
    (22, 23): 'Ti45',   # Species with n=22.0, z=23.0
    (22, 24): 'V46',    # Species with n=22.0, z=24.0
    (23, 24): 'Cr47',   # Species with n=23.0, z=24.0
    (22, 25): 'Mn47',   # Species with n=22.0, z=25.0
    (23, 25): 'Fe48',   # Species with n=23.0, z=25.0
    (24, 25): 'Co49',   # Species with n=24.0, z=25.0
    (23, 26): 'Fe49',   # Species with n=23.0, z=26.0
    (24, 26): 'Ni50',   # Species with n=24.0, z=26.0
    (24, 27): 'Co51',   # Species with n=24.0, z=27.0
    (25, 27): 'Ni52',   # Species with n=25.0, z=27.0
    (25, 28): 'Cu53',   # Species with n=25.0, z=28.0
    (28, 29): 'Zn57',   # Species with n=28.0, z=29.0
    (28, 30): 'Ga58',   # Species with n=28.0, z=30.0
    (35, 33): 'Br68',   # Species with n=35.0, z=33.0
    (37, 35): 'Rb72',   # Species with n=37.0, z=35.0
    (39, 37): 'Y76',    # Species with n=39.0, z=37.0
    (41, 39): 'Nb80',   # Species with n=41.0, z=39.0
    (43, 41): 'Tc84',   # Species with n=43.0, z=41.0
    (44, 42): 'Ru86',   # Species with n=44.0, z=42.0
    (45, 41): 'Rh86',   # Species with n=45.0, z=41.0
    (50, 48): 'Sn98',   # Species with n=50.0, z=48.0
    (51, 47): 'Sb98',   # Species with n=51.0, z=47.0
    (52, 50): 'Te102',  # Species with n=52.0, z=50.0
    (53, 49): 'I102',   # Species with n=53.0, z=49.0
    (53, 50): 'I103',   # Species with n=53.0, z=50.0
    (54, 49): 'Xe103',  # Species with n=54.0, z=49.0
    (54, 50): 'Xe104',  # Species with n=54.0, z=50.0
    (55, 49): 'Cs104',  # Species with n=55.0, z=49.0
    (55, 50): 'Cs105',  # Species with n=55.0, z=50.0
    (56, 49): 'Ba105',  # Species with n=56.0, z=49.0
    (18, 18): 'Ar36',   # Species with n=18.0, z=18.0
    (31, 29): 'Ga60',   # Species with n=31.0, z=29.0
    (31, 30): 'Ge61',   # Species with n=31.0, z=30.0
    (30, 31): 'Ga61',   # Species with n=30.0, z=31.0
    (31, 31): 'Ge62',   # Species with n=31.0, z=31.0
    (32, 30): 'As62',   # Species with n=32.0, z=30.0
    (32, 31): 'Ge63',   # Species with n=32.0, z=31.0
    (30, 32): 'As62',   # Species with n=30.0, z=32.0
    (31, 32): 'As63',   # Species with n=31.0, z=32.0
    (32, 32): 'Se64',   # Species with n=32.0, z=32.0
    (33, 31): 'Br64',   # Species with n=33.0, z=31.0
    (33, 32): 'Kr65',   # Species with n=33.0, z=32.0
    (32, 33): 'Br65',   # Species with n=32.0, z=33.0
    (33, 33): 'Kr66',   # Species with n=33.0, z=33.0
    (34, 32): 'Kr66',   # Species with n=34.0, z=32.0
    (34, 33): 'Rb67',   # Species with n=34.0, z=33.0
    (33, 34): 'Kr67',   # Species with n=33.0, z=34.0
    (16, 19): 'K35',    # Species with n=16.0, z=19.0
    (20, 23): 'Sc43',   # Species with n=20.0, z=23.0
    (32, 34): 'Rb68',   # Species with n=32.0, z=34.0
    (35, 34): 'Kr69',   # Species with n=35.0, z=34.0
    (35, 35): 'Br70',   # Species with n=35.0, z=35.0
    (35, 36): 'Br71',   # Species with n=35.0, z=36.0
    (36, 35): 'Kr71',   # Species with n=36.0, z=35.0
        (37, 36): 'Kr73',    # Species with n=37.0, z=36.0
    (37, 37): 'Rb74',    # Species with n=37.0, z=37.0
    (37, 38): 'Sr75',    # Species with n=37.0, z=38.0
    (38, 37): 'Rb75',    # Species with n=38.0, z=37.0
    (39, 38): 'Sr77',    # Species with n=39.0, z=38.0
    (38, 39): 'Y77',     # Species with n=38.0, z=39.0
    (39, 39): 'Y78',     # Species with n=39.0, z=39.0
    (38, 40): 'Zr78',    # Species with n=38.0, z=40.0
    (39, 40): 'Zr79',    # Species with n=39.0, z=40.0
    (40, 39): 'Y79',     # Species with n=40.0, z=39.0
    (41, 40): 'Zr81',    # Species with n=41.0, z=40.0
    (41, 41): 'Nb82',    # Species with n=41.0, z=41.0
    (41, 42): 'Mo83',    # Species with n=41.0, z=42.0
    (42, 41): 'Nb83',    # Species with n=42.0, z=41.0
    (43, 42): 'Mo85',    # Species with n=43.0, z=42.0
    (43, 43): 'Tc86',    # Species with n=43.0, z=43.0
    (43, 44): 'Ru87',    # Species with n=43.0, z=44.0
    (44, 43): 'Tc87',    # Species with n=44.0, z=43.0
    (44, 44): 'Ru88',    # Species with n=44.0, z=44.0
    (45, 43): 'Tc88',    # Species with n=45.0, z=43.0
    (45, 44): 'Ru89',    # Species with n=45.0, z=44.0
    (45, 45): 'Rh90',    # Species with n=45.0, z=45.0
    (46, 44): 'Ru90',    # Species with n=46.0, z=44.0
    (46, 45): 'Rh91',    # Species with n=46.0, z=45.0
    (45, 46): 'Pd91',    # Species with n=45.0, z=46.0
    (46, 46): 'Pd92',    # Species with n=46.0, z=46.0
    (47, 45): 'Rh92',    # Species with n=47.0, z=45.0
    (47, 46): 'Pd93',    # Species with n=47.0, z=46.0
    (48, 45): 'Rh93',    # Species with n=48.0, z=45.0
    (48, 46): 'Pd94',    # Species with n=48.0, z=46.0
    (47, 47): 'Ag94',    # Species with n=47.0, z=47.0
    (48, 47): 'Ag95',    # Species with n=48.0, z=47.0
    (48, 48): 'Cd96',    # Species with n=48.0, z=48.0
    (49, 47): 'Ag96',    # Species with n=49.0, z=47.0
    (49, 48): 'Cd97',    # Species with n=49.0, z=48.0
    (50, 47): 'Ag97',    # Species with n=50.0, z=47.0
    (50, 49): 'In99',    # Species with n=50.0, z=49.0
    (51, 48): 'Cd99',    # Species with n=51.0, z=48.0
    (51, 49): 'In100',   # Species with n=51.0, z=49.0
    (51, 50): 'Sn101',   # Species with n=51.0, z=50.0
    (52, 49): 'In101',   # Species with n=52.0, z=49.0
    (52, 48): 'Cd100',   # Species with n=52.0, z=48.0
    (56, 50): 'Sn106',   # Species with n=56.0, z=50.0
    (38, 36): 'Kr74',    # Species with n=38.0, z=36.0
    (40, 38): 'Sr78',    # Species with n=40.0, z=38.0
    (42, 40): 'Zr82',    # Species with n=42.0, z=40.0
    (45, 42): 'Mo87',    # Species with n=45.0, z=42.0
    (46, 43): 'Tc89',    # Species with n=46.0, z=43.0
    (11, 11): 'Na22',    # Species with n=11.0, z=11.0
    (15, 15): 'P30',     # Species with n=15.0, z=15.0
    (14, 18): 'Ar32',    # Species with n=14.0, z=18.0
    (23, 27): 'Co50',    # Species with n=23.0, z=27.0
    (27, 30): 'Zn57',    # Species with n=27.0, z=30.0
    (21, 25): 'Mn46',    # Species with n=21.0, z=25.0
    (21, 26): 'Fe47',    # Species with n=21.0, z=26.0
    (50, 50): 'Sn100',   # Species with n=50.0, z=50.0
    (8, 7): 'N15',       # Species with n=8.0, z=7.0
    (9, 8): 'O17',       # Species with n=9.0, z=8.0
    (10, 8): 'O18',      # Species with n=10.0, z=8.0
    (10, 9): 'F19',      # Species with n=10.0, z=9.0
    (11, 10): 'Ne21',    # Species with n=11.0, z=10.0
    (12, 11): 'Na23',    # Species with n=12.0, z=11.0
    (12, 10): 'Ne22',    # Species with n=12.0, z=10.0
    (13, 12): 'Mg25',    # Species with n=13.0, z=12.0
    (13, 13): 'Al26',    # Species with n=13.0, z=13.0
    (14, 12): 'Mg26',    # Species with n=14.0, z=12.0
    (14, 13): 'Al27',    # Species with n=14.0, z=13.0
    (15, 14): 'Si29',    # Species with n=15.0, z=14.0
    (53, 49): 'Sb102', # Species with n=53.0, z=49.0
    (53, 50): 'Sb103', # Species with n=53.0, z=50.0
    (54, 49): 'Te103', # Species with n=54.0, z=49.0
    (54, 50): 'Te104', # Species with n=54.0, z=50.0
    (55, 49): 'I104',  # Species with n=55.0, z=49.0
    (55, 50): 'I105',  # Species with n=55.0, z=50.0
    (56, 49): 'Xe105', # Species with n=56.0, z=49.0
    (36, 32): 'Br68',  # Species with n=36.0, z=32.0
    (38, 34): 'Kr72',  # Species with n=38.0, z=34.0
    (46, 40): 'Rh86',  # Species with n=46.0, z=40.0
    (55, 48): 'Xe103', # Species with n=55.0, z=48.0
    (56, 47): 'I103',  # Species with n=56.0, z=47.0
    (58, 48): 'Xe105', # Species with n=58.0, z=48.0
    (57, 47): 'I104',  # Species with n=57.0, z=47.0
    (53, 45): 'Sb98',  # Species with n=53.0, z=45.0
    (54, 48): 'Te103', # Species with n=54.0, z=48.0
    (55, 47): 'I103',  # Species with n=55.0, z=47.0
    (57, 49): 'Xe106', # Species with n=57.0, z=49.0
    (58, 47): 'I105',  # Species with n=58.0, z=47.0
}



# Initialize lists to store data from all files
all_data = []

# Assuming all your data files are in a directory called 'data_files'
for filename in glob.glob('flow/*.dat'):
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        # Extract header information
        header = lines[1].strip().split()
        time = float(header[0])
        temp = float(header[1])
        dens = float(header[2])
        
        # Skip the first three lines (header)
        reaction_data = lines[4:]
        
        # Read reaction data into a DataFrame
        df = pd.read_csv(StringIO(''.join(reaction_data)), 
                        delim_whitespace=True, 
                        names=['nin', 'zin', 'yin', 'nout', 'zout', 'yout', 'flow'])

        
        # Add header info to DataFrame
        df['time'] = time
        df['temp'] = temp
        df['dens'] = dens
        
        # Append to the list
        all_data.append(df)

# Concatenate all data into a single DataFrame
data = pd.concat(all_data, ignore_index=True)

print("DATA")
print(data)


# Group by time and sum the flows
flow_by_time = data.groupby('time')['flow'].sum().reset_index()

##Top Reactions##

# Identify top reactions at each time point
#TODO:why not zin and zout?
top_reactions = data.groupby(['time', 'nin', 'nout'])['flow'].sum().reset_index()
top_reactions = top_reactions.sort_values(['time', 'flow'], ascending=[True, False])
#print(top_reactions)



##Top Reactions at Each Timestep##


# Define a function to get species label
def get_species_label(n, z):
    return species_names.get((n, z), f"n{int(n)}, z{int(z)}")

# Initialize a list to store the results
results = []

# Get all unique time points and sort them
time_points = sorted(data['time'].unique())

# Process data for each time point
for time_point in time_points:
    # Filter data for the current time point
    current_data = data[data['time'] == time_point]
    
    # Sort reactions by flow in descending order
    current_data = current_data.sort_values(by='flow', ascending=False)
    
    # Select the top 10 reactions
    top_reactions = current_data.head(10)
    
    # Format reactions and store results
    for idx, row in top_reactions.iterrows():
        reactant_label = get_species_label(row['nin'], row['zin'])
        product_label = get_species_label(row['nout'], row['zout'])
        reaction_str = f"{reactant_label} → {product_label}"
        results.append({
            'Time': time_point,
            'Reaction': reaction_str,
            'Flow': row['flow']
        })

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Optional: Pivot the table to have reactions as columns (not requested but available)
# pivot_df = results_df.pivot(index='Time', columns='Reaction', values='Flow')

with open('top_reactions_by_time.txt', 'w') as f:
    # Write a header
    f.write("Top 10 Reactions at Each Time Step\n")
    f.write("="*40 + "\n\n")
    
    # Group the results by time
    grouped = results_df.groupby('Time')
    
    for time_point, group in grouped:
        f.write(f"Time: {time_point}\n")
        f.write("-"*40 + "\n")
        for idx, row in group.iterrows():
            f.write(f"Reaction: {row['Reaction']}, Flow: {row['Flow']}\n")
        f.write("\n")





##Graph Network Attempt##

# Filter reactions with flow greater than 1e-5
significant_flows = data[data['flow'] > 1e-2]

print("SIGNIFICANT FLOWS")
print(significant_flows)

import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Add edges based on significant reactions
for idx, row in significant_flows.iterrows():
    source = (row['nin'], row['zin'])
    target = (row['nout'], row['zout'])
    G.add_edge(source, target, weight=row['flow'])

# Calculate node sizes (sum of yin and yout for each species)
node_sizes = {}
all_nodes = set(G.nodes())

for node in all_nodes:
    yin_sum = significant_flows[(significant_flows['nin'] == node[0]) & (significant_flows['zin'] == node[1])]['yin'].sum()
    yout_sum = significant_flows[(significant_flows['nout'] == node[0]) & (significant_flows['zout'] == node[1])]['yout'].sum()
    node_sizes[node] = yin_sum + yout_sum

# Scale node sizes for visualization
max_node_size = 1000
max_node_value = max(node_sizes.values()) if node_sizes else 1  # Avoid division by zero
node_size_values = [max_node_size * (node_sizes[node] / max_node_value) for node in G.nodes()]

# Scale edge widths for visualization
max_edge_width = 5
max_flow = significant_flows['flow'].max() if not significant_flows.empty else 1  # Avoid division by zero
edge_widths = [max_edge_width * (G[u][v]['weight'] / max_flow) for u, v in G.edges()]


# Create node labels using the species_names dictionary
node_labels = {node: species_names.get(node, f"n{node[0]}, z{node[1]}") for node in G.nodes()}

# Identify nodes without labels in species_names
unlabeled_nodes = [node for node in G.nodes() if node not in species_names]
if unlabeled_nodes:
    print("Warning: The following species are not in species_names and will be labeled with their (n, z) values:")
    for node in unlabeled_nodes:
        print(f"Species with n={node[0]}, z={node[1]}")

pos = nx.spring_layout(G, k=0.5, iterations=50)

plt.figure(figsize=(12, 8))

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=node_size_values, node_color='lightblue')

# Draw edges
nx.draw_networkx_edges(G, pos, width=edge_widths, arrowstyle='->', arrowsize=15)

# Draw labels
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12)

plt.title('Reaction Network with Flow > 1e-5')
plt.axis('off')
plt.show()



#Graph incorporating time#



##GOOD But not quite##

# Get all unique time points and sort them
#time_points = sorted(data['time'].unique())

# import matplotlib.animation as animation

# # Initialize the figure and axis
# fig, ax = plt.subplots(figsize=(12, 8))

# def update_graph(time_point):
#     ax.clear()
#     # Filter data for the current time point
#     current_data = data[data['time'] == time_point]
#     significant_flows = current_data[current_data['flow'] > 1e-3]
    
#     # Create a directed graph
#     G = nx.DiGraph()
    
#     # Add edges based on significant reactions
#     for idx, row in significant_flows.iterrows():
#         source = (row['nin'], row['zin'])
#         target = (row['nout'], row['zout'])
#         G.add_edge(source, target, weight=row['flow'])
    
#     # Calculate node sizes
#     node_sizes = {}
#     all_nodes = set(G.nodes())
#     for node in all_nodes:
#         yin_sum = significant_flows[(significant_flows['nin'] == node[0]) & (significant_flows['zin'] == node[1])]['yin'].sum()
#         yout_sum = significant_flows[(significant_flows['nout'] == node[0]) & (significant_flows['zout'] == node[1])]['yout'].sum()
#         node_sizes[node] = yin_sum + yout_sum
    
#     # Scale node sizes
#     max_node_size = 1000
#     max_node_value = max(node_sizes.values()) if node_sizes else 1  # Avoid division by zero
#     node_size_values = [max_node_size * (node_sizes[node] / max_node_value) for node in G.nodes()]
    
#     # Scale edge widths
#     max_edge_width = 5
#     max_flow = significant_flows['flow'].max() if not significant_flows.empty else 1  # Avoid division by zero
#     edge_widths = [max_edge_width * (G[u][v]['weight'] / max_flow) for u, v in G.edges()]
    
#     # Create node labels using species_names
#     node_labels = {node: species_names.get(node, f"n{node[0]}, z{node[1]}") for node in G.nodes()}
    
#     # Generate positions for the nodes (consistent layout)
#     pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
#     # Draw the graph
#     nx.draw_networkx_nodes(G, pos, node_size=node_size_values, node_color='lightblue', ax=ax)
#     nx.draw_networkx_edges(G, pos, width=edge_widths, arrowstyle='->', arrowsize=15, ax=ax)
#     nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, ax=ax)
    
#     ax.set_title(f'Reaction Network at Time = {time_point:.2e}')
#     ax.axis('off')

# ani = animation.FuncAnimation(fig, update_graph, frames=time_points, repeat=False)

# # To display the animation in a Jupyter notebook
# #from IPython.display import HTML
# #HTML(ani.to_jshtml())

# ani.save('reaction_network_animation.mp4', writer='ffmpeg', fps=2)


##Attempt 2.0##


# Define the overall significant threshold
overall_threshold = 5e-5

# Filter data to get all significant reactions over all time
overall_significant_flows = data[data['flow'] > overall_threshold]

# Get all unique nodes involved in significant reactions
all_nodes = set(overall_significant_flows.apply(lambda row: (row['nin'], row['zin']), axis=1)).union(
            set(overall_significant_flows.apply(lambda row: (row['nout'], row['zout']), axis=1))
)

# Create a base graph with all significant nodes
G_base = nx.DiGraph()
G_base.add_nodes_from(all_nodes)

# Calculate max_flow over all time points for consistent scaling
max_flow = overall_significant_flows['flow'].max() if not overall_significant_flows.empty else 1  # Avoid division by zero
max_edge_width = 8

# Calculate max_node_value over all time points
all_node_sizes = {}
for node in all_nodes:
    yin_sum = overall_significant_flows[(overall_significant_flows['nin'] == node[0]) & (overall_significant_flows['zin'] == node[1])]['yin'].sum()
    yout_sum = overall_significant_flows[(overall_significant_flows['nout'] == node[0]) & (overall_significant_flows['zout'] == node[1])]['yout'].sum()
    total_abundance = yin_sum + yout_sum
    all_node_sizes[node] = total_abundance

# Global maximum node value
max_node_value = max(all_node_sizes.values()) if any(all_node_sizes.values()) else 1

# Compute positions once
pos = nx.spring_layout(G_base, k=0.5, iterations=50, seed=42)

# Prepare the time points
time_points = sorted(data['time'].unique())
# Initialize the figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

def update_graph(time_point):
    ax.clear()
    # Filter data for the current time point
    current_data = data[data['time'] == time_point]
    significant_flows = current_data[current_data['flow'] > overall_threshold]
    
    # Create a graph for current time point, starting from base graph
    G = G_base.copy()
    G.remove_edges_from(list(G.edges()))  # Remove any existing edges
    
    # Add edges based on significant reactions at current time
    for idx, row in significant_flows.iterrows():
        source = (row['nin'], row['zin'])
        target = (row['nout'], row['zout'])
        G.add_edge(source, target, weight=row['flow'])
    
    # Calculate node sizes based on current data
    node_sizes = {}
    for node in all_nodes:
        yin_sum = current_data[(current_data['nin'] == node[0]) & (current_data['zin'] == node[1])]['yin'].sum()
        yout_sum = current_data[(current_data['nout'] == node[0]) & (current_data['zout'] == node[1])]['yout'].sum()
        total_abundance = yin_sum + yout_sum
        node_sizes[node] = total_abundance
    
    # Scale node sizes
    max_node_size = 2000
    min_node_size = 30  # Ensure nodes are visible
    node_size_values = [
        max(min_node_size, max_node_size * (node_sizes[node] / max_node_value)) for node in G.nodes()
    ]
    
    # Calculate edge widths
    edge_widths = []
    for u, v in G.edges():
        edge_weight = G[u][v]['weight']
        edge_widths.append(max_edge_width * (edge_weight / max_flow))
    
    # Create node labels
    node_labels = {node: species_names.get(node, f"n{node[0]}, z{node[1]}") for node in G.nodes()}
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size_values, node_color='lightblue', ax=ax)
    
    # Draw edges
    # Draw edges with darker color
    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        arrowstyle='->',
        arrowsize=25,
        edge_color='black',  # Set edge color to black
        ax=ax
    )
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, ax=ax)
    
    ax.set_title(f'Reaction Network at Time = {time_point:.2e}')
    ax.axis('off')

ani = animation.FuncAnimation(fig, update_graph, frames=time_points, repeat=False)
ani.save('reaction_network_animation.mp4', writer='ffmpeg')






##Plot a specific reaction##

# Select a specific reaction (e.g., nin=1, nout=2)
reaction = data[(data['nin'] == 1) & (data['nout'] == 2)]

# Plot flow over time
plt.figure(figsize=(10, 6))
plt.plot(reaction['time'], reaction['flow'], marker='o')
plt.xlabel('Time')
plt.ylabel('Flow')
plt.title('Reaction Flow Over Time for Reaction 1 -> 2')
plt.show()



#Correlation#

# Correlation matrix
correlation = data[['flow', 'temp', 'dens']].corr()
print(correlation)

# Scatter plot of flow vs temperature
plt.scatter(data['temp'], data['flow'])
plt.xlabel('Temperature')
plt.ylabel('Flow')
plt.title('Flow vs Temperature')
plt.show()

# Scatter plot of flow vs density
plt.scatter(data['dens'], data['flow'])
plt.xlabel('Density')
plt.ylabel('Flow')
plt.title('Flow vs Density')
plt.show()


from scipy.stats import pearsonr

corr_temp_flow, _ = pearsonr(data['temp'], data['flow'])
corr_dens_flow, _ = pearsonr(data['dens'], data['flow'])
print(f"Correlation between Temperature and Flow: {corr_temp_flow}")
print(f"Correlation between Density and Flow: {corr_dens_flow}")


import statsmodels.api as sm

# Predict flow based on temperature and density
X = data[['temp', 'dens']]
y = data['flow']
X = sm.add_constant(X)  # Adds a constant term to the predictor
model = sm.OLS(y, X).fit()
predictions = model.predict(X)
print(model.summary())



