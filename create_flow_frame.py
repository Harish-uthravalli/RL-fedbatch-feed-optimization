import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import random 
import config
from scipy.interpolate import CubicSpline
import utils
import os

# Initial conditions
X0 = config.X0  # CDW g/L 
S0 = config.S0 # mol/L
E0 = config.E0 # U/L 

# Process conditions
T = config.INIT_TEMPRATURE #'C 
T_opt = config.OPTIMUM_TEMPERATURE #'C
A = config.INIT_AGITATION # RPM

# model parameters
Ks = config.KS    # mol/L
Yxs =  config.YXS  # CDW g/mol
MuE_opt = config.MUE_OPT    # U/CDW g
mu_max = config.MU_MAX   # /h
del_t = config.DEL_T # hours ie. 36 seconds
t_end = config.T_END
total_sim_steps = int(t_end/del_t)
tvec = [del_t + i * del_t for i in range(total_sim_steps)]
ns = len(tvec)
kl = config.KL # mol/L
cell_death_timer = config.CELL_DEATH_TIMER
cell_death_time = config.CELL_DEATH_TIME

# Substrate addition calculations
tank_capacity = config.TANK_CAPACITY # L
substrate_in_tank_liters = config.SUBSTRATE_IN_TANK_LITERS # L
max_substrate_limit_liters = config.MAX_SUBSTRATE_LIMIT_LITERS # L
substrate_transfer_amount_liters = config.SUBSTRATE_TRANSFER_AMOUNT_LITERS # L 
media_transfer_gap =  config.MEDIA_TRANSFER_GAP # Hours this is after 10 steps ie. 6 minutes
media_transfer_step = int(media_transfer_gap/del_t)
substrate_concentration = S0
sub_in_tank_moles = S0 * substrate_in_tank_liters # mol

# external media tank configurations
ext_tank_substrate_conc = config.EXT_TANK_SUBSTRATE_CONC # mol/L
substrate_transfer_moles = substrate_transfer_amount_liters * ext_tank_substrate_conc # mol 

# X S E delE delX, muE
D = np.zeros((ns+1, 6))
D[0][0] = X0
D[0][1] = substrate_concentration
D[0][2] = E0

for i in range(ns):
    X = D[i][0]
    S = D[i][1]
    E = D[i][2]

    MuX = utils.cell_growth_rate(S)
    # new cells that are generated
    dXdt = utils.cells_produced(X, MuX)

    # SUbstrate consumption
    dSdt = utils.substrate_consumed(dXdt)

    # Find change in cells 
    delX = dXdt * del_t
    # Update cells
    D[i+1][0] = X + delX
    
    # Find change in substrate
    delS = dSdt * del_t

    # # Update substrate
    if substrate_in_tank_liters < max_substrate_limit_liters:
        if i%(media_transfer_step) == 0 and i != 0:
            substrate_action = 0.01
            if True:
                # get substrate concentraion for current timestep
                substrate_concentration = S
                sub_in_tank_moles = substrate_concentration * substrate_in_tank_liters
                # Add substrate and calculate new concentraion
                substrate_in_tank_liters = substrate_in_tank_liters + substrate_action # Liters of media
                sub_in_tank_moles = sub_in_tank_moles + ext_tank_substrate_conc * substrate_action  # grams
                substrate_concentration = sub_in_tank_moles/substrate_in_tank_liters # substrate grams/liter
                S = substrate_concentration

    # Check if substrate is less than or close to 0
    if S + delS < 0.000001:
        D[i+1][1] = 0
    else:
        D[i+1][1] = S + delS

    # Enzyme determination 
    sub_cell_ratio = (S/X) * 1e6

    weibull = utils.get_weibull_y_value(sub_cell_ratio, peak= config.OPT_SUB_CELL_RATIO*1e6)
    MuE = MuE_opt * weibull

    if dXdt == 0:
        MuE = 0

    # new enzyme from fresh cells
    delE = MuE * X * del_t
    
    # Update enzyme variable
    D[i+1][2] = E + delE

    # terminate if tank capacity is full and cells start dying
    if substrate_in_tank_liters >= max_substrate_limit_liters and dXdt == 0:
        break


import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Parameters
initial_volume = 9e-5       # Starting population
ramp_rate = 0.1             # Growth rate (e.g., 20% per iteration)

# Initialize variables
flow_volume = initial_volume
volumes = []
time_vector = tvec

# Simulate exponential growth
for i in range(ns):
    if i % media_transfer_step == 0 and i != 0:
        if sum(volumes) <= 1.5:
            flow_volume = flow_volume * math.exp(ramp_rate)  # Increase population exponentially
            volumes.append(flow_volume)
            # Set global font to Times New Roman
            plt.rcParams["font.family"] = "DejaVu Sans"

            # Plotting
            plt.figure(figsize=(7, 5))  # Set figure size
            plt.plot(time_vector[:len(volumes)], np.array(volumes)*1e3)
            plt.xlim(0, 8)
            plt.ylim(0, 120)

            # Add labels with fontsize 15
            plt.xlabel("Time (h)", fontsize=15)
            plt.ylabel("Feed Volume (mL)", fontsize=15)

            # Define custom legend
            custom_lines = [
                Line2D([0], [0], color='none', marker=None, linestyle='None', label=f'Initial Feed Volume : 0.78 (mL)'),
                Line2D([0], [0], color='none', marker=None, linestyle='None', label=fr'Feed Ramp Factor : 0.170 (h$^{{-1}}$)'),
                Line2D([0], [0], color='none', marker=None, linestyle='None', label=f'Feeding Interval : 4.434 (min)'),
            ]

            # Add the custom legend
            plt.xticks(fontsize=15)  # Set x-tick font size
            plt.yticks(fontsize=15)  # Set y-tick font size
            plt.legend(
            handles=custom_lines,
            fontsize=20,
            handlelength=0,
            handletextpad=0,
            loc="upper left"  # Move legend to top left
            )            
            figure_path = os.path.join('flow_frames',f'figure_{i}.png')
            plt.savefig(figure_path, bbox_inches="tight", dpi=300)
           
        else:
            volumes.append(0)
            break
    else:
        volumes.append(0)




