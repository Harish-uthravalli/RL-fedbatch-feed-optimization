import json
import argparse
import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from gymnasium.envs.registration import register
import config
import shutil
import os
import torch

def get_random_action():
    return random.randint(0, 100)

# get amount of cells produced
def cells_produced(X, MuX):
    return MuX * X 

# calculate the amount of substrate consumed
def substrate_consumed(dXdt):
    return -(1/config.YXS) * dXdt

# calculate amount of enzymes produced
def enzymes_produced(X, MuE):
    return MuE * X * config.DEL_T

# # Get the rate of cell growth 
# def cell_growth_rate(S):
#     S = round(S,3)
#     if S > 0.002 and S < 0.01: 
#         rate = cell_production_rate(S)
#     elif S >= 0.01 and S < 0.035:
#         rate = logistic(S)
#     elif S <= 0.002 or S >= 0.035:
#         rate = 0

    
#     return rate

def reward_function(current_enzyme_activity, target_activity=4):
    # Calculate the absolute distance to the target
    distance = abs(target_activity - current_enzyme_activity)
    
    # Reward inversely proportional to the distance
    # Add a small epsilon to avoid division by zero when distance is very small
    epsilon = 1e-5
    reward = 1 / (distance + epsilon)
    
    # Optionally scale or normalize the reward to avoid extreme values
    reward = min(reward, 10)  # Cap the reward at 100 for stability
    
    return reward

# Get the rate of cell growth 
def cell_growth_rate(S):
    if S < 0.002 or S > 0.030:
            rate = 0
    else:
        rate = config.MU_MAX
    
    return rate

# Get the enzyme production rate
def enzyme_production_rate(scr, cs):
    if scr > 9000 or scr < 100:
        weibull = 0
    else:
        weibull = cs(scr)
    return weibull

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Substrate addition logic
def add_substrate(S, substrate_action, substrate_in_tank_liters):
    # get substrate concentration for current timestep
    substrate_concentration = S
    sub_in_tank_moles = substrate_concentration * substrate_in_tank_liters
    # Add substrate and calculate new concentraion
    substrate_in_tank_liters = substrate_in_tank_liters + substrate_action # Liters of media
    sub_in_tank_moles = sub_in_tank_moles + config.EXT_TANK_SUBSTRATE_CONC * substrate_action  # mol
    substrate_concentration = sub_in_tank_moles/substrate_in_tank_liters # substrate grams/liter
    return substrate_concentration, substrate_in_tank_liters

# Plot final substrate, cell and enzyme concentrations
def save_plot(filename,title, tvec, X, S, E):
    
    fig, ax1 = plt.subplots(figsize=(8,6), constrained_layout=True)
    plt.title(title)
    # Plot Enzyme Activity on ax1
    ax1.plot(tvec[:len(E)], E, color="red", label="Enzyme Activity U/L")
    ax1.set_ylabel("Enzyme Activity U/L", color="red")
    ax1.set_xlabel("Time (hours)")
    ax1.tick_params(axis='y', colors="red")

    # Create ax2 for Substrate (shares x-axis with ax1)
    ax2 = ax1.twinx()
    ax2.plot(tvec[:len(S)], S, color="orange", label="Substrate mol/L")
    ax2.set_ylabel("Substrate mol/L", color="orange")
    ax2.tick_params(axis='y', colors="orange")
    ax2.spines['right'].set_color("orange")

    # Create ax3 for Cells and position it outward to the right of ax2
    ax3 = ax1.twinx()
    ax3.plot(tvec[:len(X)], X, color="blue", label="Cells CDW g/L")
    ax3.set_ylabel("Cells CDW g/L", color="blue")
    ax3.tick_params(axis='y', colors="blue")
    ax3.spines['right'].set_position(('outward', 60))  # Move ax3 to the right
    ax3.spines['right'].set_color("blue")

    # Adjust legend placement to avoid overlap
    #fig.legend(loc="upper left", bbox_to_anchor=(0.1, 1))

    plt.savefig(filename)
    plt.close()

def plot_scr(filename,tvec, ratio):
    ratio = ratio * 1e6 
    plt.plot(tvec, ratio, label="Substrate to cell ratio")
    plt.axhline(y=config.OPT_SUB_CELL_RATIO*1e6)
    plt.grid(True)
    plt.xlabel("Time h")
    plt.ylabel("S/X")
    plt.legend()

    plt.savefig(filename)
    plt.close()

def calculate_rmse(values, optimum):
    """
    Calculate the Root Mean Squared Error (RMSE) between a list of values and an optimum value.

    Parameters:
    values (list or array-like): List of numerical values.
    optimum (float): The target or optimum value.

    Returns:
    float: The RMSE value.
    """
    mse = np.mean([(v - optimum) ** 2 for v in values])
    rmse = np.sqrt(mse)
    return rmse

def calculate_probe(substrate, current_e, prev_e):
    change = current_e - prev_e
    print("Change: ",change)
    print("Subsrate: ", substrate)
    if change <= 0:
        if substrate < 0.002:
            sub_action = 0.03
        elif substrate > 0.030: 
            sub_action = 0
        else:
            sub_action = 0.030
    else:
        sub_action = 0.03

    return sub_action

def calculate_scaled_distance(current_activity, target_activity):
    # Use the target activity as the maximum distance
    max_distance = target_activity
    # Calculate the absolute distance
    distance = abs(target_activity - current_activity)
    # Scale the distance to a range of 0 to 10
    scaled_distance = (distance / max_distance) * 10
    # Ensure the scaled distance is capped at 10
    return min(scaled_distance, 10)


def copy_files(files):
    for file in files:
        file_path = os.path.join('copy_scripts', file)
        shutil.copy(file_path, f"experiments/{config.EXPERIMENT_NAME}")

def copy_animation_files(experiment_name):
    files = ["animate_scr.py", "animate_XSE.py"]
    # Copy the file to the destination folder
    for file in files:
        source_file = os.path.join("important_scripts", file)
        shutil.copy(source_file, f"experiments/{experiment_name}")

def copy_analysis_nb(experiment_name):
    file = "important_scripts/training_analysis.ipynb"
    shutil.copy(file, f"experiments/{experiment_name}")

def monod(self, S, Ks=config.KS):
        return config.MU_MAX * (S/(Ks + S))
    
def cell_production_rate(sub_conc, R_max=0.20, S_0=0.0045, k=5000):
    """
    Calculate the rate of cell production based on a sigmoid curve.

    Parameters:
    sub_conc (float): Substrate concentration (mol/L).
    R_max (float): Maximum rate of cell production (default is 0.20).
    S_0 (float): Midpoint substrate concentration, where rate is half of R_max (default is 0.025 mol/L).
    k (float): Steepness of the curve. A higher value makes the curve steeper (default is 50).

    Returns:
    float: Rate of cell production.
    """
    # Sigmoid curve formula for rate of cell production
    rate = R_max / (1 + np.exp(-k * (sub_conc - S_0)))
    return rate

def logistic(substrate, L=config.MU_MAX, k=5000, t0=0.0325):
    """
    Logistic function for cell production rate where:
    - substrate: input concentration (mol/L)
    - L: maximum production rate (0.20 in this case)
    - k: steepness of the curve (adjust for desired fall-off)
    - t0: midpoint concentration (adjust to match the substrate range)
    """
    return L * (1 - (1 / (1 + np.exp(-k * (substrate - t0)))))
    