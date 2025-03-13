import numpy as np
import matplotlib.pyplot as plt
import config
import utils
import os
from stable_baselines3 import SAC
import utils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style



    
best_model_path = os.path.join("experiments","sac_evalcb",'model',"best_model.zip" )
loaded_model = SAC.load(best_model_path)

num_experiments = 1

for experiment in range(1, num_experiments+1):
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
                model_input = np.array([i, E, X])
                action = loaded_model.predict(model_input, deterministic=False)
                substrate_action = action[0][0]
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

        weibull = utils.get_weibull_y_value(sub_cell_ratio, peak=config.OPT_SUB_CELL_RATIO * 1e6)
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
 
        X = D[:,0][0:i]
        S = D[:,1][0:i]
        E = D[:,2][0:i]
        S_C_R = (S/X)
        
        if i%(media_transfer_step) == 0 and i != 0:
            import matplotlib.pyplot as plt
            import os
            import imageio

            # Create folders if they don't exist
            os.makedirs("xse_plots", exist_ok=True)
            os.makedirs("scr_plots", exist_ok=True)

            # Save the first plot (X, S, E)
            xse_path = os.path.join("xse_plots", f"experiment_{experiment:02d}_{i}.png")
            fig, ax1 = plt.subplots(figsize=(7, 5))  # Ensure fig is assigned
            fig.suptitle(f"Enzyme Activity: {round(E[-1],2)}")
            ax1.plot(tvec[0:i], E, color="red", label="Enzyme Activity U/L")
            ax1.set_ylim(0, 3.5)  # Set Y-axis scale for Enzyme Activity
            ax1.set_xlim(0, 15)

            ax2 = ax1.twinx()
            ax2.plot(tvec[0:i], S * 1e3, color="orange", label="Substrate mmol/L")
            ax2.set_ylim(0, 25)  # Set Y-axis scale for Substrate

            ax3 = ax1.twinx()
            ax3.plot(tvec[0:i], X, color="blue", label="Cells CDW g/L")
            ax3.spines["right"].set_position(("axes", 1.20))
            ax3.set_ylim(0, 11)  # Set Y-axis scale for Cells CDW

            ax1.set_ylabel("Enzyme Activity (U/L)", color="red", fontsize=25)
            ax1.set_xlabel("Time (h)", fontsize=25)
            ax2.set_ylabel("Substrate (mmol/L)", color="orange", fontsize=25)
            ax3.set_ylabel("Cells CDW (g/L)", color="blue", fontsize=25)

            ax1.tick_params(axis="y", colors="red", labelsize=25)
            ax2.tick_params(axis="y", colors="orange", labelsize=25)
            ax3.tick_params(axis="y", colors="blue", labelsize=25)
            ax1.tick_params(axis="x", labelsize=25)

            ax2.spines["right"].set_color("orange")
            ax3.spines["right"].set_color("blue")
            ax3.spines["left"].set_color("red")

            plt.savefig(xse_path, bbox_inches="tight", dpi=300)
            plt.close(fig)  # 

            # Save the second plot (Substrate-to-Cell Ratio)
            scr_path = os.path.join("scr_plots", f"experiment_{experiment:02d}_{i}.png")
            fig, ax = plt.subplots()  # Assign fig to ensure proper closing

            ax.plot(tvec[: len(S_C_R)], S_C_R * 1e3, label="Substrate to cell ratio (mmol/g)")
            ax.set_xlim(0, 15)
            ax.set_ylim(0, 8)

            ax.set_xlabel("Time (h)", fontsize=25)
            ax.set_ylabel(r"$\frac{S}{X}$ (mmol/g)", fontsize=25)
            ax.tick_params(axis="both", labelsize=25)
            #ax.legend(fontsize=25)
            ax.axhline(y=config.OPT_SUB_CELL_RATIO * 1e3, linestyle="--", color="black")

            plt.savefig(scr_path, bbox_inches="tight", dpi=300)
            plt.close(fig)  # 



import imageio.v2 as imageio  # Use v2 to avoid the deprecation warning
import os
from PIL import Image

def create_combined_gif(folder1, folder2, output_gif, fps=20):
    images = []
    files1 = sorted(os.listdir(folder1))
    files2 = sorted(os.listdir(folder2))

    # Ensure both have the same number of frames
    num_frames = min(len(files1), len(files2))

    # Get the size of one image
    first_image_path = os.path.join(folder1, files1[0])
    with Image.open(first_image_path) as img:
        width, height = img.size
        target_size = (width, height)

    for i in range(num_frames):
        img1_path = os.path.join(folder1, files1[i])
        img2_path = os.path.join(folder2, files2[i])

        with Image.open(img1_path) as img1, Image.open(img2_path) as img2:
            img1 = img1.resize(target_size)
            img2 = img2.resize(target_size)

            # Create a new blank image (double width for side-by-side)
            combined_img = Image.new("RGB", (2 * width, height))
            combined_img.paste(img1, (0, 0))
            combined_img.paste(img2, (width, 0))

            images.append(combined_img)

    imageio.mimsave(output_gif, images, fps=fps, loop=0)  # loop=0 for infinite looping

# Create a single GIF combining both plots
os.makedirs("gifs", exist_ok=True)
create_combined_gif("xse_plots", "scr_plots", "gifs/combined_animation.gif")


