import numpy as np
import gymnasium
from gymnasium import spaces
import math
import pandas as pd
from csv import writer
import config
import random
from scipy.interpolate import CubicSpline
import utils

class Reactor(gymnasium.Env):

    def __init__(self, experiment_name=None):
        # Observation Space
        # time, enzymes, cells, substrate, substrate/cells
        self.observation_space  = spaces.Box(low=np.array([0,0]), high = np.array([math.inf, math.inf]), dtype=np.float64)

        # Action Space
        self.action_space = spaces.Box(low=0, high=0.1)

        #self.action_space = spaces.Discrete(2)
        #self.action_space = spaces.Box(low = np.array([0,0]), high=np.array([2,1]), dtype=np.int32) 
        #self.action_space = spaces.MultiDiscrete(nvec=[3, 2], start=[-1 , 0], dtype=np.int16)
        
        self.experiment_name = experiment_name
        
        # CSV File
        self.df = pd.DataFrame(columns=config.TRAINING_DATA_LOG_COLUMNS)
        self.df.to_csv(f"experiments/{self.experiment_name}/{config.TRAINING_DATA_LOGS_FILENAME}", index=True)  

        self.experiment_number = config.EXPERIMENT_NUMBER

        self.weibull_values = pd.read_csv('pdfcsv.csv')
        self.xvalues = self.weibull_values['x']
        self.y_values = self.weibull_values['pdf']

        # Create a cubic spline interpolation model
        self.cs = CubicSpline(self.xvalues, self.y_values)


    def reset(self, seed=None, options=None):        
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # ==============================================    Reactor Setup    ======================================================

       # Initial Tank conditions
        #self.X0 = config.X0  # g/L
        #self.S0 = config.S0 # mol/L
        self.X0 = round(random.uniform(0.7, 0.9),3) # g/L 
        self.S0 = round(random.uniform(0.005, 0.007),3) # mol/L
        self.E0 = config.E0 # U/L  
    
        # Process conditions
        self.T = config.INIT_TEMPRATURE #'C
        self.A = config.INIT_AGITATION # RPM

        # model parameters
        self.Ks = config.KS
        self.Yxs = config.YXS
        self.MuE_opt = config.MUE_OPT #round(random.uniform(0.05, 0.15),3)
        self.mu_max = config.MU_MAX #round(random.uniform(0.1, 0.3),3)
        self.del_t = config.DEL_T
        self.t_end = config.T_END
        self.total_sim_steps = int(self.t_end/self.del_t)
        self.tvec = [self.del_t + i * self.del_t for i in range(self.total_sim_steps)]
        self.ns = len(self.tvec)
        
        # Substrate addition calculations
        self.tank_capacity = config.TANK_CAPACITY # liters
        self.substrate_in_tank_liters = config.SUBSTRATE_IN_TANK_LITERS # Liters
        self.max_substrate_limit_liters = config.MAX_SUBSTRATE_LIMIT_LITERS # L
        self.substrate_transfer_amount_liters = config.SUBSTRATE_TRANSFER_AMOUNT_LITERS # L
        self.media_transfer_gap =  config.MEDIA_TRANSFER_GAP # Hours this is after 10 steps ie. 6 minutes
        self.media_transfer_step = int(self.media_transfer_gap/self.del_t)
        self.sub_in_tank_moles = self.S0 * self.substrate_in_tank_liters 

        # external Substrate tank configuration
        self.ext_tank_substrate_conc = config.EXT_TANK_SUBSTRATE_CONC # mol/L
        self.substrate_transfer_moles = self.substrate_transfer_amount_liters * self.ext_tank_substrate_conc # moles

        self.simulation_timestep = config.SIMULATION_TIMSTEP
        # =============================================================================================================

        '''  
             0              1               2              3                   4                5           6           7          8          9          10       11     12        13
        "timestep","experiment_number","biomass","substrate_in_tank","enzyme_activity","temperature","feeding_action","reward", "change", "distance", "nochange", "t4 ", "t5", "flow_volume"

        '''
        # Create experiment table using above headers
        self.D = np.zeros((self.ns+1, 14))

        # Get all values of a certain column
        self.timestep = self.D[:,0]
        self.experiment_index = self.D[:,1]
        self.biomass = self.D[:,2]
        self.substrate = self.D[:,3]
        self.enzyme_activity = self.D[:,4]
        self.temperature = self.D[:,5]
        self.feeding_action = self.D[:,6]
        self.reward = self.D[:,7]
        self.change = self.D[:,8]
        self.dist = self.D[:,9]
        self.nochange = self.D[:,10]
        self.t4 = self.D[:,11]
        self.t5 = self.D[:,12]
        self.flow_vol = self.D[:13]

        # initialise the column values with inital conditions ie row 1
        self.timestep[0] = self.simulation_timestep
        self.experiment_index[0] = self.experiment_number
        self.biomass[0] = self.X0
        self.substrate[0] = self.S0
        self.enzyme_activity[0] = self.E0
        self.temperature[0] = self.T
        self.feeding_action[0] = 0
        self.reward[0] = 0
        

        self.cell_death_timer = config.CELL_DEATH_TIMER
        self.cell_death_hour = config.CELL_DEATH_TIME        
        self.intervention_time = config.INTERVENTION_TIME  # hours
        self.intervention_step = int(self.intervention_time/self.del_t)
        self.cell_death_rate = config.CELL_DEATH_RATE
        self.terminate = False
        self.step_called = False
        self.tank_is_full = False
        self.first_reward = True
        self.max_slope = 0
        self.exp_slopes = []
        self.max_rew_sub_stop = 0

        self.current_e_activ = 0.0
        self.before_e_activ =  self.E0

        # Clear contents of plotting file
        with open(f"experiments/{self.experiment_name}/plot.txt",'w') as f:
            f.truncate(0)
            f.close()

        # return : Observation, information
        # time, enzymes
        return np.array([
            self.simulation_timestep, 
            self.enzyme_activity[self.simulation_timestep]
            ]),{}


    def step(self, action):
        while True:

            #---------------------------------------------------------------------------------------#
            #                                   Simluation 
            #---------------------------------------------------------------------------------------#
            if self.step_called:
                # Get the action from RL 
                substrate_action = action[0]
                self.feeding_action[self.simulation_timestep] = substrate_action
            
            # =========================== Cell Production ===========================
            # Rate of cell production based on substrate
            MuX = utils.cell_growth_rate(self.substrate[self.simulation_timestep])
            # Cells produced
            dXdt = utils.cells_produced(self.biomass[self.simulation_timestep], MuX)
            # =======================================================================

            # =========================== Substrate Consumption ===========================
            dSdt = utils.substrate_consumed(dXdt)
            # =============================================================================

            # =========================== Update Cell Concentration ===========================
            # Change in cells 
            delX = dXdt * self.del_t
            # Update cell concentration
            self.biomass[self.simulation_timestep + 1] = self.biomass[self.simulation_timestep] + delX
            # =============================================================================

            # =========================== Check no change in cell concentration =========================== 
            if dXdt == 0:
                self.cell_death_timer += 1
            else:
                self.cell_death_timer = 0
            # =================================================================================
        
            # =========================== Substrate Addition ===========================
            # Update substrate
            if self.step_called:
                # check tank is full
                if self.substrate_in_tank_liters < self.max_substrate_limit_liters:
                    # Add substrate
                    self.substrate[self.simulation_timestep], self.substrate_in_tank_liters = utils.add_substrate(self.substrate[self.simulation_timestep], substrate_action, self.substrate_in_tank_liters)
                # Tank is full
                else:
                    self.tank_is_full = True
                # Make step False till new action is taken
                self.step_called = False
            
            # Calculate change in substrate
            delS = dSdt * self.del_t
            # Update Substrate concentration
            # Check if substrate is less than or close to 0
            if self.substrate[self.simulation_timestep] + delS < 0.000001:
                self.substrate[self.simulation_timestep + 1] = 0
            else:
                self.substrate[self.simulation_timestep + 1] = self.substrate[self.simulation_timestep] + delS
            # =======================================================================


            # =========================== Enzyme Production ===========================
            # Enzyme determination 
            sub_cell_ratio = ( self.substrate[self.simulation_timestep] / self.biomass[self.simulation_timestep] ) * 1e6
            # Check if any cells were produced in this timestep
            if dXdt == 0:
                MuE = 0
            # Get rate of enzyme production based on the substrate to cell ratio value
            else:
                weibull = utils.enzyme_production_rate(sub_cell_ratio, self.cs)
                MuE = self.MuE_opt * weibull
            
            # calculate Rate of enzyme production
            
            # Amount of enzymes produced
            dEdt = utils.enzymes_produced(self.biomass[self.simulation_timestep], MuE)
            # Update enzyme variable
            self.enzyme_activity[self.simulation_timestep + 1] = self.enzyme_activity[self.simulation_timestep] + dEdt
            # =======================================================================
            
            
            self.simulation_timestep += 1
            self.timestep[self.simulation_timestep] = self.simulation_timestep
            self.experiment_index[self.simulation_timestep] = self.experiment_number

            #---------------------------------------------------------------------------------------#
            #                                   Termination Conditions
            #---------------------------------------------------------------------------------------#
            # Check if Episode is over
            if self.simulation_timestep == int(self.ns - 2):
                self.terminate = True
                
                break
            
            # Terminate cells are not produced for more than 2 hours
            if self.cell_death_timer >= self.cell_death_hour/self.del_t:
                self.terminate = True

                break

            # Terminate if tank is full and substrate has reached 0
            if self.tank_is_full and dXdt == 0:
                self.terminate = True

                break

            # INTERVENTION TIME: Break the loop when its time to take an action
            if (self.simulation_timestep) % (self.intervention_step) == 0 and self.simulation_timestep != 0 and self.tank_is_full == False: #and self.substrate_in_tank_liters < self.max_substrate_limit_liters:
                self.step_called = True

                break

            #---------------------------------------------------------------------------------------#

        # -------------------------------------------------------------------------------
        # ------------------------------ Reward ----------------------------------------- 
        # -------------------------------------------------------------------------------
        
        # ------------- Change in enzymes -----------------
        self.current_e_activ = self.enzyme_activity[self.simulation_timestep]
        change = (self.current_e_activ - self.before_e_activ) * 1000
        self.before_e_activ = self.current_e_activ
        # ------------- No change in enzymes -----------------
        if change <= 0:
            nochange = -1
        else:
            nochange = 0
        # ------------- Calculate Reward -----------------
        reward = change + nochange + self.enzyme_activity[self.simulation_timestep]
        #print(f" change is : {change} and no change is: {nochange} reward is: {reward}")
        # --------------- Write it in csv -----------------
        self.change[self.simulation_timestep] = change
        self.reward[self.simulation_timestep] = reward
    

        # Save plot.txt
        with open(f'{f"experiments/{self.experiment_name}"}/plot.txt','a') as plotting_file:
            plotting_file.write(f"{self.tvec[self.simulation_timestep]},{self.biomass[self.simulation_timestep]},{self.substrate[self.simulation_timestep]},{self.enzyme_activity[self.simulation_timestep]},{self.T},{reward}\n")
            plotting_file.close()

        # Next episode/experiement
        if self.terminate:
            self.experiment_number += 1
            df_info = pd.DataFrame(self.D)
            df_imp = df_info.head(self.simulation_timestep)
            df_imp.to_csv(f"experiments/{self.experiment_name}/{config.TRAINING_DATA_LOGS_FILENAME}", mode='a',header=False, index=True)
        
        return (
            np.array([
            self.simulation_timestep, 
            self.enzyme_activity[self.simulation_timestep]
            ]), 
            reward, 
            self.terminate, 
            False, 
            {}
        )