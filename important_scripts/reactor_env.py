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
        self.observation_space  = spaces.Box(low=np.array([0,0,0]), high = np.array([math.inf, math.inf,math.inf]), dtype=np.float64)

        # Action Space
        self.action_space = spaces.Box(low=0, high=0.1)

        #self.action_space = spaces.Discrete(2)
        #self.action_space = spaces.Box(low = np.array([0,0]), high=np.array([2,1]), dtype=np.int32) 
        #self.action_space = spaces.MultiDiscrete(nvec=[3, 2], start=[-1 , 0], dtype=np.int16)
        
        self.experiment_name = experiment_name
        
        # CSV File
        self.df = pd.DataFrame(columns=config.TRAINING_DATA_LOG_COLUMNS)
        self.df.to_csv(config.TRAINING_DATA_LOGS_FILENAME, index=True)  

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
        with open("plot.txt",'w') as f:
            f.truncate(0)
            f.close()

        
        # return : Observation, information
        # time, enzymes
        return np.array([
            self.simulation_timestep, 
            self.enzyme_activity[self.simulation_timestep],
            self.substrate_in_tank_liters
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
            
            
            if self.substrate[self.simulation_timestep] < 0.002 or self.substrate[self.simulation_timestep] > 0.03:
                MuX = 0
            else:
                MuX = 0.2

            # Cells produced
            dXdt = MuX * self.biomass[self.simulation_timestep]
            # Substrate consumed
            dSdt = -(1/self.Yxs) * dXdt

            # Change in cells and substrate using Euler
            delX = dXdt * self.del_t 
            delS = dSdt * self.del_t # mol/L

            # Update cells for next timestep
            
            self.biomass[self.simulation_timestep + 1] = self.biomass[self.simulation_timestep] + delX
            
            # Update substrate
            if self.step_called:
                if self.substrate_in_tank_liters < self.max_substrate_limit_liters:
                    if substrate_action:
                        # if self.simulation_timestep == 0 or self.simulation_timestep == 1:
                        #     print("Substrate added")
                        self.tank_is_full = False
                        cur_sub_conc = self.substrate[self.simulation_timestep]
                        self.sub_in_tank_moles = cur_sub_conc * self.substrate_in_tank_liters
                        # Adjust the substrate concentration based on the continuous flow volume action
                        #self.substrate_in_tank_liters += self.substrate_transfer_amount_liters
                        self.substrate_in_tank_liters += substrate_action
                        #self.sub_in_tank_moles += self.substrate_transfer_amount_liters * self.ext_tank_substrate_conc
                        self.sub_in_tank_moles += substrate_action * self.ext_tank_substrate_conc
                        substrate_concentration = self.sub_in_tank_moles / self.substrate_in_tank_liters
                        self.substrate[self.simulation_timestep] = substrate_concentration  
                        #print("Substrate Concentration Change: ", self.substrate[self.simulation_timestep])
                    # else:
                    #     if self.simulation_timestep == 0 or self.simulation_timestep == 1:
                    #         print("substrate not added")
                else:
                    self.tank_is_full = True

                self.step_called = False
            
            # Updating Substrate consumption
            if (self.substrate[self.simulation_timestep] + delS) < 0.000001:
                self.substrate[self.simulation_timestep+1] = 0
            else:
                self.substrate[self.simulation_timestep+1] = self.substrate[self.simulation_timestep] + delS

            # Cells start dying if no substrate for more than x hours
            if self.substrate[self.simulation_timestep+1] == 0:
                self.cell_death_timer += 1
            else:
                self.cell_death_timer = 0

            # # if cell_death_timer == 2 hours then cells start dying
            # if int(self.cell_death_timer) >= int(self.cell_death_hour/self.del_t):
            #     self.biomass[self.simulation_timestep+1] = self.biomass[self.simulation_timestep+1] - (self.biomass[self.simulation_timestep+1]*self.cell_death_rate)
                
            # Enzyme determination 
            #print(f"substrate in tank: {self.substrate[self.simulation_timestep]} and cells in tank: {self.biomass[self.simulation_timestep]} at timestep: {self.simulation_timestep}")
            sub_cell_ratio = self.substrate[self.simulation_timestep]/self.biomass[self.simulation_timestep]
            sub_cell_ratio = sub_cell_ratio * 1e6
            if sub_cell_ratio > 11000:
                MuE = 0
            else:
                MuE = self.MuE_opt * self.cs(sub_cell_ratio)

            # if self.simulation_timestep == 0 or self.simulation_timestep == 1:
            #     print(f"Timestep : {self.simulation_timestep} S_C_R: {sub_cell_ratio}")
            
            # new enzyme from fresh cells
            if dXdt == 0:
                MuE = 0
            delE = MuE * self.biomass[self.simulation_timestep] * self.del_t 
            
            # Update enzyme variable
            self.enzyme_activity[self.simulation_timestep + 1] = self.enzyme_activity[self.simulation_timestep] + delE
            
            self.simulation_timestep += 1
            self.timestep[self.simulation_timestep] = self.simulation_timestep
            self.experiment_index[self.simulation_timestep] = self.experiment_number
            self.temperature[self.simulation_timestep] = self.T
            #---------------------------------------------------------------------------------------#
            #                                   Termination Conditions
            #---------------------------------------------------------------------------------------#
            # Check if Episode is over
            if self.simulation_timestep == int(self.ns - 2):
                self.terminate = True
                
                break

            # after no substrate cell will become 0
            if self.biomass[self.simulation_timestep] <= 0.00001:
                self.terminate = True
                
                break
        
            # # terminate if tank capacity is full and cells start dying
            # if self.substrate_in_tank_liters >= self.max_substrate_limit_liters and int(self.cell_death_timer) >= int(self.cell_death_hour/self.del_t):
            #     self.terminate = True
                
            #     break

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
        change = (self.current_e_activ - self.before_e_activ)
        self.before_e_activ = self.current_e_activ

        # ------------- No change in enzymes -----------------
        if change <= 0:
            nochange = - 0.001
            pos_change = 0
        else:
            nochange = 0
            pos_change = 10

        # ------------- Distance from the target enzymes -----------------
        dist =( 1/ (config.TARGET_ENZYME_ACTIVTIY - self.current_e_activ) ) * 100
        
        # ------------- Inverse of Time -----------------
        time_inverse = (1/self.simulation_timestep) * 100
    
        # ------------- Calculate Reward -----------------
        reward = change + nochange 

        # --------------- Write it in csv -----------------
        self.change[self.simulation_timestep] = change
        self.dist[self.simulation_timestep] = dist 
        self.nochange[self.simulation_timestep] = nochange 
        self.reward[self.simulation_timestep] = reward
    

        # Save plot.txt
        with open('plot.txt','a') as plotting_file:
            plotting_file.write(f"{self.tvec[self.simulation_timestep]},{self.biomass[self.simulation_timestep]},{self.substrate[self.simulation_timestep]},{self.enzyme_activity[self.simulation_timestep]},{self.T},{reward}\n")
            plotting_file.close()

        # Next episode/experiement
        if self.terminate:
            self.experiment_number += 1
            df_info = pd.DataFrame(self.D)
            df_imp = df_info.head(self.simulation_timestep)
            df_imp.to_csv(config.TRAINING_DATA_LOGS_FILENAME, mode='a',header=False, index=True)
        
        return (
            np.array([
            self.simulation_timestep, 
            self.enzyme_activity[self.simulation_timestep],
            self.substrate_in_tank_liters
            ]), 
            reward, 
            self.terminate, 
            False, 
            {}
        )