'''
Simulation with parameters used from papers
- https://link.springer.com/article/10.1023/A:1008840904246


Parameters required from literature
- X0 
- S0
- Ks
- mu_max
- mue_max
- Yxs

From the above paper, parameters found for glucoamylase production 
- X0:      **10^5 cells/ml**       Converted to   **0.7 CDWg/L**     (Estimated conversion)
- S0 :     **5 m mol/L**           converted to   **0.005 mol/L**   
- Mu_max : **0.2 /h**
- Yxs :    **210 CDW g/mol**  

From https://static-content.springer.com/esm/art%3A10.1007%2Fs12257-020-0153-z/MediaObjects/12257_2020_153_MOESM1_ESM.pdf parameters found 
- Ks :      **0.1 g/L**                      converted to    **0.0005 mol/L**   (Calculated conversion, molecular weight of glucose = 180.156 g/mol)
- Mue_max : **8.865 Lactate g/CDW g**        converted to    **8.865 U/CDW g**    (Needs to be adjusted)

'''

# ---------------------------------- training logs csv file ------------------------------------------------

TRAINING_DATA_LOGS_FILENAME = "training_data.csv"
TRAINING_DATA_LOG_COLUMNS = ["timestep","experiment_number","biomass","substrate_in_tank","enzyme_activity","temperature","feeding_action","reward","change","distance","nochange","t4","t5","flow_volume"]
EXPERIMENT_NAME = "sac_target3.5"

# ---------------------------------- Reactor Parameters ------------------------------------------------

EXPERIMENT_NUMBER = 1 # hours
SIMULATION_TIMSTEP = 0 
X0 = 0.7  # CDW g/L 
S0 = 0.005 # mol/L
E0 = 0.0 # U/L 

INIT_TEMPRATURE = 30 #'C 
OPTIMUM_TEMPERATURE = 30 #'C
OPT_TEMP_RANGE = 2
INIT_AGITATION = 250 # RPM
OPT_SUB_CELL_RATIO = 0.004
SCR_VIABLE_RANGE = 2000

# model parameters
KS = 0.0003    # mol/L
YXS = 210   # CDW g/mol
MUE_OPT = 0.1    # U/CDW g
MU_MAX = 0.2  # /h
DEL_T = 0.01 # hours ie. 36 seconds
T_END = 24*4 # H
KL = 0.0001 # mol/L
S_MAX = 0.05 
S_MIN = 0.0185

# Logistic curve parameters
L = MU_MAX  # Adjust based on maximum cell production rate
K = 100    # Steepness, higher means faster drop-off
T0 = 0.07 # Midpoint (where the production rate is half of L)

# Substrate addition calculations
TANK_CAPACITY = 2 # L
SUBSTRATE_IN_TANK_LITERS = 0.5 # Liters
MAX_SUBSTRATE_LIMIT_LITERS = 1.5 # L
SUBSTRATE_TRANSFER_AMOUNT_LITERS = 0.05 # L
MEDIA_TRANSFER_GAP =  0.1 # Hours this is after 10 steps ie. 6 minutes

# external substrate tank configurations
EXT_TANK_SUBSTRATE_CONC = 0.05 # mol/L

# ------------------------------------- OTHER PARAMETERS ------------------------------------------------

CELL_DEATH_TIMER = 0
CELL_DEATH_TIME = 1 # hours
INTERVENTION_TIME = 0.1 
INTERVENTION_STEP = int(INTERVENTION_TIME/DEL_T)
CELL_DEATH_RATE = 0.010
TARGET_ENZYME_ACTIVTIY = 4
TEMP_CHANGE = 0.1

# ------------------------------------- RL ALGO PARAMETERS ------------------------------------------------
MODELS_PATH = "model"
MODEL = "SAC"
BEST_MODEL_NAME = "best_model.zip"
