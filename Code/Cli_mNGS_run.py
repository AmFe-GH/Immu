import os
import random
import optuna
from models import *
from utils import *
from trainer import *
import torch
import pandas as pd
from datetime import datetime
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
os.chdir(current_dir)

#Ensure the logs directory exists
logs_dir = './logs'
os.makedirs(logs_dir, exist_ok=True)
# set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Generate a timestamped filename
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_filename = os.path.join(logs_dir, f'optuna_{timestamp}.log')

# Basic logging configuration
logging.basicConfig(
    filename=log_filename, 
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)

# Get the Optuna logger and set handlers
optuna_logger = optuna.logging.get_logger("optuna")
file_handler = logging.FileHandler(log_filename)
stream_handler = logging.StreamHandler(sys.stdout)

# Optional: Set a format for the handlers
formatter = logging.Formatter('%(asctime)s %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add handlers to the Optuna logger
optuna_logger.addHandler(file_handler)
optuna_logger.addHandler(stream_handler)

# To avoid duplicate logs due to multiple handlers
optuna_logger.propagate = False
logging.basicConfig(filename='optuna.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')
optuna.logging.get_logger("optuna").addHandler(
    logging.FileHandler('optuna.log'))
optuna.logging.get_logger("optuna").addHandler(
    logging.StreamHandler(sys.stdout))

levellist = ['T', 'S', 'G', 'F']
level = 'S'

merged_df = pd.read_excel(f'../Data/cli_mNGS.BS.{level}.xlsx')
features_df_cli = merged_df.loc[:, 'Gender':'TreatmentType_4']
features_df_mNGS = merged_df.iloc[:,
                                  merged_df.columns.get_loc('Responce2_SD')+1:]
mNGS_cols = features_df_mNGS.columns
cli_cols = features_df_cli.columns

assert no_all_zero_col(features_df_mNGS)
assert no_all_zero_col(features_df_cli)


labels_df = merged_df.loc[:, 'Responce2_CR-PR':'Responce2_SD']
features_cli,features_mNGS,labels = df2tensor(features_df_cli,features_df_mNGS,labels_df)

#Keep the dimensions consistent
Scaling_factor = features_mNGS.max(axis=1)[0].mean()
features_mNGS = features_mNGS/Scaling_factor

seed_nums = [1, 14, 43, 320]
epochs = 300
reduciton_times = 20
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

objective = objective_mul_param(features_cli, features_mNGS, labels,
                                seed_nums, device, reduciton_times, epochs, cli_cols, 
                                mNGS_cols,model_type='gated_mlp',tag = timestamp)

study = optuna.create_study(direction='maximize')
try:
    study.optimize(objective.objective, n_jobs=1, n_trials=100)
except KeyboardInterrupt:
    print("Optimization interrupted by KeyboardInterrupt")
trial_df = study.trials_dataframe()
trial_df.to_csv(f'./result/{timestamp}/trial_results.csv')

