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
###########################log setting#################################

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
#########################Data loading#################################
level = 'S'
top_count = 80
merged_df = pd.read_excel(f'../Data/cli_mNGS_meta_name.{level}.xlsx')
features_df_meta = merged_df.loc[:, 'Uric acid':'PG(a-13:0/i-18:0)']


selected_mNGS = pd.read_csv(f'../Data/result_best_performance/{level}_level/NGS_marker.csv')['feature_name'][:top_count]
features_df_mNGS = merged_df[selected_mNGS]
mNGS_cols = features_df_mNGS.columns
meta_cols = features_df_meta.columns


labels_df = merged_df.loc[:, 'Responce2_CR-PR':'Responce2_SD']
features_meta,features_mNGS,labels = df2tensor(features_df_meta,features_df_mNGS,labels_df)

###############Keep the dimensions consistent#######################
Scaling_factor = features_mNGS.max(axis=1)[0].mean()
features_mNGS = features_mNGS/Scaling_factor

Scaling_factor = features_meta.max(axis=1)[0].mean()
features_meta = features_meta/Scaling_factor
####################################################################

seed_nums = [1, 14, 43, 320]
epochs = 300
reduciton_times = 20
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

objective = objective_meta(features_mNGS, features_meta, labels,
                                seed_nums, device, reduciton_times, epochs,mNGS_cols , 
                                meta_cols,model_type='gated_mlp',tag = timestamp)

study = optuna.create_study(direction='maximize')
try:
    study.optimize(objective.objective, n_jobs=1, n_trials=100)
except KeyboardInterrupt:
    print("Optimization interrupted by KeyboardInterrupt")
trial_df = study.trials_dataframe()
trial_df.to_csv(f'./result/{timestamp}/trial_results.csv')

# import torch
# import pandas as pd
# import optuna
# from models import *
# from utils import *
# from trainer import *
# from datetime import datetime
# import random
# current_file_path = os.path.abspath(__file__)
# current_dir = os.path.dirname(current_file_path)
# os.chdir(current_dir)

# # Ensure the logs directory exists
# logs_dir = './logs'
# os.makedirs(logs_dir, exist_ok=True)
# # set random seed
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# # Generate a timestamped filename
# timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# log_filename = os.path.join(logs_dir, f'optuna_{timestamp}.log')

# # Basic logging configuration
# logging.basicConfig(
#     filename=log_filename, 
#     level=logging.INFO,
#     format='%(asctime)s %(message)s'
# )

# # Get the Optuna logger and set handlers
# optuna_logger = optuna.logging.get_logger("optuna")
# file_handler = logging.FileHandler(log_filename)
# stream_handler = logging.StreamHandler(sys.stdout)

# # Optional: Set a format for the handlers
# formatter = logging.Formatter('%(asctime)s %(message)s')
# file_handler.setFormatter(formatter)
# stream_handler.setFormatter(formatter)

# # Add handlers to the Optuna logger
# optuna_logger.addHandler(file_handler)
# optuna_logger.addHandler(stream_handler)

# # To avoid duplicate logs due to multiple handlers
# optuna_logger.propagate = False
# logging.basicConfig(filename='optuna.log', level=logging.INFO,
#                     format='%(asctime)s %(message)s')
# optuna.logging.get_logger("optuna").addHandler(
#     logging.FileHandler('optuna.log'))
# optuna.logging.get_logger("optuna").addHandler(
#     logging.StreamHandler(sys.stdout))

# levellist = ['T', 'S', 'G', 'F']
# level = 'G'
# T_dic = {
#     'ranking_data_path':'../Data/result_best_performance/T_level/trial96_NGS_ranking.csv',
#     'reduction_index':18,
# }
# S_dic = {
#     'ranking_data_path':'../Data/result_best_performance/S_level/trial74_NGS_ranking.csv',
#     'reduction_index':16,
# }
# G_dic = {
#     'ranking_data_path': ['../Data/result_best_performance/G_level/trial30_cli_ranking.csv','../Data/result_best_performance/G_level/trial30_NGS_ranking.csv'],
#     'reduction_index':11+1, 
# }
# dic_set = {'T':T_dic,
#            'S':S_dic,
#            'G':G_dic
#            }

# merged_data = pd.read_excel(f'../Data/cli_mNGS_meta.CLR.CBT.{level}.xlsx')
# filted_markers = get_markers_from_paths(dic_set,level)
# features_df_filted = merged_data[filted_markers]
# filted_cols = features_df_filted.columns
# features_df_meta = merged_data.iloc[:,
#                     merged_data.columns.get_loc('L-a-Lysophosphatidylserine'):] #Metabolic data start from this column
# meta_cols = features_df_meta.columns

# assert no_all_zero_col(features_df_filted)
# assert no_all_zero_col(features_df_meta)


# labels_df = merged_data.loc[:, 'Responce2_CR-PR':'Responce2_SD']
# features_filted,features_meta,labels = df2tensor(features_df_filted,features_df_meta,labels_df)
# seed_nums = [1, 14, 43, 320]
# epochs = 300
# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# reduciton_times = 20
# objective = objective_meta(features_filted, features_meta,labels,
#                                 seed_nums, device, reduciton_times, epochs, filted_cols, 
#                                 meta_cols,model_type='attn',tag = timestamp)

# study = optuna.create_study(direction='maximize')
# try:
#     study.optimize(objective.objective, n_jobs=1, n_trials=100)
# except KeyboardInterrupt:
#     print("Optimization interrupted by KeyboardInterrupt")
# trial_df = study.trials_dataframe()
# trial_df.to_csv(f'./result/{timestamp}/trial_results.csv')