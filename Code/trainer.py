import numpy as np
import torch
from models import Mixed_model, validate,Attention_model,Gated_model,Attention_model_v2
from utils import split_train_val, check_tensor, combined_ranking, get_part_ranking
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import shap
import copy
import logging
import optuna
import sys
import pandas as pd
import os
import threading
import queue


class objective_mul_param:
    def __init__(self, features_cli, features_mNGS, labels, seed_nums,
                 device, reduction_times, epochs, cli_cols, mNGS_cols,model_type,tag):
        self.features_cli = features_cli
        self.features_mNGS = features_mNGS
        self.labels = labels
        self.tag = tag
        os.makedirs(f'result/{self.tag}', exist_ok=True)
        self.hidden_layers_cli = [
            # [50, 100, 200, 100, 50, 20, 10, 10],
            # [50, 100, 200, 100, 50, 20, 5, ],
            #[40, 80, 160, 80, 40, 20, 10,],
            #[40, 80, 160, 80, 40, 10, 5],
            #[30, 60, 140, 70, 25, 10],
            [30, 60, 140, 70, 10, 5],
            [20, 40, 80, 20, 10],
            [20, 40, 80, 15, 5],
            [20, 10, 10, 5],
            [20, 10],
            [0,0,0],
            
            
        ]
        self.hidden_layers_NGS = [
            #[200, 150, 100, 75, 50, 25, 10, 10, 10],
            #[200, 100, 50, 20, 10, 5],
            #[200, 100, 10, 10],
            #[100, 70, 35, 20, 10],
            [50, 50, 50, 10, 10, 10, 10, 10],
            [50, 30, 10,10],
            [30, 10, 10],
            [20, 10, 10],
            [10, 10, 5],
            [10, 5],
            [0,0,0],
            
        ]
        outputsize = labels.shape[-1]
        self.hidden_layers_agg = [[20, 10, 10, 10, 10, outputsize],
                                  [20, 10, outputsize],
                                  [10, outputsize],
                                  [5,outputsize]
                                  ]
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.reduction_ratio_cli = 0.9
        self.reduction_ratio_NGS = 0.8
        self.seed_nums = seed_nums
        self.device = device
        self.reduction_times = reduction_times
        self.epochs = epochs
        self.cli_cols = cli_cols
        self.mNGS_cols = mNGS_cols
        self.model_type = model_type
        
        logging.info(
                f"model_type:{self.model_type} device:{self.device}, reduction_times:{self.reduction_times}, epochs:{self.epochs}"
            )
    def objective(self, trial):
        auc_per_reduction = []
        
        cli_layer_scale=trial.suggest_int('cli layer scale',0,len(self.hidden_layers_cli)-1)
        ngs_layer_scale=trial.suggest_int('NGS layer scale',0,len(self.hidden_layers_NGS)-1)
        agg_layer_scale=trial.suggest_int('aggregation layer scale',0,len(self.hidden_layers_agg)-1)
        
        selec_layers_cli = self.hidden_layers_cli[cli_layer_scale]
        selec_layers_mNGS = self.hidden_layers_NGS[ngs_layer_scale]
        selec_layers_agg = self.hidden_layers_agg[agg_layer_scale]
        
        # selec_layers_cli = trial.suggest_categorical(
        #     'selec_layers_cli', self.hidden_layers_cli)
        # selec_layers_mNGS = trial.suggest_categorical(
        #     'selec_layers_mNGS', self.hidden_layers_NGS)
        # selec_layers_agg = trial.suggest_categorical(
        #     'selec_layers_agg', self.hidden_layers_agg)
        dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.3)
        lr = 0.015
        
        k_fold_num = 5
        
        if selec_layers_cli[-1]==0 and selec_layers_mNGS[-1]==0:
            return np.float32(0.5)
        
        features_cli_reduc = self.features_cli
        features_mNGS_reduc = self.features_mNGS
        temp_cli_cols = self.cli_cols.tolist()
        temp_mNGS_cols = self.mNGS_cols.tolist()

        # df to store the importance of each feature in different reduction
        df_cli_reduction = pd.DataFrame( columns=self.cli_cols)
        df_NGS_reduction = pd.DataFrame( columns=self.mNGS_cols)

        
        for reduciton_iter in range(self.reduction_times):
            logging.info(
                f"Features of Trial {trial.number} Reductuon {reduciton_iter} process: \
                clincal: {features_cli_reduc.shape[-1]}, mNGS: {features_mNGS_reduc.shape[-1]}"
            )

            layers_cli = [features_cli_reduc.shape[-1]] +  selec_layers_cli
            layers_NGS = [features_mNGS_reduc.shape[-1]] + selec_layers_mNGS
            layers_agg = [layers_cli[-1] + layers_NGS[-1]] + selec_layers_agg

            val_auc_list = []
            cli_rank_list = []
            NGS_rank_list = []

            # multi-thread computing
            if self.model_type=='mlp' or self.model_type=='gated_mlp':
                for seed_num in self.seed_nums:
                    val_auc, cli_rank, NGS_rank = K_fold_cross_validation_DL(
                        k_fold_num, self.loss_fn, layers_cli, layers_NGS, layers_agg,
                        self.epochs, features_cli_reduc, features_mNGS_reduc, self.labels, seed_num, 
                        self.device, lr, dropout_rate,self.model_type
                    )
                    val_auc_list.append(val_auc)
                    cli_rank_list.append(cli_rank)
                    NGS_rank_list.append(NGS_rank)
            elif self.model_type=='attn' or self.model_type=='gated_attn':
                
                if self.model_type=='attn':
                    nhead = trial.suggest_int('nhead',1,2)
                    expands = trial.suggest_int('expands',1,2)
                elif self.model_type=='gated_attn':
                    nhead = trial.suggest_int('nhead',1,5)
                    expands = trial.suggest_int('expands',1,5)
                
                for seed_num in self.seed_nums:
                    val_auc, cli_rank, NGS_rank,attn_weight = K_fold_cross_validation_DL(
                        k_fold_num, self.loss_fn, layers_cli, layers_NGS, layers_agg,
                        self.epochs, features_cli_reduc, features_mNGS_reduc, self.labels, 
                        seed_num, self.device, lr, dropout_rate,self.model_type,
                        expands,nhead
                    )
                    if attn_weight is not None:  # attn_weight will be None when  selec_layers_cli[-1]==0 or selec_layers_mNGS[-1]==0
                        attn_weight = attn_weight.mean(axis=0)
                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        im = ax.imshow(attn_weight, cmap=plt.cm.hot_r)
                        cbar = fig.colorbar(im)
                        cbar.set_label('Attention Weight')
                        plt.savefig(f'./result/{self.tag}/trial{trial.number}_redc_{reduciton_iter}_{seed_num}_attn_weight.png')
                        plt.close()
                        assert isinstance(temp_cli_cols, list) and isinstance(temp_mNGS_cols, list)
                        temp_cols = temp_cli_cols+temp_mNGS_cols
                        Attention_df = pd.DataFrame(attn_weight, columns=temp_cols, index=temp_cols)
                        Attention_df.to_excel(f'./result/{self.tag}/trial{trial.number}_redc_{reduciton_iter}_{seed_num}_Attention.xlsx')
                    
                    val_auc_list.append(val_auc)
                    cli_rank_list.append(cli_rank)
                    NGS_rank_list.append(NGS_rank)
            else:
                raise TypeError
            

            
            
            auc_single_reduction = np.mean(val_auc_list)
            logging.info(f"auc_single_reduction:  {auc_single_reduction}")
            auc_per_reduction.append(auc_single_reduction)
            
            
            combined_rank_cli, combined_scores_cli = combined_ranking(
                cli_rank_list, temp_cli_cols)
            df_cli_reduction = add_to_dataframe(
                df_cli_reduction, temp_cli_cols, combined_scores_cli)
            
            cli_selec_fea = get_part_ranking(
                combined_rank_cli, ratio=self.reduction_ratio_cli)
            temp_cli_cols = [temp_cli_cols[index] for index in cli_selec_fea]
            features_cli_reduc = features_cli_reduc[:, cli_selec_fea]
            
            
            combined_rank_NGS, combined_scores_NGS = combined_ranking(
                NGS_rank_list, temp_mNGS_cols)
            df_NGS_reduction = add_to_dataframe(
                df_NGS_reduction, temp_mNGS_cols, combined_scores_NGS)
            
            
            NGS_selec_fea = get_part_ranking(
                combined_rank_NGS, ratio=self.reduction_ratio_NGS)
            temp_mNGS_cols = [temp_mNGS_cols[index] for index in NGS_selec_fea]
            features_mNGS_reduc = features_mNGS_reduc[:,NGS_selec_fea]

            if reduciton_iter==5 and auc_per_reduction[-1]<0.6 and auc_per_reduction[-2]<0.6:
                logging.info(f"Triggering the early stopping strategy")
                break
           
            
        df_NGS_reduction.to_csv(f'result/{self.tag}/trial{trial.number}_NGS_ranking.csv')
        df_cli_reduction.to_csv(f'result/{self.tag}/trial{trial.number}_cli_ranking.csv')
        fig = plt.figure()
        plt.plot(auc_per_reduction)
        plt.savefig(f'result/{self.tag}/trial{trial.number}_auc_per_reduction.png')
        plt.close()
        logging.info(f"Final AUC per reduction: {auc_per_reduction}")
        return max(auc_per_reduction)



class objective_meta:
    def __init__(self, features_filted, features_meta, labels, seed_nums,
                 device, reduction_times, epochs, filted_cols, meta_cols,model_type,tag):
        self.features_filted = features_filted
        self.features_meta = features_meta
        self.labels = labels
        self.tag = tag
        os.makedirs(f'result/{self.tag}', exist_ok=True)
        self.hidden_layers_filted =  [
            # [50, 100, 200, 100, 50, 20, 10, 10],
            # [50, 100, 200, 100, 50, 20, 5, ],
            # [40, 80, 160, 80, 40, 20, 10,],
            #[40, 80, 160, 80, 40, 10, 5],
            #[30, 60, 140, 70, 25, 10],
            [30, 60, 140, 70, 10, 5],
            [20, 20, 15, 5],
            [10, 10, 5],
            [10, 5],
            [0,0,0],
        ]
        self.hidden_layers_meta = [
            # [200, 150, 100, 75, 50, 25, 10, 10, 10],
            # [200, 100, 50, 20, 10, 5],
            # [200, 100, 10, 10],
            [100, 70, 35, 20, 10],
            [50, 50, 50, 10, 10, 10, 10, 10],
            [50, 30, 10,10],
            [30, 10, 10],
            [10, 10, 5],
            [0,0,0]
        ]
        outputsize = labels.shape[-1]
        self.hidden_layers_agg = [
            [20, 20, 10, 10,outputsize],
            [20, 10, 10, outputsize],
            [20, 10, outputsize],
            [10, outputsize],
            [5, outputsize],
        ]
        self.loss_fn = torch.nn.CrossEntropyLoss()
       
        self.reduction_ratio_meta = 0.8
        self.seed_nums = seed_nums
        self.device = device
        self.reduction_times = reduction_times
        self.epochs = epochs
        self.filted_cols = filted_cols
        self.meta_cols = meta_cols
        self.model_type = model_type

        logging.info(
            f"model_type:{self.model_type} device:{self.device}, reduction_times:{self.reduction_times}, epochs:{self.epochs}"
                    )    
    def objective(self, trial):
        auc_per_reduction = []
        
        filted_layer_scale=trial.suggest_int('filted layer scale',0,len(self.hidden_layers_filted)-1)
        meta_layer_scale=trial.suggest_int('meta layer scale',0,len(self.hidden_layers_meta)-1)
        agg_layer_scale=trial.suggest_int('aggregation layer scale',0,len(self.hidden_layers_agg)-1)
        
        
        selec_layers_filted = self.hidden_layers_filted[filted_layer_scale]
        selec_layers_meta = self.hidden_layers_meta[meta_layer_scale]
        selec_layers_agg = self.hidden_layers_agg[agg_layer_scale]
        
        
        # selec_layers_filted =  trial.suggest_categorical(
        #     'selec_layers_filted', self.hidden_layers_filted)
        # selec_layers_meta = trial.suggest_categorical(
        #     'selec_layers_meta', self.hidden_layers_meta)
        # selec_layers_agg = trial.suggest_categorical(
        #     'selec_layers_agg', self.hidden_layers_agg)
        dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.3)
        # lr = trial.suggest_float('lr', 0.002, 0.02, log=True)
        lr = 0.015
        k_fold_num = 5
        
        if selec_layers_filted[-1]==0 and selec_layers_meta[-1]==0:
            return np.float32(0.5)
        
        features_filted = self.features_filted
        features_meta_reduc = self.features_meta
        filted_cols = self.filted_cols.tolist()
        temp_meta_cols = self.meta_cols.tolist()

        # df to store the importance of each feature in different reduction
        df_meta_reduction = pd.DataFrame(columns=self.meta_cols)

        for reduciton_iter in range(self.reduction_times):
            logging.info(
                f"Features of Trial {trial.number} Reductuon {reduciton_iter} process: \
                meta: {features_meta_reduc.shape[-1]}"
            )

            layers_filted = [features_filted.shape[-1]] +  selec_layers_filted
            layers_meta = [features_meta_reduc.shape[-1]] + selec_layers_meta
            layers_agg = [layers_filted[-1] + layers_meta[-1]] + selec_layers_agg

            val_auc_list = []
            
            meta_rank_list = []

            # multi-thread computing
            if self.model_type=='mlp' or self.model_type=='gated_mlp':
                for seed_num in self.seed_nums:
                    val_auc, _, meta_rank = K_fold_cross_validation_DL(
                        k_fold_num, self.loss_fn, layers_filted, layers_meta, layers_agg,
                        self.epochs, features_filted, features_meta_reduc, self.labels, seed_num, 
                        self.device, lr, dropout_rate,self.model_type
                    )
                    val_auc_list.append(val_auc)
                    meta_rank_list.append(meta_rank)
            elif self.model_type=='attn' or self.model_type=='gated_attn':
                
                if self.model_type=='attn':
                    nhead = trial.suggest_int('nhead',1,2)
                    expands = trial.suggest_int('expands',1,2)
                elif self.model_type=='gated_attn':
                    nhead = trial.suggest_int('nhead',1,3)
                    expands = trial.suggest_int('expands',1,3)
                    
                for seed_num in self.seed_nums:
                    val_auc, _, meta_rank,attn_weight = K_fold_cross_validation_DL(
                        k_fold_num, self.loss_fn, layers_filted, layers_meta, layers_agg,
                        self.epochs, features_filted, features_meta_reduc, self.labels, 
                        seed_num, self.device, lr, dropout_rate,self.model_type,
                        expands,nhead
                    )
                    if attn_weight is not None:
                        attn_weight = attn_weight.mean(axis=0)
                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        im = ax.imshow(attn_weight, cmap=plt.cm.hot_r)
                        cbar = fig.colorbar(im)
                        cbar.set_label('Attention Weight')
                        plt.savefig(f'./result/{self.tag}/trial{trial.number}_redc_{reduciton_iter}_{seed_num}_attn_weight.png')
                        plt.close()
                        assert  isinstance(temp_meta_cols, list)
                        temp_cols = filted_cols+temp_meta_cols
                        Attention_df = pd.DataFrame(attn_weight, columns=temp_cols, index=temp_cols)
                        Attention_df.to_excel(f'./result/{self.tag}/trial{trial.number}_redc_{reduciton_iter}_{seed_num}_Attention.xlsx')
                        
                    val_auc_list.append(val_auc)
                    meta_rank_list.append(meta_rank)
            else:
                raise TypeError

            auc_single_reduction = np.mean(val_auc_list)
            logging.info(f"auc_single_reduction:  {auc_single_reduction}")
            auc_per_reduction.append(auc_single_reduction)
            
            combined_rank_meta, combined_scores_meta = combined_ranking(
                meta_rank_list, temp_meta_cols)
            df_meta_reduction = add_to_dataframe(
                df_meta_reduction, temp_meta_cols, combined_scores_meta)

            meta_selec_fea = get_part_ranking(
                combined_rank_meta, ratio=self.reduction_ratio_meta)
            temp_meta_cols = [temp_meta_cols[index] for index in meta_selec_fea]
            features_meta_reduc = features_meta_reduc[:,meta_selec_fea]
           
            if reduciton_iter==5 and auc_per_reduction[-1]<0.6 and auc_per_reduction[-2]<0.6:
                logging.info(f"Triggering the early stopping strategy")
                break

        df_meta_reduction.to_csv(f'result/{self.tag}/trial{trial.number}_meta_ranking.csv')
        fig = plt.figure()
        plt.plot(auc_per_reduction)
        plt.savefig(f'result/{self.tag}/trial{trial.number}_auc_per_reduction.png')
        plt.close()
        logging.info(f"Final AUC per reduction: {auc_per_reduction}")
        return max(auc_per_reduction)

def add_to_dataframe(df, col_names, combined_scores):
    new_row_values = {col_name: score for col_name,
                      score in zip(col_names, combined_scores)}
    df.loc[len(df)] = new_row_values
    return df


def K_fold_cross_validation_DL(k_fold_num, loss_fn, hidden_list_0,
                               hidden_list_1, hidden_list_agg, epochs,
                               features_0, features_1, labels,
                               seed_num, device, lr, dropout_rate,
                               model_type,
                                expands=None,nhead=None):
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)

    kfold = StratifiedKFold(n_splits=k_fold_num,
                            shuffle=True, random_state=seed_num)

    Stratifiedlabels = torch.argmax(labels, dim=-1)

    loss_curve = [[] for _ in range(k_fold_num)]
    train_auc_curve = [[] for _ in range(k_fold_num)]
    val_auc_curve = [[] for _ in range(k_fold_num)]

    best_model = None
    best_val_auc = 0
    best_trn_dataset = None
    for fold, (train_idx, val_idx) in enumerate(kfold.split(features_0.cpu(), Stratifiedlabels.cpu())):

        
        train_dataset = get_dataset_from_index(
            train_idx, features_0, features_1, labels)
        val_dataset = get_dataset_from_index(
            val_idx, features_0, features_1, labels)
        train_dataset = balanced_data(train_dataset, device)
        val_dataset = balanced_data(val_dataset, device)
        

        if model_type =='mlp':
            model = Mixed_model(hidden_list_0, hidden_list_1, hidden_list_agg,
                            dropout_rate=dropout_rate, act='leakyrelu').to(device)
        elif model_type =='gated_mlp':
            model = Gated_model(hidden_list_0, hidden_list_1, hidden_list_agg,
                            dropout_rate=dropout_rate, act='leakyrelu').to(device)
        elif model_type=='attn':
            model = Attention_model(hidden_list_0, hidden_list_1, hidden_list_agg,
                            dropout_rate=dropout_rate, act='leakyrelu',expands=expands,
                            N_sample_train=len(train_dataset['labels']),nhead=nhead).to(device)
        elif model_type=='gated_attn':
            model = Attention_model_v2(hidden_list_0, hidden_list_1, hidden_list_agg,
                            dropout_rate=dropout_rate, act='leakyrelu',expands=expands,
                            N_sample_train=len(train_dataset['labels']),nhead=nhead).to(device)
        else:
            raise ValueError('model_type must be one of [mlp, attn,gated_attn], now:',model_type)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        import time
        
        
        for epoch_index in tqdm(range(epochs)):
            model.train()
            
            output = model(train_dataset['features_0'],
                           train_dataset['features_1'])
            loss = loss_fn(output, train_dataset['labels'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

           
            # 计算AUC
            with torch.no_grad():
                train_auc = roc_auc_score(
                    train_dataset['labels'].cpu(), output.cpu().detach(), average='micro')

                loss_curve[fold].append(loss.item())
                train_auc_curve[fold].append(train_auc)
               
                val_auc = validate(model,   val_dataset)
                
                if val_auc > best_val_auc:
                    model.eval()
                    best_model = copy.deepcopy(model)
                    best_trn_dataset = copy.deepcopy(train_dataset)
                    best_val_auc = val_auc
                val_auc_curve[fold].append(val_auc)

                
    model.eval()
    mean_loss = np.array(loss_curve).min(axis=1).mean()
    train_perform = np.array(train_auc_curve).mean(axis=0)
    mean_train_auc = train_perform.max()
    val_perform = np.array(val_auc_curve).mean(axis=0)
    mean_val_auc = val_perform.max()
    arg_max_val = val_perform.argmax()

    logging.info(
        f"loss, train_auc, val_auc, epoch index of best val_auc: {mean_loss}, {mean_train_auc}, {mean_val_auc}, {arg_max_val}")
    # visualization_results(k_fold_num, loss_curve,
    # train_auc_curve, val_auc_curve)

    
    best_model.eval()
    dataset_shap = [best_trn_dataset['features_0'],
                        best_trn_dataset['features_1']]
    explainer = shap.DeepExplainer(best_model, dataset_shap)

    shap_values = explainer.shap_values(
        dataset_shap, check_additivity=False)
    # print(len(shap_values),shap_values[0][0].shape,shap_values[0][1].shape,shap_values[1][0].shape,shap_values[1][1].shape,len(shap_values[1]))
    # input()
    for input_index in range(len(shap_values)):
        for output_index in range(len(shap_values[0])):
            shap_values[input_index][output_index] = np.abs(shap_values[input_index][output_index])
    feature_importance_0 = (shap_values[0][0]+shap_values[1][0]+shap_values[2][0])/3
    feature_importance_0 = feature_importance_0.mean(axis=0)
    feature_importance_1 = (shap_values[0][1]+shap_values[1][1]+shap_values[2][1])/3
    feature_importance_1 = feature_importance_1.mean(axis=0)
    
    

    # 对特征重要性得分进行排序
    sorted_indices_0 = np.argsort(feature_importance_0)[::-1]  # 降序排序的特征索引
    sorted_indices_1 = np.argsort(feature_importance_1)[::-1]
    
    if model_type == 'attn' or model_type =='gated_attn':
        if hidden_list_0[-1]==0 or hidden_list_1[-1]==0:
            return mean_val_auc, sorted_indices_0, sorted_indices_1,None 
            #best_model has no attn_weight when single matrix inputted
        else:
            return mean_val_auc, sorted_indices_0, sorted_indices_1,best_model.attn_weight.detach().cpu().numpy()
    elif model_type == 'mlp' or model_type =='gated_mlp':
        return mean_val_auc, sorted_indices_0, sorted_indices_1
    else:
        raise TypeError


def visualization_results(k_fold_num, loss_curve, train_auc_curve, val_auc_curve):
    fig, axes = plt.subplots(k_fold_num, 2, figsize=(10, 10))

    # 绘制每个子图
    for index in range(k_fold_num):
        axes[index, 0].plot(loss_curve[index], label=f'Fold {index+1}')
        axes[index, 1].plot(train_auc_curve[index],
                            label=f'{max(train_auc_curve[index])}')
        axes[index, 1].plot(val_auc_curve[index],
                            label=f'{max(val_auc_curve[index])}')
        axes[index, 0].legend(loc='upper right')
        axes[index, 1].legend(loc='upper right')
    plt.legend()
    plt.tight_layout()

    plt.show()


def get_dataset_from_index(index, features_0, features_1, labels):
    get_features_0 = features_0[index]
    get_features_1 = features_1[index]
    get_labels = labels[index]

    # 创建 TensorDataset 对象
    _dataset = {
        'features_0': get_features_0,
        'features_1': get_features_1,
        'labels': get_labels
    }

    return _dataset


def balanced_data(dataset, device):

    labels = dataset['labels']
    features_0 = dataset['features_0']
    features_1 = dataset['features_1']

    flattened_labels = torch.argmax(labels, dim=1)
    ros = RandomOverSampler()
    resampled_features_0, resampled_labels = ros.fit_resample(
        features_0, flattened_labels)
    resampled_features_1, _ = ros.fit_resample(
        features_1, flattened_labels)
    resampled_labels_onehot = torch.zeros(
        (resampled_labels.size, labels.shape[1]))
    resampled_labels_onehot[torch.arange(
        resampled_labels.size), resampled_labels] = 1

    dataset['labels'] = resampled_labels_onehot.to(device)
    dataset['features_0'] = torch.tensor(
        resampled_features_0, device=device)
    dataset['features_1'] = torch.tensor(
        resampled_features_1, device=device)

    return dataset
