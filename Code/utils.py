import pandas as pd
import re
import numpy as np
import torch
from pycombat import pycombat
from collections import Counter
from skbio.diversity import alpha_diversity
import copy
import seaborn as sns
import torch.nn as nn
reversed_dict_map = {
    'D': 1,  # Domain
    'K': 2,  # Kingdom
    'P': 3,  # Phylum
    'C': 4,  # Class
    'O': 5,  # Order
    'F': 6,  # Family
    'G': 7,  # Genus
    'S': 8,  # Species
    'T': 9   # Strain
}
dict_map = {
    'Domain': 'D',
    'Kingdom': 'K',
    'Phylum': 'P',
    'Class': 'C',
    'Order': 'O',
    'Family': 'F',
    'Genus': 'G',
    'Species': 'S',
    'Strain': 'T'
}


def end_with(col_name):
    if isinstance(col_name, str):
        split_results = col_name.split('|')
        final_level = split_results[-1].split("__")
        assert len(final_level) == 2
        return final_level[0]  # D,K,P,C,O,F......
    elif isinstance(col_name, pd.Series):  # Series
        result_list = []
        for item in col_name:
            split_results = item.split('|')
            final_level = split_results[-1].split("__")
            assert len(final_level) == 2
            result_list.append(final_level[0])
        return result_list


def act_functions(self, act: str):
        if act == 'relu':
            return nn.ReLU()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'softmax':
            return nn.Softmax(dim=-1)
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'softplus':
            return nn.Softplus()
        elif act == 'gelu':
            return nn.GELU()
        elif act == 'leakyrelu':
            return nn.LeakyReLU()
        else:
            raise ValueError('Invalid activation function: {}'.format(act))

def check_mNGS_df_col_name(data_frame: pd.DataFrame):
    last_level = 0
    for col in data_frame.columns[1:]:
        if reversed_dict_map[end_with(col)]-last_level > 1 and last_level != 1:
            raise TypeError(last_col, col)
        last_level = reversed_dict_map[end_with(col)]
        last_col = col


def get_resolution(data_frame: pd.DataFrame, level: str):
    assert level in ['Domain', 'Kingdom', 'Phylum', 'Class',
                     'Order', 'Family', 'Genus', 'Species', 'Strain']

    filtered_columns = [
        col for col in data_frame.columns if end_with(col) == dict_map[level]]
    return data_frame[filtered_columns], filtered_columns


def split_train_val(features_cli, features_mNGS, labels, val_ratio=0.2, shuffle=True):
    """_summary_

    Args:
        features (torch.tensor): input features of all samples [N*featreure_dim]
        multi_labels (torch.tensor): all labels  [N*class_nums]
        val_ratio (float, optional):  Defaults to 0.2.
        shuffle (bool, optional): Defaults to True.

    """

    # 获取样本数量
    n_samples = features_cli.shape[0]

    # 创建索引数组
    indices = np.arange(n_samples)

    # 是否打乱数据
    if shuffle:

        np.random.shuffle(indices)

    # 计算验证集的样本数量
    val_size = int(n_samples * val_ratio)

    # 划分索引
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    # 根据索引划分数据
    train_features_cli = features_cli[train_indices]
    train_features_mNGS = features_mNGS[train_indices]
    train_labels = labels[train_indices]
    val_features_cli = features_cli[val_indices]
    val_features_mNGS = features_mNGS[val_indices]
    val_labels = labels[val_indices]

    # 创建 TensorDataset 对象
    train_dataset = {
        'features_cli': train_features_cli,
        'features_mNGS': train_features_mNGS,
        'labels': train_labels
    }
    val_dataset = {
        'features_cli': val_features_cli,
        'features_mNGS': val_features_mNGS,
        'labels': val_labels
    }
    return train_dataset, val_dataset


def check_tensor(input):
    if isinstance(input, torch.Tensor):
        if torch.isnan(input).any():
            raise ValueError('contains NaN values.')
        if torch.isinf(input).any():
            raise ValueError('contains Inf values.')
    elif isinstance(input, np.ndarray):
        if np.isnan(input).any():
            raise ValueError('contains NaN values.')
        if np.isinf(input).any():
            raise ValueError('contains Inf values.')
    else:
        raise TypeError('input must be a torch.tensor or numpy.ndarray.')


def combined_ranking(rankings, temp_cols):
    assert all([
        len(rankings[0])==len(rankings[index]) for index in range(len(rankings))
        ])
    assert len(temp_cols) ==len(rankings[0])
    n_features = len(rankings[0])
    scores = np.zeros(n_features)

    # 分配分数：最高排名的特征得分数最多，最低排名的特征得分数最少
    for rank in rankings:
        for i, feature in enumerate(rank):
            scores[feature] += (n_features - i)

    # 根据总分数进行综合排序

    combined_rank = np.argsort(-scores)
    return combined_rank, scores


def get_part_ranking(rank, ratio):
    length = len(rank)
    if length <= 5:
        return rank
    else:
        return rank[:int(length*ratio)]


def compare_list(lst, value):
    return [elem == value for elem in lst]


def get_stepped_mNGS(df):
    temp = end_with(df[df.columns[0]])
    select_row = compare_list(temp, 'T')
    select_df = df[select_row]


def CLR_normalize(features_mNGS,eps):
    
    #eps = 1e-15
    dim = 1
    if isinstance(features_mNGS,torch.Tensor):
        if (features_mNGS == 0).any():
            features_mNGS = features_mNGS + eps
        mean_features = torch.exp(torch.mean(
            torch.log(features_mNGS), dim=dim, keepdim=True))

        assert mean_features.shape[0] == features_mNGS.shape[0], [
            mean_features.shape[0], features_mNGS.shape[0]]
        clr_data = torch.log(features_mNGS/mean_features)
        return clr_data
    elif isinstance(features_mNGS,np.ndarray):
        if (features_mNGS == 0).any():
            features_mNGS = features_mNGS + eps
        mean_features = np.exp(np.mean(
            np.log(features_mNGS), axis=dim, keepdims=True))

        assert mean_features.shape[0] == features_mNGS.shape[0], [
            mean_features.shape[0], features_mNGS.shape[0]]
        clr_data = np.log(features_mNGS/mean_features)
        return clr_data
    elif isinstance(features_mNGS,pd.DataFrame):
        columns=features_mNGS.columns
        features_mNGS = features_mNGS.values
        if (features_mNGS == 0).any():
            features_mNGS = features_mNGS + eps
        mean_features = np.exp(np.mean(
            np.log(features_mNGS), axis=dim, keepdims=True))

        assert mean_features.shape[0] == features_mNGS.shape[0], [
            mean_features.shape[0], features_mNGS.shape[0]]
        clr_data = np.log(features_mNGS/mean_features)
        return pd.DataFrame(clr_data, columns=columns)
    else:
        raise TypeError
def min_max_normalize(input, dim):
    eps = 1e-20

    if isinstance(input, torch.Tensor):
        min_val, _ = input.min(dim=dim,keepdim=True)
        print(min_val.shape)
        max_val, _ = input.max(dim=dim,keepdim=True)
        normalized_tensor = (input - min_val) / (max_val - min_val + eps)

        
        return normalized_tensor
    elif isinstance(input, np.ndarray):
        min_val = input.min(axis=dim,keepdims=True)

        max_val = input.max(axis=dim,keepdims=True)
        normalized_tensor = (input - min_val) / (max_val - min_val + eps)

        return normalized_tensor
    


def mean_std_normalize(input, dim):
    eps = 1e-8
    mean = input.mean(dim=0)
    print(mean.shape)
    std = input.std(dim=0)
    normalized_tensor = (input - mean) / (std+eps)
    normalized_tensor[normalized_tensor == 0.] = eps
    return normalized_tensor

def get_markers_from_paths(dic_set,level):
    paths = dic_set[level]['ranking_data_path']
    assert  isinstance(paths,list)
    filted_markers = []
    for path in paths:
        ranking_markers = pd.read_csv(path)
        filted_markers+=( ranking_markers.drop(columns=['Unnamed: 0']).\
            iloc[dic_set[level]['reduction_index']].dropna().index.tolist())
    return filted_markers

def no_all_zero_col(features):
    if isinstance(features,pd.DataFrame):
        npvalue = features.values.astype(float)
        zero_col = (npvalue==0.).all(axis=0)
        return zero_col.sum() == 0
    elif isinstance(features,torch.Tensor):
        zero_col = (features==0.).all(axis=0)
        return zero_col.sum() == 0

def df2tensor(*dfs):
    tensor_values = [torch.tensor(df.values.astype('float32')) for df in dfs]
    return tensor_values

def TIC_norm(features):
    if isinstance(features,torch.Tensor):
        sum_feas = features.sum(dim=1,keepdim=True)
        assert not (sum_feas==0).any()
        assert (sum_feas>=0).all()
        return features/sum_feas
    elif isinstance(features,np.ndarray):
        sum_feas = features.sum(axis=1, keepdims=True)
        assert not (sum_feas == 0).any(), "Sum of features along axis 1 contains zero values."
        assert (sum_feas >= 0).all(), "Sum of features along axis 1 contains negative values."
        return features / sum_feas
    elif isinstance(features,pd.DataFrame):
        sum_feas = features.sum(axis=1)
        assert not (sum_feas==0).any()
        assert (sum_feas>=0).all()
        return features.div(sum_feas, axis=0)
    else:
        raise TypeError
    
def sort_by_keys(item):
    key, value = item
    # 如果key包含'CR'，给它一个小的排序值；否则给一个大的排序值
    if 'CR' in key:
        return -1
    elif 'PD' in key:
        return 1
    elif 'SD' in key:
        return 0
    else:
        raise TypeError
def calculate_alpha_diversity(features, labels_str):
    # 计算每个类别的样本数
    labels_str = np.array(labels_str)
    unique_labels = set(labels_str)

    # 如果你需要结果是一个列表，可以再将set转换为list
    unique_labels_list = list(unique_labels)
    

    # 初始化 alpha 多样性结果的张量，形状为 (class_num, samples)
    alpha_diversity_dict = {}

    # 对于每个类别，计算 alpha 多样性
    for class_name in unique_labels_list:
        # 获取属于当前类别的样本特征的索引
        class_indices = np.where(labels_str==class_name)
        
        # 获取属于当前类别的样本特征
        class_features = features[class_indices]
        
        # 计算当前类别的 alpha 多样性
        if len(class_features) > 1:
            Shannon_list = alpha_diversity('shannon', class_features.numpy(),validate=True)
            # if 'PD' in class_name:
            #     max_index = Shannon_list.idxmax()
            #     Shannon_list= Shannon_list.drop(max_index)
        alpha_diversity_dict[class_name]=Shannon_list
    sorted_items = sorted(alpha_diversity_dict.items(), key=sort_by_keys)
    sorted_dict = dict(sorted_items)
    return sorted_dict




def dense_columns(features_df_mNGS,group_tag,Prevalence_rate,threshold_low_abun,Pre_single):
    features_mNGS = np.copy(features_df_mNGS.values)
    
    all_group_sum = Counter(group_tag)
    all_group_sum = dict(all_group_sum)
    if isinstance(group_tag,pd.DataFrame):
        group_tag = group_tag.values
    
    # Low abundance values will be considered as non-existent
    print('low_abun ratio: ',((features_mNGS<threshold_low_abun)).sum()/features_mNGS.size)
    features_mNGS[features_mNGS<threshold_low_abun] = 0
    
    
    # Screening by prevalence
    zero_or_not_features = (features_mNGS !=0.)
    
    selec_col_index_list=[]
    for col_index in range(zero_or_not_features.shape[1]):
        fea_on_samples = zero_or_not_features[:,col_index].squeeze()
        group_on_samples = group_tag[fea_on_samples]
        
        group_sum = Counter(group_on_samples)
        group_sum = dict(group_sum)
        
        select_tag=0
        for key in group_sum.keys():
            
            if (group_sum[key]/all_group_sum[key]) >= Prevalence_rate:
                select_tag=1
                break
            
        if len(group_sum.keys())==1 and select_tag==0:
            key_single = next(iter(group_sum.keys()))
            if group_sum[key_single]/all_group_sum[key_single]>(Pre_single*Prevalence_rate):
                select_tag=1
            
            
        if select_tag==1:
            selec_col_index_list.append(col_index)
        elif select_tag==0:
            # plt.plot(features_mNGS[:,col_index])
            # unique_str = np.unique(group_tag)
            # colors = ['red', 'green', 'blue']
            # color_map = dict(zip(unique_str, colors))
            # color_list = [color_map[group_tag[index]] for index in range(len(group_tag))]
            # plt.scatter(x=range(121),y=[ 0 for _ in range(len(group_tag))], color=color_list,s=3)
            # plt.show()
            # plt.close()
            pass
        else:
            raise TypeError
    selec_col_index_list = np.array(selec_col_index_list)
    return selec_col_index_list




# Pareto scaling
def pareto_scaling(df):
    if isinstance(df,torch.Tensor):
        std_devs = df.std(dim=0)

        # 处理标准偏差为0的情况
        std_devs[std_devs == 0] = 1.0  # 或者使用非常小的正数替代0，如 1e-8

        # Pareto scaling
        scaled_features_tensor = df / torch.sqrt(std_devs)
        return scaled_features_tensor
    elif isinstance(df,pd.DataFrame):
        std_devs = df.std()
        # 对于标准偏差为0的列，我们需要处理，避免除以0
        std_devs = std_devs.replace(0, 1)  # 或者使用非常小的正数替代0，如 1e-8
        return df.divide(np.sqrt(std_devs), axis='columns')
    else:
        raise TypeError
    
def clst(lst):
    tlst = []
    for i in lst:
        if i == 'Up':
            tlst.append('r')
        elif i == 'Down':
            tlst.append('g')
        else:
            tlst.append('k')
    return tlst
def plot_pca(data, batch_labels, title, path=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    import pandas as pd
    import os
    
    if path is not None:
        os.makedirs(path, exist_ok=True)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    
    # Create a DataFrame for plotting
    df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    df_pca['Batch'] = batch_labels
    
    # Explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    
    # Define custom colors for each Batch value
    palette = {
        2: '#00BFC4',
        1: '#F8766D'
    }
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")
    scatter = sns.scatterplot(x='PC1', y='PC2', hue='Batch', data=df_pca, palette=palette, s=100, alpha=0.7)
    
    # Add labels for explained variance
    plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}% explained variance)')
    plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}% explained variance)')
    
    # Title and grid
    plt.title(title, fontsize=16)
    plt.legend(title='Batch')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save or show plot
    plt.tight_layout()
    if path is not None:
        plt.savefig(os.path.join(path, title.replace(' ', '_') + '.pdf'))
    plt.show()
    
def batch_standardize(features, batch_labels):
    """
    Standardizes features within each batch.

    Args:
    features (torch.Tensor): The feature matrix to be standardized.
    batch_labels (np.ndarray): The array of batch labels corresponding to the features.

    Returns:
    torch.Tensor: Standardized feature matrix.
    """
    

    if isinstance(features,torch.Tensor):
        unique_batches = np.unique(batch_labels)
        standardized_features = torch.zeros_like(features)
    
        assert (features==0).all(dim=0).sum()==0
        for batch in unique_batches:
            
            batch_mask = batch_labels == batch
            batch_features = features[batch_mask]

            batch_mean = torch.mean(batch_features, dim=0)
            batch_std = torch.std(batch_features, dim=0)
            standardized_features[batch_mask] = (batch_features - batch_mean) / (batch_std + 1e-15)
        
        
        

    elif isinstance(features,pd.DataFrame):
        unique_batches = np.unique(batch_labels)
        standardized_features = pd.DataFrame(index=features.index, columns=features.columns)
        assert (features == 0).all().sum() == 0
        for batch in unique_batches:
            batch_mask = batch_labels == batch
            batch_features = features[batch_mask]

            batch_mean = batch_features.mean(axis=0)
            batch_std = batch_features.std(axis=0)
            standardized_features.loc[batch_mask] = (batch_features - batch_mean) / (batch_std + 1e-15)
    return standardized_features