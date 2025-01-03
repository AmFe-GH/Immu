import matplotlib.pyplot as plt
from skbio.diversity import alpha_diversity
from models import *
from utils import *
import trainer
from trainer import *
import torch
import pandas as pd
import torch.nn.functional as F
from combat.pycombat import pycombat
import seaborn as sns
import scipy.stats as stats
from collections import Counter

def sort_by_keys(item):
    key, value = item
    # 如果key包含'CR'，给它一个小的排序值；否则给一个大的排序值
    if 'CR' in key:
        return -1
    elif 'PD' in key:
        return 1
    elif 'SD' in key:
        return 0
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
            Shannon_list = alpha_diversity('shannon', class_features.numpy())
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
def delete_samples(dict_data):
    new_dict = copy.copy(dict_data)
    first_key = next(iter(new_dict.keys()))
    
    
    first_value = list(new_dict[first_key])
    sorted_values = sorted(first_value)[:]
    
    
    new_dict[first_key]=sorted_values
    return new_dict

Prevalence_rate_list=[0.0,0.05,0.09,0.10,0.12,0.13,0.14,0.15,0.20,0.25,0.30]
abun_list=[0.0001,0.001,0.0005,0.00001,0.00005,0]
Pre_single_list = [0.4,0.6,0.8,1.0]
levellist = ['S']
labels_sets={'Response':["CR-PR",'PD','SD'],'ORR':["CR-PR",'PD-SD','PD-SD'],'DCR':["CR-PR-SD",'PD',"CR-PR-SD"]}
#labels_sets={'ORR':["CR-PR",'PD-SD','PD-SD']}

alpha_div_df = pd.DataFrame(columns=['labels_sets','Pre','Abun','Pre_single','Level','P-value','Div sort','Percentage'])
for level in levellist:
        
    merged_df = pd.read_excel(f'../Data/cli_mNGS.Microbe.{level}.xlsx')             
    features_df_mNGS_all = merged_df.iloc[:,
                                    merged_df.columns.get_loc('Responce2_SD')+1:]
    labels_df = merged_df.loc[:,'Responce2_CR-PR':'Responce2_SD']
    labels_arg = labels_df.values.argmax(axis=1)
    batch_data = merged_df.loc[:, 'Batch']
    for  Pre_single in Pre_single_list:
        for threshold_low_abun in abun_list:
            for Prevalence_rate in Prevalence_rate_list:
                for group_key,label_tag in labels_sets.items():
                    
                        
                        label_str = np.array([label_tag[index] for index in labels_arg])
                        dense_col =  dense_columns(features_df_mNGS_all,group_tag=label_str,\
                            Prevalence_rate=Prevalence_rate,threshold_low_abun=threshold_low_abun,\
                                    Pre_single=Pre_single)
                       
                        Percentage = len(dense_col)/features_df_mNGS_all.shape[1]
                        features_df_mNGS = features_df_mNGS_all.iloc[:,dense_col]   
                        #[markers[:len(markers)//1]]
                        
                        mNGS_cols = features_df_mNGS.columns
                        
                        

                        #assert no_all_zero_col(features_df_mNGS)


                        

                        
                        # features_mNGS = torch.tensor(
                        #     features_df_mNGS.values.astype('float32'))
                        #features_df_mNGS_CLR = CLR_normalize(features_df_mNGS)
                        check_tensor(features_df_mNGS.values)
                        features_df_mNGS_TIC = TIC_norm(features_df_mNGS)
                        print('tag1',features_df_mNGS_TIC.sum(axis=1).min())
                        check_tensor(features_df_mNGS_TIC.values)
                        # features_df_mNGS_CBT = pycombat(
                        #                     data=features_df_mNGS_TIC.T, \
                        #                     batch=batch_data.values.astype(int),\
                        #                     par_prior=False
                        #                         ).T
                        # features_df_mNGS_CBT = min_max_normalize(features_df_mNGS_CBT,dim=1)
                        features_mNGS = torch.tensor(features_df_mNGS_TIC.values, dtype=torch.float32)
                        #features_mNGS = features_mNGS[:,~((features_mNGS<0.01).all(dim=0))]

                        #features_mNGS = min_max_normalize(features_mNGS,dim=1)
                        #features_mNGS = min_max_normalize(features_mNGS.requires_grad_(False),dim=1)
                        check_tensor(features_mNGS)

                        # features_mNGS = mean_std_normalize(features_mNGS, dim=0).requires_grad_(False)
                        
                        print(features_mNGS[0].min(), features_mNGS[0].max())

                        alpha_diversity_dict = calculate_alpha_diversity(features_mNGS, label_str)
                        alpha_diversity_dict = delete_samples(alpha_diversity_dict)
                        
                        groups = alpha_diversity_dict.values()
                        
                        
                        h_value, p_value = stats.kruskal(*groups)
                        
                        groups_list = list(groups)
                        abun_sort = np.median(groups_list[0]) > np.median(groups_list[-1])
                        # 颜色列表，每个类对应一种颜色
                        colors = ['#8ECFC9', '#FA7F6F', '#BEB8DC', '#E7DAD2']

                        # 创建图形
                        plt.figure(figsize=(8, 6))
                        ax = plt.gca()
                        # 绘制箱线图，并设置 patch_artist=True 以便设置颜色
                        box = plt.boxplot(alpha_diversity_dict.values(), labels=list(alpha_diversity_dict.keys()), showfliers=False, patch_artist=False)
                        textstr = f'P-value: {p_value:.2e}'
                        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                        ax.text(0., 1.05, textstr, transform=ax.transAxes, fontsize=12,
                            verticalalignment='top', bbox=props)
                        
                        # 设置箱线图的颜色
                        for patch, median, whisker, cap, flier, color in zip(box['boxes'], box['medians'], box['whiskers'], box['caps'], box['fliers'], colors):
                            patch.set_edgecolor(color)
                            patch.set_facecolor('none')  # 内部填充透明
                            median.set_color('black')
                            for whisker_part in whisker:
                                whisker_part.set_color(color)
                            for cap_part in cap:
                                cap_part.set_color(color)

                        # 设置标题和y轴标签
                        plt.title(f'Alpha diversity: {level}\n Groupby{group_key}_Pre{Prevalence_rate}_Abun_{threshold_low_abun}')
                        plt.ylabel('Shannon Diversity Index')

                        # 为每个类别绘制散点图
                        for index, (key, color) in enumerate(zip(alpha_diversity_dict.keys(), colors)):
                            
                            x = alpha_diversity_dict[key]
                            plt.plot([index+1+(np.random.rand()-0.5)*0.1 for _ in range(len(x))], x, 'o', color=color, alpha=0.6)

                        # 关闭网格
                        plt.grid(False)
                        os.makedirs('./Figure', exist_ok=True)
                        plt.savefig(f'./Figure/Alpha_div_Groupby{group_key}_Pre{Prevalence_rate}_Abun{threshold_low_abun}_Pre4sin{Pre_single}_{level}.pdf')
                        #plt.show()
                        plt.close()
                        
                        new_row_data = [group_key,Prevalence_rate,threshold_low_abun,Pre_single,level,p_value,abun_sort,Percentage]
                        alpha_div_df.loc[len(alpha_div_df)] = new_row_data
alpha_div_df.to_csv(f'../Data/Alpha_Beta_div/alpha_div_df.csv')