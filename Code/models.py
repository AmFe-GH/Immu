import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from math import ceil

class Attention_model_v2(nn.Module):
    def __init__(self, hidden_list_1, hidden_list_2, hidden_list_agg, dropout_rate, act,expands,N_sample_train,nhead):
        super(Attention_model_v2, self).__init__()
        F1 = hidden_list_1[0]
        F2 = hidden_list_2[0]
        self.feature_1 = hidden_list_1[0]
        self.feature_2 = hidden_list_2[0]
        self.feature_all = self.feature_1+self.feature_2
        self.embed_dim = expands*nhead #embed dim must been divisible by nhead
        self.embedding_pro = nn.Linear(1,self.embed_dim)
        self.K_pro = nn.Linear(self.embed_dim, self.embed_dim)
        self.Q_pro = nn.Linear(self.embed_dim, self.embed_dim)
        self.V_pro = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=nhead,batch_first=True,dropout=dropout_rate)
        
        self.anti_embedding_pro_1 = nn.Sequential(
            nn.Linear(self.embed_dim, ceil(self.embed_dim/2)),nn.Linear(ceil(self.embed_dim/2), 1),Squeeze())
        self.anti_embedding_pro_2 = nn.Sequential(
            nn.Linear(self.embed_dim, ceil(self.embed_dim/2)),nn.Linear(ceil(self.embed_dim/2), 1),Squeeze())
        # register 'mask' to make self.mask can be move to device when using "model.to(device)"
        self.register_buffer('mask', torch.ones((self.feature_all, self.feature_all), requires_grad=False, dtype=torch.bool))
        self.mask[:F1, F1:F1+F2] = False
        self.mask[F1:F1+F2, :F1] = False

        if hidden_list_1[-1] == 0:
            self.use_1 = False
        else:
            self.use_1 = True
            layers_1 = []
            for i in range(len(hidden_list_1) - 1):
                layers_1.append(
                    nn.Linear(hidden_list_1[i], hidden_list_1[i+1]))
                if i < len(hidden_list_1) - 2:
                    layers_1.append(self.act_functions(act))
                    layers_1.append(nn.Dropout(p=dropout_rate))
            self.model_1 = nn.Sequential(*layers_1)
            self.dropout1 = nn.Dropout(dropout_rate)
            
        if hidden_list_2[-1] == 0:
            self.use_2 = False
        else:
            self.use_2 = True
            layers_2 = []
            for i in range(len(hidden_list_2) - 1):
                layers_2.append(
                    nn.Linear(hidden_list_2[i], hidden_list_2[i+1]))
                if i < len(hidden_list_2) - 2:
                    layers_2.append(self.act_functions(act))
                    layers_2.append(nn.Dropout(p=dropout_rate))
            self.model_2 = nn.Sequential(*layers_2)
            self.dropout2 = nn.Dropout(dropout_rate)

        self.gate= GatedBlock(hidden_list_1[-1],hidden_list_2[-1] )
        
        layers_agg = []
        for i in range(len(hidden_list_agg) - 1):
            layers_agg.append(
                nn.Linear(hidden_list_agg[i], hidden_list_agg[i+1]))
            if i < len(hidden_list_agg) - 2:
                layers_agg.append(self.act_functions(act))
                layers_agg.append(nn.Dropout(p=dropout_rate))
        self.model_agg = nn.Sequential(*layers_agg)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.nhead = nhead
    
    def forward(self, input_1, input_2):

        if self.use_2 and self.use_1:
            
            X =torch.cat((input_1,input_2),dim=-1)
            
            X = X.unsqueeze(2) #Batch * feature_lenght * 1
            X_emb = self.embedding_pro(X)
            # keys = self.K_pro(X_emb)# Batch * feature_lenght * d_model
            # querys = self.Q_pro(X_emb)# Batch * feature_lenght * d_model
            # values = self.V_pro(X_emb)# Batch * feature_lenght * d_model
            
            attn_output, self.attn_weight = self.attention(X_emb,X_emb,X_emb,attn_mask=self.mask,average_attn_weights=True)
            attn_output = attn_output+X_emb
            
            # proc_1 = X_emb[:,:self.feature_1,:] #same shape as input 1
            # proc_2 = X_emb[:,self.feature_1:,:]
            proc_1 = attn_output[:,:self.feature_1,:] #same shape as input 1
            proc_2 = attn_output[:,self.feature_1:,:]
            
            proc_1 = self.anti_embedding_pro_1(proc_1)
            proc_2 = self.anti_embedding_pro_2(proc_2)
            
            
            # X = X.squeeze()
            # proc_1 = X[:,:self.feature_1]
            # proc_2 = X[:,self.feature_1:]
            output_1 = self.model_1(proc_1)
            output_2 = self.model_2(proc_2)
            
            output_1 = self.dropout1(output_1)
            output_2 = self.dropout2(output_2)
            input_of_agg = self.gate([output_1,output_2])
            #input_of_agg = torch.cat([output_1,output_2],dim=-1)
            
        elif self.use_1:
            input_of_agg = self.model_1(input_1)
        elif self.use_2:
            input_of_agg = self.model_2(input_2)
        else:
            raise ValueError
        x = self.model_agg(input_of_agg)
        x = self.dropout(x)
        return x

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
        
class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return torch.squeeze(x)
    
class GatedBlock(nn.Module):
    def __init__(self, size1,size2):
        super(GatedBlock, self).__init__()
        self.gate_blocks = nn.ModuleList([
            Gated_mlp(d_model=size1),
            Gated_mlp(d_model=size2)
        ])
    def forward(self, x_list:list):
        # get gate tensor [0~1]
        temp = [self.gate_blocks[index](x_list[index]) for index in range(len(x_list))]
        result = torch.cat(temp, dim=-1)
        return result


class Gated_mlp(nn.Module):
    def __init__(self, d_model):
        super(Gated_mlp, self).__init__()
        # self.norm = nn.LayerNorm(d_model)
        # self.proj1 = nn.Linear(d_model, d_model)
        # self.gelu = nn.LeakyReLU()
        self.spatial_gating_unit = SpatialGatingUnit(d_model)
        self.proj2 = nn.Linear(d_model, d_model)

    def forward(self, x):
        shortcut = x
        # x = self.norm(x)
        # x = self.proj1(x)
        # x = self.gelu(x)
        x = self.spatial_gating_unit(x)
        x = self.proj2(x)
        return x + shortcut
    

class SpatialGatingUnit(nn.Module):
    def __init__(self, d_model):
        super(SpatialGatingUnit, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model,bias=False)  # The projection for v
        self.init_bias = torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        u, v = [x,x]
        v = self.norm(v)
        v = self.proj(v) + self.init_bias
        return u * v  # Element-wise multiplication
class Attention_model(nn.Module):
    def __init__(self, hidden_list_1, hidden_list_2, hidden_list_agg, dropout_rate, act,expands,N_sample_train,nhead):
        super(Attention_model, self).__init__()
        F1 = hidden_list_1[0]
        F2 = hidden_list_2[0]
        self.feature_1 = hidden_list_1[0]
        self.feature_2 = hidden_list_2[0]
        self.feature_all = self.feature_1+self.feature_2
        self.embed_dim = expands*N_sample_train*nhead #embed dim must been divisible by nhead
        self.K_pro = nn.Linear(N_sample_train, self.embed_dim)
        self.Q_pro = nn.Linear(N_sample_train, self.embed_dim)
        self.V_pro = nn.Linear(N_sample_train, self.embed_dim)
        
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=nhead,batch_first=False,dropout=dropout_rate)
        # register 'mask' to make self.mask can be move to device when using "model.to(device)"
        self.register_buffer('mask', torch.ones((self.feature_all, self.feature_all), requires_grad=False, dtype=torch.bool))
        self.mask[:F1, F1:F1+F2] = False
        self.mask[F1:F1+F2, :F1] = False

        if hidden_list_1[-1] == 0:
            self.use_1 = False
        else:
            self.use_1 = True
            layers_1 = []
            for i in range(len(hidden_list_1) - 1):
                layers_1.append(
                    nn.Linear(hidden_list_1[i], hidden_list_1[i+1]))
                if i < len(hidden_list_1) - 2:
                    layers_1.append(self.act_functions(act))
                    layers_1.append(nn.Dropout(p=dropout_rate))
            self.model_1 = nn.Sequential(*layers_1)

        if hidden_list_2[-1] == 0:
            self.use_2 = False
        else:
            self.use_2 = True
            layers_2 = []
            for i in range(len(hidden_list_2) - 1):
                layers_2.append(
                    nn.Linear(hidden_list_2[i], hidden_list_2[i+1]))
                if i < len(hidden_list_2) - 2:
                    layers_2.append(self.act_functions(act))
                    layers_2.append(nn.Dropout(p=dropout_rate))
            self.model_2 = nn.Sequential(*layers_2)



        self.gate= GatedUnit(
            [hidden_list_1[-1],hidden_list_2[-1]],
            hidden_list_agg[0]
            )
        
        layers_agg = []
        for i in range(len(hidden_list_agg) - 1):
            layers_agg.append(
                nn.Linear(hidden_list_agg[i], hidden_list_agg[i+1]))
            if i < len(hidden_list_agg) - 2:
                layers_agg.append(self.act_functions(act))
                layers_agg.append(nn.Dropout(p=dropout_rate))
        self.model_agg = nn.Sequential(*layers_agg)
        self.dropout = nn.Dropout(dropout_rate)
        self.nhead = nhead
        self.N_sample_train = N_sample_train
    def forward(self, input_1, input_2):

        if self.use_2 and self.use_1:
            
            X =torch.cat((input_1,input_2),dim=-1)
            X = X.transpose(0, 1)
            if self.training:
                keys = self.K_pro(X)
                querys = self.Q_pro(X)
                values = self.V_pro(X)
                _, attn_weight = self.attention(querys,keys,values,attn_mask=self.mask,average_attn_weights=False)
                self.attn_weight = attn_weight.detach().clone()
                attn_output_mulhead = torch.zeros(size=[self.nhead,self.feature_all,input_1.shape[0]],device = input_1.device)
                for head_index in range(self.nhead):
                    attn_output_mulhead[head_index] = torch.matmul(attn_weight[head_index],X)
                    
            elif not self.training:
                attn_output_mulhead = torch.zeros(size=[self.nhead,self.feature_all,input_1.shape[0]],device = input_1.device)
                for head_index in range(self.nhead):
                    attn_output_mulhead[head_index] = torch.matmul(self.attn_weight[head_index],X)
            attn_output = attn_output_mulhead.mean(dim=0)
            attn_output = (attn_output+X).transpose(0,1) 
            proc_1 = attn_output[:,:self.feature_1] #same shape as input 1
            proc_2 = attn_output[:,self.feature_1:]
            
            output_1 = self.model_1(proc_1)
            output_2 = self.model_2(proc_2)
            
            input_of_agg = self.gate([output_1,output_2])
            
        elif self.use_1:
            input_of_agg = self.model_1(input_1)
        elif self.use_2:
            input_of_agg = self.model_2(input_2)
        else:
            raise ValueError
        x = self.model_agg(input_of_agg)
        x = self.dropout(x)
        return x

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
class GatedUnit(nn.Module):
    def __init__(self, inuput_dim:list,output_dim):
        super(GatedUnit, self).__init__()
        
        self.fc_gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(inuput_dim[index], output_dim) ,
                )
            for index in range(len(inuput_dim))
            ])
        self.proj_layers = nn.ModuleList([
                nn.Linear(inuput_dim[index], output_dim)
                    for index in range(len(inuput_dim))
            ])
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    def forward(self, x_list:list):
        # get gate tensor [0~1]
        temp = [self.fc_gate[index](x_list[index]) for index in range(len(x_list))]
        addition = 0
        for element in temp:
            addition += element
        gate = self.sigmoid(addition)
        # Controlling Data Flow using "gate"
        result = gate * self.proj_layers[0](x_list[0]) \
            + (1-gate)*self.proj_layers[1](x_list[1])
        
        return result
    
class Gated_model(nn.Module):
    def __init__(self, hidden_list_1, hidden_list_2, hidden_list_agg, dropout_rate, act):
        super(Gated_model, self).__init__()

        if hidden_list_1[-1] == 0:
            self.use_1 = False
        else:
            self.use_1 = True
            layers_1 = []
            for i in range(len(hidden_list_1) - 1):
                layers_1.append(
                    nn.Linear(hidden_list_1[i], hidden_list_1[i+1]))
                if i < len(hidden_list_1) - 2:
                    layers_1.append(self.act_functions(act))
                    layers_1.append(nn.Dropout(p=dropout_rate))
            self.model_1 = nn.Sequential(*layers_1)

        if hidden_list_2[-1] == 0:
            self.use_2 = False
        else:
            self.use_2 = True
            layers_2 = []
            for i in range(len(hidden_list_2) - 1):
                layers_2.append(
                    nn.Linear(hidden_list_2[i], hidden_list_2[i+1]))
                if i < len(hidden_list_2) - 2:
                    layers_2.append(self.act_functions(act))
                    layers_2.append(nn.Dropout(p=dropout_rate))
            self.model_2 = nn.Sequential(*layers_2)

        self.gate = GatedBlock(hidden_list_1[-1],hidden_list_2[-1])
        # self.gate= GatedUnit(
        #     [hidden_list_1[-1],hidden_list_2[-1]],
        #     hidden_list_agg[0]
        #     )
        layers_agg = []
        for i in range(len(hidden_list_agg) - 1):
            layers_agg.append(
                nn.Linear(hidden_list_agg[i], hidden_list_agg[i+1]))
            if i < len(hidden_list_agg) - 2:
                layers_agg.append(self.act_functions(act))
                layers_agg.append(nn.Dropout(p=dropout_rate))
        self.model_agg = nn.Sequential(*layers_agg)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_1, input_2):
        if self.use_2 and self.use_1:
            input_2 = self.model_2(input_2)
            input_1 = self.model_1(input_1)
            x = self.gate(x_list = [input_1,input_2])
        elif self.use_1:
            x = self.model_1(input_1)
        elif self.use_2:
            x = self.model_2(input_2)
        else:
            raise ValueError
        x = self.model_agg(x)
        x = self.dropout(x)
        return x


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
class Mixed_model(nn.Module):
    def __init__(self, hidden_list_1, hidden_list_2, hidden_list_agg, dropout_rate, act):
        super(Mixed_model, self).__init__()

        if hidden_list_1[-1] == 0:
            self.use_1 = False
        else:
            self.use_1 = True
            layers_1 = []
            for i in range(len(hidden_list_1) - 1):
                layers_1.append(
                    nn.Linear(hidden_list_1[i], hidden_list_1[i+1]))
                if i < len(hidden_list_1) - 2:
                    layers_1.append(self.act_functions(act))
                    layers_1.append(nn.Dropout(p=dropout_rate))
            self.model_1 = nn.Sequential(*layers_1)

        if hidden_list_2[-1] == 0:
            self.use_2 = False
        else:
            self.use_2 = True
            layers_2 = []
            for i in range(len(hidden_list_2) - 1):
                layers_2.append(
                    nn.Linear(hidden_list_2[i], hidden_list_2[i+1]))
                if i < len(hidden_list_2) - 2:
                    layers_2.append(self.act_functions(act))
                    layers_2.append(nn.Dropout(p=dropout_rate))
            self.model_2 = nn.Sequential(*layers_2)

        layers_agg = []
        for i in range(len(hidden_list_agg) - 1):
            layers_agg.append(
                nn.Linear(hidden_list_agg[i], hidden_list_agg[i+1]))
            if i < len(hidden_list_agg) - 2:
                layers_agg.append(self.act_functions(act))
                layers_agg.append(nn.Dropout(p=dropout_rate))
        self.model_agg = nn.Sequential(*layers_agg)
        self.dropout = nn.Dropout(dropout_rate)
        

    def forward(self, input_1, input_2):
        if self.use_2 and self.use_1:
            input_2 = self.model_2(input_2)
            input_1 = self.model_1(input_1)
            x = torch.cat((input_2, input_1), dim=-1)
        elif self.use_1:
            x = self.model_1(input_1)
        elif self.use_2:
            x = self.model_2 (input_2)
        else:
            raise ValueError
        x = self.model_agg(x)
        x = self.dropout(x)
        return x

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




def validate(model, val_dataset):
    """

    """
    model.eval()  # 将模型设置为评估模式

    with torch.no_grad():
        output = model(val_dataset['features_0'],
                       val_dataset['features_1'])

    auc_score = roc_auc_score(
        val_dataset['labels'].cpu(), output.detach().cpu(), average='micro')
    # fpr, tpr, thresholds = roc_curve(val_dataset['labels'].cpu(), output.detach().cpu())
    # auc_score = auc(fpr, tpr)
    return auc_score


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


# def columnwise_correlation(tensor1, tensor2):
#     # 确保输入是2D张量
#     if tensor1.dim() != 2 or tensor2.dim() != 2:
#         raise ValueError("Both tensors must be 2D")

#     # 确保两个张量的行数相同
#     if tensor1.size(0) != tensor2.size(0):
#         raise ValueError("Both tensors must have the same number of rows")

#     eps=1e-8
#     # 标准化每一列（去均值，归一化）
#     tensor1_mean = tensor1.mean(dim=0)
#     tensor1_std = tensor1.std(dim=0)
#     tensor2_mean = tensor2.mean(dim=0)
#     tensor2_std = tensor2.std(dim=0)

#     tensor1_std[tensor1_std==0.]+=eps
#     tensor2_std[tensor2_std==0.]+=eps

#     tensor1_normalized = (tensor1 - tensor1_mean) / tensor1_std
#     tensor2_normalized = (tensor2 - tensor2_mean) / tensor2_std
#     check_tensor(tensor1_normalized)
#     check_tensor(tensor2_normalized)

#     # 计算标准化后的张量的点积
#     correlation_matrix = torch.mm(tensor1_normalized.t(), tensor2_normalized) / (tensor1.size(0) - 1)

#     return correlation_matrix

    

def columnwise_correlation(tensor1, tensor2):
    # 确保输入是2D张量
    if tensor1.dim() != 2 or tensor2.dim() != 2:
        raise ValueError("Both tensors must be 2D")

    # 确保两个张量的行数相同
    if tensor1.size(0) != tensor2.size(0):
        raise ValueError("Both tensors must have the same number of rows")

    # 将两个张量合并为一个张量
    combined_tensor = torch.cat((tensor1, tensor2), dim=1)

    # 计算相关系数矩阵
    correlation_matrix = torch.corrcoef(combined_tensor.T)

    # 从相关系数矩阵中提取对应列之间的相关系数
    num_cols_tensor1 = tensor1.size(1)
    correlation_matrix = correlation_matrix[:
                                            num_cols_tensor1, num_cols_tensor1:]

    return correlation_matrix
