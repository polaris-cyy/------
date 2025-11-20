from utils import DGraphFin
from utils.utils import prepare_folder
from utils.evaluator import Evaluator

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T

import numpy as np
from torch_geometric.data import Data
import os

#设置gpu设备
device = 0
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)


path='./datasets/632d74d4e2843a53167ee9a1-momodel/' #数据保存路径
save_dir='./results/' #模型保存路径
dataset_name='DGraph'
dataset = DGraphFin(root=path, name=dataset_name, transform=T.ToSparseTensor())

nlabels = dataset.num_classes
if dataset_name in ['DGraph']:
    nlabels = 2    #本实验中仅需预测类0和类1

data = dataset[0]
data.adj_t = data.adj_t.to_symmetric() #将有向图转化为无向图


if dataset_name in ['DGraph']:
    x = data.x
    x = (x - x.mean(0)) / x.std(0)
    data.x = x
if data.y.dim() == 2:
    data.y = data.y.squeeze(1)

split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}  #划分训练集，验证集

train_idx = split_idx['train']
result_dir = prepare_folder(dataset_name,'mlp')

import torch
import random
from torch import nn
import torch.nn.functional as F

class SageLayer(nn.Module):
    def __init__(self, input_size, out_size, gcn=False):
        super(SageLayer, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.gcn = gcn
        self.weight = nn.Parameter(torch.FloatTensor(out_size, input_size * (1 if gcn else 2)))
        self.init_params()
    
    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)
        
    def forward(self, self_feats, aggregate_feats, neighs=None):
        if not self.gcn:
            combined = torch.cat([self_feats, aggregate_feats], dim=1)
        else:
            combined = aggregate_feats
        combined = F.relu(self.weight.mm(combined.t())).t()
        return combined
    
class Classification(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(Classification, self).__init__()
        self.layer = nn.Sequential(nn.Linear(embedding_size, num_classes))
        self.init_params()
    
    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
    
    def forward(self, embeds):
        logists = torch.log_softmax(self.layer(embeds), 1)
        return logists

class GraphSage(nn.Module):
    def __init__(
        self,
        num_layers,
        input_size,
        out_size,
        raw_features,
        adj_lists,
        device,
        num_classes=2,
        gcn=False,
        agg_func="mean",
    ):
        super(GraphSage, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.num_layers = num_layers
        self.raw_features = raw_features
        self.gcn = gcn
        self.device = device
        self.agg_func = agg_func
        self.adj_lists = adj_lists
        self.classifier = Classification(out_size, num_classes)
        layers = []
        
        for index in range(1, num_layers+1):
            layer_size = out_size if index != 1 else input_size
            layers.append(SageLayer(layer_size, out_size, self.gcn))
        layers.append(self.classifier)
        self.layers = nn.ModuleList(layers)
        
    def forward(self, nodes_batch):
        lower_layer_nodes = list(nodes_batch) #把当前训练的结点转化成list
        nodes_batch_layers = [(lower_layer_nodes,)]
        for i in range(self.num_layers):
            lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes = self._get_unique_neighs_list(lower_layer_nodes)
            nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))
        pre_hidden_embs = self.raw_features
        for index in range(1, self.num_layers + 1):
            nb = nodes_batch_layers[index][0]
            pre_neighs = nodes_batch_layers[index-1]
            aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)
            sage_layer = self.layers[index - 1]
            if index > 1:
                nb = self._nodes_map(nb, pre_hidden_embs, pre_neighs)
            cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb], aggregate_feats=aggregate_feats)
            pre_hidden_embs = cur_hidden_embs
        return self.layers[-1](pre_hidden_embs)
    
    def _nodes_map(self, nodes, hidden_embs, neighs):
        layer_nodes, samp_neighs, layer_nodes_dict = neighs
        assert len(samp_neighs) == len(nodes)
        index = [layer_nodes_dict[x] for x in nodes]                    # 记录将上一层的节点编号。
        return index
    
    def _get_unique_neighs_list(self, nodes, num_sample=10):
        """
        从 SparseTensor (self.adj_lists) 高效采样邻居，复杂度 O(B × d)
        """
        adj_t = self.adj_lists
        if hasattr(nodes, "tolist"):
            nodes = nodes.tolist()

        # === 用 CSR 索引取邻居 ===
        # rowptr[i]: 第 i 行（节点）的邻居起始位置
        # col[rowptr[i]:rowptr[i+1]]: 第 i 行的所有邻居
        rowptr, col, _ = adj_t.csr()

        samp_neighs = []
        for node in nodes:
            start = rowptr[node].item()
            end = rowptr[node + 1].item()
            neigh = col[start:end].tolist()

            if len(neigh) == 0:
                s = set()
            elif num_sample is None:
                s = set(neigh)
            elif len(neigh) >= num_sample:
                s = set(random.sample(neigh, num_sample))
            else:
                s = set(random.choices(neigh, k=num_sample))

            # 包含自身
            s.add(node)
            samp_neighs.append(s)

        # 取并集得到涉及到的所有节点
        unique_nodes_list = list(set().union(*samp_neighs))
        unique_nodes = {nid: idx for idx, nid in enumerate(unique_nodes_list)}

        return samp_neighs, unique_nodes, unique_nodes_list

    def aggregate(self, nodes, pre_hidden_embs, pre_neighs, num_sample=10):
        unique_nodes_list, samp_neighs, unique_nodes = pre_neighs        # batch涉及到的所有节点,本身+邻居,邻居节点编号->字典中编号  
        assert len(nodes) == len(samp_neighs)
        indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]  # 是否包含本身
        assert (False not in indicator)
        if not self.gcn:
            samp_neighs = [(samp_neighs[i]-set([nodes[i]])) for i in range(len(samp_neighs))]  # 在把中心节点去掉
        if len(pre_hidden_embs) == len(unique_nodes):                     # 保留需要使用的节点特征。
            embed_matrix = pre_hidden_embs
        else:
            embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]                                               
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))           # (本层节点数量，邻居节点数量)
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]  # 保存列 每一行对应的邻居真实index做为列。
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]# 保存行 每行邻居数
        mask[row_indices, column_indices] = 1
        num_neigh = mask.sum(1, keepdim=True)                         # 按行求和，保持和输入一个维度
        mask = mask.div(num_neigh).to(embed_matrix.device)            # 归一化操作
        aggregate_feats = mask.mm(embed_matrix)                       # 矩阵相乘，相当于聚合周围邻接信息求和

        return aggregate_feats

import math
import tqdm
from sklearn.metrics import roc_auc_score

torch.manual_seed(0)
def train_one_epoch(data, graphsage, batch_size, device, optimizer):
    graphsage.train()

    train_idx = data.train_mask
    perm = torch.randperm(train_idx.size(0), device=device)
    train_idx = train_idx[perm]
    labels = data.y.to(device)

    total_loss = 0.0
    batches = math.ceil(len(train_idx) / batch_size)
    pbar = tqdm.trange(1, batches + 1, desc="Training", ncols=100)

    for index in pbar:
        start = (index - 1) * batch_size
        end = index * batch_size
        nodes_batch = train_idx[start:end]

        labels_batch = labels[nodes_batch]
        logits = graphsage(nodes_batch)

        loss = F.nll_loss(logits, labels_batch)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(graphsage.parameters()) , 5)
        optimizer.step()
        avg_loss = total_loss / index
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{avg_loss:.4f}",
            })

    return total_loss / batches


def train(data, model, batch_size, device, epoches, name="graphsage"):
    max_val_auc = 0.0
    optimizer = torch.optim.Adam(
        list(model.parameters()), lr=5e-4
    )

    for epoch in range(1, epoches + 1):
        print(f"\n===== Epoch {epoch} =====")
        loss = train_one_epoch(data, model, batch_size, device, optimizer)
        print(f"Train Loss: {loss:.4f}")
        max_val_auc = evaluate(data, model, batch_size, device, max_val_auc, name, epoch)

    print(f"Best Validation AUC: {max_val_auc:.4f}")

import torch
from sklearn.metrics import roc_auc_score

@torch.no_grad()
def evaluate(data, model, batch_size, device, max_val_auc, name, cur_epoch):
    model.eval()

    x, y = data.x.to(device), data.y.to(device)
    val_idx = data.valid_mask
    test_idx = data.test_mask

    # === 验证集 ===
    val_probs = torch.zeros(len(val_idx), device=device)
    for i in tqdm.tqdm(range(0, len(val_idx), batch_size), desc=f"Validating Epoch {cur_epoch}"):
        batch = val_idx[i:i+batch_size]
        logits = model(batch)
        val_probs[i:i+batch_size] = logits.exp()[:, 1]

    val_labels = y[val_idx]
    val_auc = roc_auc_score(val_labels.cpu(), val_probs.cpu())
    print(f"Validation AUC: {val_auc:.4f}")

    # === 若更优，再测试 ===
    if val_auc > max_val_auc:
        torch.save(
            model.state_dict(),
            f'./results/model_best_{name}_ep{cur_epoch}_{val_auc:.4f}.pt'
        )
        max_val_auc = val_auc

    return max_val_auc


torch.manual_seed(0)
num_labels = 2
batch_size = 100
device = "cuda:0" if torch.cuda.is_available() else "cpu"
feature_size = 20
epoches = 100
hidden_size = 64
num_layers = 2 # 采样邻居depth，论文中的B

data = data.to(device)
model = GraphSage(
    num_layers, 
    data.x.size(-1),
    hidden_size,
    data.x,
    data.adj_t,
    device
).to(device)
## 生成 main.py 时请勾选此 cell
from utils import DGraphFin
from utils.evaluator import Evaluator
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import Data
import numpy as np
import os

# 这里可以加载你的模型
model.load_state_dict(torch.load('./results/small/model_best_graphsage_ep21_0.7700.pt'))

def predict(data,node_id):
    """
    加载模型和模型预测
    :param node_id: int, 需要进行预测节点的下标
    :return: tensor, 类0以及类1的概率, torch.size[1,2]
    """
    
    # 模型预测时，测试数据已经进行了归一化处理
    # -------------------------- 实现模型预测部分的代码 ---------------------------
    with torch.no_grad():
        model.eval()
        out = model([node_id])
        y_pred = out.exp()  # (N,num_classes)
        
    return y_pred.squeeze(0)
