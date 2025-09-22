import random
import os
import torch
import argparse
from tqdm import tqdm
import dgl
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import MinMaxScaler



# Set argument
# 一阶二阶邻居完全分离,传播前后多尺度一致性对比,对比06更改得分
#cora:95.27, citeseer:95.66 dblp:73.01 citation:71.54 pubmed:94.03
parser = argparse.ArgumentParser(description='CNCL-GAD')
parser.add_argument('--dataset', type=str, default='cora')  # 'cora'  'citeseer'  'pubmed'
parser.add_argument('--lr', type=float)
parser.add_argument('--cuda', type=str, default='2')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=3)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--epochs', type=int, default=1)#100
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
parser.add_argument('--auc_test_rounds', type=int, default=1)#256
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--alpha', type=float, help='control the balance between origianl features')
parser.add_argument('--beta', type=float, help='control the balance of nn')
parser.add_argument('--gama', type=float, help='control the balance between different level of contrastive learning')
parser.add_argument('--temperature', type=float, default=5, help='temperature for fx')
parser.add_argument('--have_neg', type=bool, help='anomaly score and LOSS contain negtive pairs OT', default=True)
parser.add_argument('--neg_top_k', type=float, help='top max k of OT to select negtive pairs', default=20)
parser.add_argument('--K_1', type=int, help='view 1')
parser.add_argument('--K_2', type=int, help='view 2')
parser.add_argument('--restart_prob_1', type=float, help='RWR restart probability on view 1', default=0.9)
parser.add_argument('--restart_prob_2', type=float, help='RWR restart probability on view 2', default=0.3)
parser.add_argument('--subgraph_mode', type=str, default='1+2')
args = parser.parse_args()
GPU = args.cuda #设置GPU 0 1 2 可见
os.environ['CUDA_VISIBLE_DEVICES'] =GPU
if args.lr is None:
    if args.dataset in ['cora', 'citeseer', 'pubmed', 'dblp', 'citation']:
        args.lr = 2e-3
    elif args.dataset == 'BlogCatalog':
        args.lr = 1e-2
    elif args.dataset == 'ACM':
        args.lr = 5e-3

if args.alpha is None:
    if args.dataset in ['cora']:
        args.alpha = 0.6#0.6
    elif args.dataset =='citeseer':
        args.alpha = 0.8#0.8
    elif args.dataset == 'pubmed':
        args.alpha = 0.5
    elif args.dataset == 'dblp':
        args.alpha = 0.2#0.2
    elif args.dataset == 'citation':
        args.alpha = 0.5#0.4
if args.beta is None:
    if args.dataset == 'cora':
        args.beta = 0.7#0.7
    elif args.dataset == 'pubmed':
        args.beta = 0.8
    elif args.dataset == 'citeseer':
        args.beta = 0.5#0.5
    elif args.dataset == 'dblp':
        args.beta = 0.8#0.3
    elif args.dataset == 'citation':
        args.beta = 0.4#0.4

if args.gama is None:
    if args.dataset == 'cora':
        args.gama = 0.8#0.8
    elif args.dataset == 'pubmed':
        args.gama = 0.8
    elif args.dataset == 'citeseer':
        args.gama = 0.3#0.5
    elif args.dataset == 'dblp':
        args.gama = 0.3#0.3
    elif args.dataset == 'citation':
        args.gama = 0.2#0.3

if args.K_1 is None:
    if args.dataset == 'cora':
        args.K_1 = 5#5
    elif args.dataset in ['citeseer']:
        args.K_1 = 10#10
    elif args.dataset in ['pubmed']:
        args.K_1 = 5
    elif args.dataset in ['dblp']:
        args.K_1 = 6#6
    elif args.dataset in ['citation']:
        args.K_1 = 8#8

if args.K_2 is None:
    if args.dataset == 'citeseer':
        args.K_2 = 8#10
    elif args.dataset == 'cora':
        args.K_2 = 7#7
    elif args.dataset in ['pubmed']:
        args.K_2 = 8
    elif args.dataset in ['dblp']:
        args.K_2 = 8#8
    elif args.dataset in ['citation']:
        args.K_2 = 6#4

AUC_list = []
batch_size = args.batch_size
subgraph_size_1 = args.K_1
subgraph_size_2 = args.K_2
print('Dataset: ', args.dataset)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed = args.seed
dgl.random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import networkx as nx
import scipy.sparse as sp
import scipy.io as sio
import random
import torch
# Load and preprocess data
def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
def load_mat(dataset, train_rate=0.3, val_rate=0.1):
    """Load .mat dataset."""
    data = sio.loadmat("./dataset/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']
    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)
    labels = np.squeeze(np.array(data['Class'], dtype=np.int64) - 1)
    num_classes = np.max(labels) + 1
    labels = dense_to_one_hot(labels, num_classes)
    ano_labels = np.squeeze(np.array(label))

    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[: num_train]
    idx_val = all_idx[num_train: num_train + num_val]
    idx_test = all_idx[num_train + num_val:]

    return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels

adj, features, labels, idx_train, idx_val, \
idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)

np.save('./' + args.dataset + 'labels.npy', ano_label)


A = adj
degree = np.sum(adj, axis=0)
degree_ave = np.mean(degree)
dgl_graph = dgl.from_scipy(adj)
raw_feature = features.todense()

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape
    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx
def idx_sample(idxes):
    num_idx = len(idxes)
    
    idx = torch.arange(0, num_idx)
    random_add = torch.randint(low=1, high=num_idx, device='cpu', size=idx.size())
    shuffled_idx = torch.remainder(idx+random_add, num_idx)

    return shuffled_idx
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)
features, _ = preprocess_features(features)
nb_nodes = features.shape[0]
ft_size = features.shape[1]
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    adj_raw = adj
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo(), adj_raw
adj, adj_raw = normalize_adj(adj)
adj = (adj + sp.eye(adj.shape[0])).todense()
adj_raw = adj_raw.todense()
features = torch.FloatTensor(features[np.newaxis])
raw_feature = torch.FloatTensor(raw_feature[np.newaxis])
adj = torch.FloatTensor(adj[np.newaxis])
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import torch.distributions as dists
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, activation=nn.PReLU()):
        super().__init__()
        self.encoder = nn.ModuleList([
            nn.Linear(in_dim, out_dim),
            activation,
            nn.Linear(out_dim, out_dim),
            activation,
        ])
    def forward(self, features):
        h = features
        for layer in self.encoder:
            h = layer(h)
        h = F.normalize(h, p=2, dim=1)  # row normalize
        return h
class GCN(nn.Module):
    def __init__(self, n_h, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(n_h, n_h)
        self.act = nn.PReLU() if act == 'prelu' else act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        out = F.normalize(out, p=2, dim=1)
        return self.act(out)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self, graph, node, neg, sim_pos=False):
        scs = []
        # positive

        scs.append(F.cosine_similarity(node, graph, dim=-1))
        scs.append(F.cosine_similarity(node, neg, dim=-1))
        logits = torch.cat(tuple(scs))
        if sim_pos:
            return logits.unsqueeze(dim=-1), F.cosine_similarity(node, graph, dim=-1).unsqueeze(dim=-1)
        else:
            return logits.unsqueeze(dim=-1)

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values

def matrix_diag(diagonal):
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result


def similarity( reps1, reps2 ):
    reps1_unit = F.normalize(reps1, dim=-1)
    reps2_unit = F.normalize(reps2, dim=-1)
    if len(reps1.shape) == 2:
        sim_mat = torch.einsum("ik,jk->ij", [reps1_unit, reps2_unit])
    elif len(reps1.shape) == 3:
        sim_mat = torch.einsum('bik,bjk->bij', [reps1_unit, reps2_unit])
    else:
        print(f"{len(reps1.shape)} dimension tensor is not supported for this function!")
    return sim_mat

class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout, have_neg = False, hidden_size = 128,
                 temperature=0.4, lamb=20, neg_top_k=50):
        super(Model, self).__init__()
        self.read_mode = readout
        self.hidden_size = hidden_size
        self.encoder = MLP(n_in, n_h)
        self.gcn_ns = GCN(n_h, n_h, activation)
        self.gcn_nn = GCN(n_h, n_h, activation)
        self.gcn_ss = GCN(n_h, n_h, activation)
        self.temperature = temperature
        self.lamb = lamb
        self.have_neg = have_neg
        self.neg_top_k = neg_top_k
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        self.discriminator = Discriminator()
        self.norm = nn.LayerNorm(n_h)
    def dominant_view_mining(self, node1, node2, feature, graph1, graph2, graph):

        err_list = []
        feat_sim = torch.mm(feature, feature.t())
        graph_sim = torch.mm(graph, graph.t())

        node1_sim = torch.mm(node1, node1.t())
        node2_sim = torch.mm(node2, node2.t())

        node1_sim = torch.mm(graph1, graph1.t())
        node2_sim = torch.mm(graph2, graph2.t())
        
        err1_node = F.cosine_similarity(node1_sim, feat_sim).mean()
        err1_graph = F.cosine_similarity(node1_sim, graph_sim).mean()
        err2_node  = F.cosine_similarity(node2_sim, feat_sim).mean()
        err2_graph  = F.cosine_similarity(node2_sim, graph_sim).mean()
        
     
        err_list.append(err1_node+err1_graph)
        err_list.append(err2_node+err2_graph)

        dominant_index = torch.argmin(torch.tensor(err_list))
        #err_loss = sum(err_list) / len(err_list)

        return dominant_index
    
    def forward(self, feature1, adj1, feature2, adj2, sparse=False, train=True):
        
        # batch_size * sub_graph_size+1 * feature_dim
        feature = feature1.clone()
        h1 = self.encoder(feature)
        #h_graph_read1 = self.read(h1[:, : -1, :])
        h_graph_read1 = self.read(h1)
        h_node1 = h1[:, -1, :]

        h2 = self.norm(self.encoder(feature2))
        #h_graph_read2 = self.read(h1[:, : -1, :])
        h_graph_read2 = self.read(h1)
        h_node2 = h2[:, -1, :]
        dominant_index1 = self.dominant_view_mining(h_node1, h_node2, feature[:, -1, :], h_graph_read1, h_graph_read2, self.read(feature))
        h_node = [h_node1, h_node2]
        h_graph = [h_graph_read1, h_graph_read2]

        idx = torch.arange(0, h_graph_read2.shape[0])
        neg_idx = idx_sample(idx)
        #传播前节点-子图级
        disc_ns1, sim_ns = self.discriminator(h_graph[dominant_index1], h_node[dominant_index1], h_graph[dominant_index1][neg_idx], True)
        #disc_ns2 = self.discriminator(h_graph_read2, h_node2, h_graph_read2[neg_idx])
        #传播前节点-节点级
        disc_nn1, sim_nn1 = self.discriminator(h_node[1-dominant_index1], h_node[dominant_index1], h_node[dominant_index1][neg_idx], True)
        disc_nn2, sim_nn2 = self.discriminator(h_node[1-dominant_index1], h_node[dominant_index1], h_node[1-dominant_index1][neg_idx], True)

        #传播前子图-子图级
        disc_ss1, sim_ss1 = self.discriminator(h_graph[1-dominant_index1], h_graph[dominant_index1], h_graph[dominant_index1][neg_idx], True)
        disc_ss2, sim_ss2 = self.discriminator(h_graph[1-dominant_index1], h_graph[dominant_index1], h_graph[1-dominant_index1][neg_idx], True)

        ####----------------
        #1
        h1_emb_ns = self.norm(self.gcn_ns(h1, adj1, sparse))
        h1_n_ns = h1_emb_ns[:, -1, :]
        h1_a_ns = self.read(h1_emb_ns)

        # 2
        h2_emb_ns = self.norm(self.gcn_ns(h2, adj2, sparse))
        h2_n_ns = h2_emb_ns[:, -1, :]
        h2_a_ns = self.read(h2_emb_ns)

        dominant_index2 = self.dominant_view_mining(h1_n_ns, h2_n_ns, h_node[dominant_index1], h1_a_ns, h2_a_ns, h_graph[dominant_index1])
        #dominant_index2 = self.dominant_view_mining(h1_n_ns, h2_n_ns, feature[:, -1, :], h1_a_ns, h2_a_ns, self.read(feature))
        h_node_p = [h1_n_ns, h2_n_ns]
        h_graph_p = [h1_a_ns, h2_a_ns]

        #传播后节点-子图级一致性学习
        disc_ns_1, sim_ns_11 = self.discriminator(h_graph_p[dominant_index2], h_node_p[dominant_index2], h_graph_p[dominant_index2][neg_idx], True)
        #disc_ns_2 = self.discriminator(h2_a_ns, h2_n_ns, h2_a_ns[neg_idx])
        
        #传播后节点-节点级一致性学习
        disc_nn_11, sim_nn11 = self.discriminator(h_node_p[1-dominant_index2], h_node_p[dominant_index2], h_node_p[dominant_index2][neg_idx], True)
        disc_nn_12, sim_nn12 = self.discriminator(h_node_p[1-dominant_index2], h_node_p[dominant_index2], h_node_p[1-dominant_index2][neg_idx], True)
        #传播后图级-图级级一致性学习
        disc_ss_21, sim_ss21 = self.discriminator(h_graph_p[1-dominant_index2], h_graph_p[dominant_index2], h_graph_p[dominant_index2][neg_idx], True)
        disc_ss_22, sim_ss22 = self.discriminator(h_graph_p[1-dominant_index2], h_graph_p[dominant_index2], h_graph_p[1-dominant_index2][neg_idx], True)
        if train:
            return disc_ns1, (disc_nn1+disc_nn2)*0.5, (disc_ss1+disc_ss2)*0.5, disc_ns_1,(disc_nn_11+disc_nn_12)*0.5,(disc_ss_21+disc_ss_22)*0.5
        else:
            #return (disc_ns1+disc_ns2)*0.5, (disc_nn1+disc_nn2)*0.5, (disc_ss1+disc_ss2)*0.5, (disc_ns_1+disc_ns_2)*0.5,(disc_nn_11+disc_nn_12)*0.5,(disc_ss_21+disc_ss_22)*0.5, \
        #(sim_nn1+sim_nn2)*0.5, (sim_ss1+sim_ss2)*0.5, (sim_nn11+sim_nn12)*0.5, (sim_ss21+sim_ss22)*0.5
            return sim_ns, sim_ns_11, \
        (sim_nn1+sim_nn2)*0.5, (sim_ss1+sim_ss2)*0.5, (sim_nn11+sim_nn12)*0.5, (sim_ss21+sim_ss22)*0.5
    def mse_loss(self, x, y):
        return ((y - x)**2).sum(-1).mean()

kwargs = {'metric': 'cosine', 'distributed': True,'random_state': 0, 'n_clusters': 2, 'verbose': False}

model = Model(n_in=ft_size, n_h=args.embedding_dim, activation='prelu', negsamp_round=args.negsamp_ratio,
            readout=args.readout, hidden_size=args.hidden_size, temperature=args.temperature,
            have_neg = args.have_neg, neg_top_k = args.neg_top_k)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if torch.cuda.is_available():
    print('Using CUDA')
    model.to(device)
    features = features.to(device)
    raw_feature = raw_feature.to(device)
    adj = adj.to(device)
    #b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).to(device))
    b_xent = nn.BCEWithLogitsLoss().to(device)
else:
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0
mse_loss = nn.MSELoss(reduction='mean')
if nb_nodes % batch_size == 0:
    batch_num = nb_nodes // batch_size
else:
    batch_num = nb_nodes // batch_size + 1
def get_first_adj(dgl_graph, adj, subgraph_size):
    """Generate the first view's subgraph with the first-order neighbor."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    subgraphs = []
    adj = np.array(adj.todense()).squeeze()
    for node_id in all_idx:
        first_adj = np.where(adj[node_id] == 1)
        first_adj = list(first_adj[0])
        if len(first_adj) < subgraph_size - 1:
            subgraphs.append(first_adj)
            first_adj.append(node_id) #自己也可以被循环选择
            subgraphs[node_id].extend(
                list(np.random.choice(first_adj, subgraph_size - len(first_adj) - 1, replace=True)))
        else:
            subgraphs.append(list(np.random.choice(first_adj, subgraph_size - 1, replace=False)))
        subgraphs[node_id].append(node_id)
    return subgraphs
def get_second_adj(dgl_graph, adj, subgraph_size):
    #Generate the second view's subgraph with the 1/2 first-order and 1/2 second-order neighbor.
    all_idx = list(range(dgl_graph.number_of_nodes()))
    subgraphs = []
    adj_2 = adj.dot(adj)
    adj = np.array(adj.todense())
    adj_2 = np.array(adj_2.todense())
    row, col = np.diag_indices_from(adj_2)
    zeros = np.zeros(adj_2.shape[0])
    adj_2[row, col] = np.array(zeros)
    adj = adj.squeeze()
    adj_2 = adj_2.squeeze()
    for node_id in all_idx:
        first_adj = np.where(adj[node_id] == 1)
        second_adj = np.where(adj_2[node_id] != 0)
        first_adj = first_adj[0].tolist()
        second_adj = second_adj[0].tolist()
        if len(first_adj) < subgraph_size // 2:
            subgraphs.append(list(np.random.choice(first_adj, subgraph_size // 2, replace=True)))
            if len(second_adj) == 0:
                first_adj.append(node_id)
                subgraphs[node_id].extend(list(np.random.choice(first_adj, (subgraph_size - 1) // 2, replace=True)))
            elif len(second_adj) < (subgraph_size - 1) // 2:
                subgraphs[node_id].extend(list(np.random.choice(second_adj, (subgraph_size - 1) // 2, replace=True)))
            else:
                subgraphs[node_id].extend(list(np.random.choice(second_adj, (subgraph_size - 1) // 2, replace=False)))
        else:
            if len(second_adj) == 0:
                first_adj.append(node_id)
                if len(first_adj) < subgraph_size - 1:
                    subgraphs.append(list(np.random.choice(first_adj, (subgraph_size - 1), replace=True)))
                else:
                    subgraphs.append(list(np.random.choice(first_adj, (subgraph_size - 1), replace=False)))
            elif len(second_adj) < (subgraph_size - 1) // 2 :
                subgraphs.append(list(np.random.choice(first_adj, subgraph_size // 2, replace=False)))
                subgraphs[node_id].extend(list(np.random.choice(second_adj, (subgraph_size - 1) // 2, replace=True)))
            else:
                subgraphs.append(list(np.random.choice(first_adj, subgraph_size // 2, replace=False)))
                subgraphs[node_id].extend(list(np.random.choice(second_adj, (subgraph_size - 1) // 2, replace=False)))
        subgraphs[node_id].append(node_id)
    return subgraphs
def generate_subgraph(args, dgl_graph, A, subgraph_size_1, subgraph_size_2):
    """Generate subgraph with RWR/first & second -neiborhood algorithm."""
    restart_prob_1 = args.restart_prob_1
    restart_prob_2 = args.restart_prob_2
    if args.subgraph_mode == '1+2':
        subgraphs_1 = get_first_adj(dgl_graph, A, subgraph_size_1)
        subgraphs_2 = get_second_adj(dgl_graph, A, subgraph_size_2)
    else:
        raise NotImplementedError
    return subgraphs_1, subgraphs_2

with tqdm(total=args.epochs) as pbar:
    pbar.set_description('Training')
    for epoch in range(args.epochs):
        model.train()
        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)
        loss_1 = 0.
        loss_2 = 0.
        loss_3 = 0.
        loss_record = 0.
        total_loss = 0.
        subgraphs_1, subgraphs_2 = generate_subgraph(args, dgl_graph, A, subgraph_size_1, subgraph_size_2)
        for batch_idx in range(batch_num):
            optimiser.zero_grad()
            is_final_batch = (batch_idx == (batch_num - 1))
            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]
            cur_batch_size = len(idx)
            lbl = torch.unsqueeze(
                torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio))), 1)
            ba = []
            ba_2 = []
            bf = []
            bf_2 = []
            subgraph_idx = []
            subgraph_idx_2 = []
            Z_l = torch.full((cur_batch_size,), 1.)
            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size_1))
            added_adj_zero_row_2 = torch.zeros((cur_batch_size, 1, subgraph_size_2))
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size_1 + 1, 1))
            added_adj_zero_col_2 = torch.zeros((cur_batch_size, subgraph_size_2 + 1, 1))
            added_adj_zero_col[:, -1, :] = 1.
            added_adj_zero_col_2[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))
            if torch.cuda.is_available():
                Z_l = Z_l.to(device)
                lbl = lbl.to(device)
                added_adj_zero_row = added_adj_zero_row.to(device)
                added_adj_zero_col = added_adj_zero_col.to(device)
                added_adj_zero_row_2 = added_adj_zero_row_2.to(device)
                added_adj_zero_col_2 = added_adj_zero_col_2.to(device)
                added_feat_zero_row = added_feat_zero_row.to(device)
            for i in idx:
                cur_adj = adj[:, subgraphs_1[i], :][:, :, subgraphs_1[i]]
                cur_adj_2 = adj[:, subgraphs_2[i], :][:, :, subgraphs_2[i]]
                cur_feat = features[:, subgraphs_1[i], :]
                cur_feat_2 = features[:, subgraphs_2[i], :]
                ba.append(cur_adj)
                ba_2.append(cur_adj_2)
                bf.append(cur_feat)
                bf_2.append(cur_feat_2)
                subgraph_idx.append(subgraphs_1[i])
                subgraph_idx_2.append(subgraphs_2[i])
            ba = torch.cat(ba)
            ba_2 = torch.cat(ba_2)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)
            
            ba_2 = torch.cat((ba_2, added_adj_zero_row_2), dim=1)
            ba_2 = torch.cat((ba_2, added_adj_zero_col_2), dim=2)

            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)
            bf_2 = torch.cat(bf_2)
            bf_2 = torch.cat((bf_2[:, :-1, :], added_feat_zero_row, bf_2[:, -1:, :]), dim=1)
            
            subgraph_idx = torch.Tensor(subgraph_idx)
            subgraph_idx_2 = torch.Tensor(subgraph_idx_2)
            subgraph_idx = subgraph_idx.int()
            subgraph_idx_2 = subgraph_idx_2.int()
            if torch.cuda.is_available():
                subgraph_idx = subgraph_idx.to(device)
                subgraph_idx_2 = subgraph_idx_2.to(device)
            #/---------------------MODEL-----------------------/#
            # 传播前：ns/nn/ss→传播后：ns/nn/ss
            disc_ns, disc_nn, disc_ss, disc_p_ns, disc_p_nn, disc_p_ss = \
                model(bf, ba, bf_2, ba_2, train=True)

            loss_ns = b_xent(disc_ns, lbl)
            loss_nn = b_xent(disc_nn, lbl)
            loss_ss = b_xent(disc_ss, lbl)
            loss_p_ns = b_xent(disc_p_ns, lbl)
            loss_p_nn = b_xent(disc_p_nn, lbl)
            loss_p_ss = b_xent(disc_p_ss, lbl)
            
            loss = args.alpha * (loss_ns+args.beta*loss_nn+args.gama*loss_ss) + \
                (1 - args.alpha)*(loss_p_ns+args.beta*loss_p_nn+args.gama*loss_p_ss)
            
            
            loss.backward()
            optimiser.step()
            loss = loss.detach().cpu().numpy()
            if not is_final_batch:
                total_loss += loss
        mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes
        if mean_loss < best:
            best = mean_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'pkl/run7' + args.dataset +args.cuda+ '.pkl')
        else:
            cnt_wait += 1
        pbar.set_postfix(loss=mean_loss)
        pbar.update(1)
path = 'pkl/run7' + args.dataset+args.cuda + '.pkl'
model.load_state_dict(torch.load(path))
multi_round_attr_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))
# batch批次个元素
nodes_embed1 = torch.zeros([nb_nodes, args.embedding_dim], dtype=torch.float).cuda()
nodes_embed2 = torch.zeros([nb_nodes, args.embedding_dim], dtype=torch.float).cuda()

with tqdm(total=args.auc_test_rounds) as pbar_test:
    pbar_test.set_description('Testing')
    for round in range(args.auc_test_rounds):
        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)
        subgraphs_1, subgraphs_2 = generate_subgraph(args, dgl_graph, A,subgraph_size_1, subgraph_size_2)
        for batch_idx in range(batch_num):
            optimiser.zero_grad()
            is_final_batch = (batch_idx == (batch_num - 1))
            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]
            cur_batch_size = len(idx)
            ba = []
            bf = []
            bf_2 = []
            ba_2 = []
            subgraph_idx = []
            subgraph_idx_2 = []
            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size_1))
            added_adj_zero_row_2 = torch.zeros((cur_batch_size, 1, subgraph_size_2))
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size_1 + 1, 1))
            added_adj_zero_col_2 = torch.zeros((cur_batch_size, subgraph_size_2 + 1, 1))
            added_adj_zero_col[:, -1, :] = 1.
            added_adj_zero_col_2[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))
            if torch.cuda.is_available():
                added_adj_zero_row = added_adj_zero_row.to(device)
                added_adj_zero_row_2 = added_adj_zero_row_2.to(device)
                added_adj_zero_col = added_adj_zero_col.to(device)
                added_adj_zero_col_2 = added_adj_zero_col_2.to(device)
                added_feat_zero_row = added_feat_zero_row.to(device)
            for i in idx:
                cur_adj = adj[:, subgraphs_1[i], :][:, :, subgraphs_1[i]]
                cur_adj2 = adj[:, subgraphs_2[i], :][:, :, subgraphs_2[i]]
                cur_feat = features[:, subgraphs_1[i], :]
                raw_f = raw_feature[:, subgraphs_1[i], :]
                cur_feat_2 = features[:, subgraphs_2[i], :]
                raw_f_2 = raw_feature[:, subgraphs_2[i], :]
                ba.append(cur_adj)
                ba_2.append(cur_adj2)
                bf.append(cur_feat)
                bf_2.append(cur_feat_2)
                subgraph_idx.append(subgraphs_1[i])
                subgraph_idx_2.append(subgraphs_2[i])
            ba = torch.cat(ba)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)
            ba_2 = torch.cat(ba_2)
            ba_2 = torch.cat((ba_2, added_adj_zero_row_2), dim=1)
            ba_2 = torch.cat((ba_2, added_adj_zero_col_2), dim=2)
            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)
            bf_2 = torch.cat(bf_2)
            bf_2 = torch.cat((bf_2[:, :-1, :], added_feat_zero_row, bf_2[:, -1:, :]), dim=1)
            subgraph_idx = torch.Tensor(subgraph_idx)
            subgraph_idx_2 = torch.Tensor(subgraph_idx_2)
            subgraph_idx = subgraph_idx.int()
            subgraph_idx_2 = subgraph_idx_2.int()
            if torch.cuda.is_available():
                subgraph_idx = subgraph_idx.to(device)
                subgraph_idx_2 = subgraph_idx_2.to(device)
            # /---------------------MODEL-----------------------/#
            with torch.no_grad():
                logits_ns, logits_p_ns, sim_nn, sim_ss, sim_nn_p, sim_ss_p \
                    = model(bf, ba, bf_2, ba_2, train=False)
                logits_ns = torch.squeeze(logits_ns)
                logits_ns = torch.sigmoid(logits_ns)
    
                logits_p_ns = torch.squeeze(logits_p_ns)
                logits_p_ns = torch.sigmoid(logits_p_ns)
    
            scaler1 = MinMaxScaler()
            scaler2 = MinMaxScaler()
            scaler3 = MinMaxScaler()
            scaler4 = MinMaxScaler()
            scaler5 = MinMaxScaler()
            scaler6 = MinMaxScaler()
            #score_l = - (logits_ns[:cur_batch_size] - logits_ns[cur_batch_size:]).cpu().numpy()
            score_l = - logits_ns.cpu().numpy()
            score_2 = - sim_nn.cpu().numpy()
            score_3 = - sim_ss.cpu().numpy()
            score_4 = - logits_p_ns.cpu().numpy()
            score_5 = - sim_nn_p.cpu().numpy()
            score_6 = - sim_ss_p.cpu().numpy()
            #score_g = - (logits_2[:cur_batch_size] - logits_2[cur_batch_size:]).cpu().numpy()
            #score_otx = (- (sim_pos_node1 + sim_pos_node2) / 2).cpu().numpy()
            #score_g = logits_2[cur_batch_size:].cpu().numpy()
            
            #nomalize
            ano_score_ns = scaler1.fit_transform(score_l.reshape(-1, 1)).reshape(-1)
            ano_score_nn = scaler2.fit_transform(score_2.reshape(-1, 1)).reshape(-1)
            ano_score_ss = scaler3.fit_transform(score_3.reshape(-1, 1)).reshape(-1)
            ano_score_p_ns = scaler4.fit_transform(score_4.reshape(-1, 1)).reshape(-1)
            ano_score_p_nn = scaler5.fit_transform(score_5.reshape(-1, 1)).reshape(-1)
            ano_score_p_ss = scaler6.fit_transform(score_6.reshape(-1, 1)).reshape(-1)
            
            ano_scores = args.alpha * (ano_score_ns+args.beta*ano_score_nn+args.gama*ano_score_ss) + \
                (1 - args.alpha)*(ano_score_p_ns+args.beta*ano_score_p_nn+args.gama*ano_score_p_ss) # anomaly score have ot(pos)
            
            multi_round_attr_ano_score[round, idx] = ano_scores
        pbar_test.update(1)
attr_ano_score_final = np.mean(multi_round_attr_ano_score, axis=0)
attr_scaler = MinMaxScaler()
attr_ano_score_final = attr_scaler.fit_transform(attr_ano_score_final.reshape(-1, 1)).reshape(-1)

best_auc = roc_auc_score(ano_label, attr_ano_score_final)

print('AUC:{:.4f}'.format(best_auc))
#print(f1_score(ano_label, attr_ano_score_final))
    
  # ano_score_final = np.mean(multi_round_ano_score, axis=0)
# auc = roc_auc_score(ano_label, ano_score_final)
# print()
# print('AUC:{:.4f}'.format(auc))
# mat_path = './results'
# os.makedirs(mat_path, exist_ok=True)
# strAUC = str(int(auc*10000))
# sio.savemat(mat_path + '/score_' + args.dataset + 'node-node_pair-TopologyAUC' + strAUC + '.mat', {'a': auc})  

