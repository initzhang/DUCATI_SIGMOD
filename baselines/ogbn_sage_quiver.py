import os
import torch
import torch.nn.functional as F
import torch.nn as tnn
import numpy as np
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected
import time

from mylog import get_logger
mlog = get_logger()

######################
# Import From Quiver
######################
import quiver
from quiver.pyg import GraphSageSampler

# occupy GPU
a = torch.rand(10,10).cuda()

name = 'ogbn-papers100M'
#name = 'ogbn-products'
assert name in ['ogbn-papers100M', 'ogbn-products']
mlog(name)

root = input() # root of dataset 
dataset = PygNodePropPredDataset(name, root)
mlog('finish load dataset')
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name=name)
data = dataset[0]

train_idx = split_idx['train']
valid_idx = split_idx['valid']

mlog(f'start to_undirected, {data.edge_index.shape}')
new_edge_index = to_undirected(data.edge_index)
mlog(f'finish to_undirected, {new_edge_index.shape}')
del data.edge_index

train_loader = torch.utils.data.DataLoader(train_idx,
                                           batch_size=1000,
                                           shuffle=True,
                                           drop_last=False)

valid_loader = NeighborSampler(new_edge_index, node_idx=valid_idx,
                             sizes=[10, 10, 10], batch_size=100,
                             shuffle=False, num_workers=0, drop_last=False)
 

mlog('finish create loader')
csr_topo = quiver.CSRTopo(new_edge_index)
mlog('finish create csr_topo')

quiver_sampler = GraphSageSampler(csr_topo, sizes=[10, 10, 10], device=0, mode='UVA')
mlog('finish create two quiver_sampler')

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
        return x.log_softmax(dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SAGE(dataset.num_features, 256, dataset.num_classes, num_layers=3)
model = model.to(device)
mlog('finish create model')

# after several trials, the largest viable cache size is 5.3G on 2080Ti (11GB)
x = quiver.Feature(rank=0, device_list=[0], device_cache_size="5.3G", cache_policy="device_replicate", csr_topo=csr_topo)
mlog('finish create quiver Feature')
feature = torch.zeros(data.x.shape)
feature[:] = data.x
x.from_cpu_tensor(feature)
mlog('finish assign quiver Feature')
del data.x

y = data.y.squeeze().to(device)
mlog('finish move y')


def train(epoch):
    model.train()

    #pbar = tqdm(total=train_idx.size(0))
    #pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0

    ######################
    # Original Pyg Code
    ######################
    # for batch_size, n_id, adjs in train_loader:
    for seeds in train_loader:
        n_id, batch_size, adjs = quiver_sampler.sample(seeds)
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        loss = F.nll_loss(out.double(), y[n_id[:batch_size]].long())
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        #pbar.update(batch_size)

    #pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)

    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()
    #pbar = tqdm(total=valid_idx.size(0))
    #pbar.set_description(f'Valid')
    outs = []
    ys = []
    #for seeds in valid_loader:
    #    n_id, batch_size, adjs = valid_sampler.sample(seeds)
    for batch_size, n_id, adjs in valid_loader:
        adjs = [adj.to(device) for adj in adjs]
        ys.append(y[n_id[:batch_size]].cpu())
        outs.append(model(x[n_id], adjs).cpu())
        #pbar.update(batch_size)
    #pbar.close()

    y_true = torch.cat(ys).unsqueeze(-1)
    y_pred = torch.cat(outs).argmax(dim=-1, keepdim=True)

    val_acc = evaluator.eval({
        'y_true': y_true,
        'y_pred': y_pred,
    })['acc']

    #y_true = y.cpu().unsqueeze(-1)
    #y_pred = out.argmax(dim=-1, keepdim=True)

    #train_acc = evaluator.eval({
    #    'y_true': y_true[split_idx['train']],
    #    'y_pred': y_pred[split_idx['train']],
    #})['acc']
    #val_acc = evaluator.eval({
    #    'y_true': y_true[split_idx['valid']],
    #    'y_pred': y_pred[split_idx['valid']],
    #})['acc']
    #test_acc = evaluator.eval({
    #    'y_true': y_true[split_idx['test']],
    #    'y_pred': y_pred[split_idx['test']],
    #})['acc']
    #return train_acc, val_acc, test_acc
    return val_acc


val_accs = []
epoch_times = []
for run in range(1, 2):
    mlog('')
    mlog(f'Run {run:02d}:')
    mlog('')

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    #best_val_acc = final_test_acc = 0
    best_val_acc = 0
    for epoch in range(1, 21):
        epoch_start = time.time()
        loss, acc = train(epoch)
        epoch_time = time.time() - epoch_start
        mlog(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}, Epoch Time: {epoch_time:.4f}s')
        epoch_times.append(epoch_time)

        #train_acc, val_acc, test_acc = test()
        val_start = time.time()
        val_acc = test()
        val_time = time.time() - val_start
        #mlog(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
        mlog(f'Val: {val_acc:.4f}, Valid Time: {val_time:.4f}s')
        val_accs.append(val_acc)

        #if val_acc > best_val_acc:
        #    best_val_acc = val_acc
        #    final_test_acc = test_acc
    #test_accs.append(final_test_acc)

#test_acc = torch.tensor(test_accs)
#mlog('============================')
#mlog(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')
mlog('============================')
mlog(val_accs)
mlog(epoch_times)
mlog('============================')
mlog(f'Peak Valid Acc: {max(val_accs)}')
mlog(f'Avg Epoch times: {np.mean(epoch_times):.2f} ± {np.std(epoch_times):.2f}s')
