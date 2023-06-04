import dgl
import time
import json
import os.path
import torch
import torch as th
import numpy as np
from dgl import backend as F1
from dgl.convert import from_scipy
from dgl.data.utils import generate_mask_tensor
from sklearn.preprocessing import StandardScaler
from scipy import sparse as sp

from common import set_random_seeds, fast_reorder
from mylog import get_logger
mlog = get_logger()

root_dir = "./preprocess"

def load_dc_raw(args, coo=False):
    assert args.dataset in ['ogbn-papers100M', 'uk', 'uk-union', 'twitter']
    mlog(f"loading raw dataset of {args.dataset}")
    tic = time.time()
    ds = dgl.load_graphs(f'{root_dir}/dgl_{args.dataset}.bin')[0][0]
    if coo:
        src, dst = ds.adj_sparse(fmt='coo')
    ds = ds.formats(['csc'])
    mlog(f'finish loading raw dataset, time elapsed: {time.time() - tic:.2f}s')
    if args.dataset == 'ogbn-papers100M':
        n_classes = 172
    else:
        n_classes = 100 # other fake ds
    if coo:
        return ds, src, dst, n_classes
    return ds, n_classes

def load_dc_raw_with_counts(args):
    # first load raw graph
    graph, n_classes = load_dc_raw(args, coo=False)
    # then perform sampling with UVA
    tic = time.time()
    if "ogbn" in args.dataset:
        train_idx = torch.nonzero(graph.ndata.pop("train_mask")).reshape(-1)
    else:
        num_train_nodes = int(graph.num_nodes() * 0.01)
        set_random_seeds(1)
        log_degs = torch.log(1+graph.in_degrees())
        probs = (log_degs / log_degs.sum()).numpy()
        train_idx = torch.from_numpy(np.random.choice(
            graph.num_nodes(), size=num_train_nodes, replace=False, p=probs)).long()

    graph.ndata.clear()
    graph.edata.clear()
    graph.pin_memory_()
    train_idx = train_idx.cuda()
    adj_counts, nfeat_counts = generate_stats(args, graph, train_idx)

    # prepare other ndata
    train_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
    train_mask[train_idx] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['adj_counts'] = adj_counts
    graph.ndata['nfeat_counts'] = nfeat_counts

    mlog(f'finish preparing dataset with counts, time elapsed: {time.time() - tic:.2f}s')
    del train_idx
    torch.cuda.empty_cache()
    return graph, n_classes


def load_dc_realtime_process(args):
    # first load raw graph
    graph, src, dst, n_classes = load_dc_raw(args, coo=True)
    num_nodes = graph.num_nodes()

    # then perform sampling with UVA
    tic = time.time()
    if "ogbn" in args.dataset:
        train_idx = torch.nonzero(graph.ndata.pop("train_mask")).reshape(-1)
    else:
        num_train_nodes = int(num_nodes * 0.01)
        set_random_seeds(1)
        train_idx = torch.randperm(num_nodes)[:num_train_nodes]
    graph.ndata.clear()
    graph.edata.clear()
    graph.pin_memory_()
    train_idx = train_idx.cuda()
    adj_counts, nfeat_counts = generate_stats(args, graph, train_idx)
    graph.unpin_memory_()
    train_idx = train_idx.cpu()
    torch.cuda.empty_cache()

    # reorder graph
    degs = graph.in_degrees() + 1
    priority = adj_counts/degs
    adj_order = priority.argsort(descending=True)
    graph = fast_reorder((src, dst), adj_order)
    del src, dst
    indptr, indices, _ = graph.adj_sparse(fmt='csc')
    del graph
    new_graph = dgl.graph(('csc', (indptr, indices, torch.tensor([]))), num_nodes=num_nodes)

    # prepare other ndata, reorder accordingly and save ndata
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    new_graph.ndata['train_mask'] = train_mask[adj_order]
    new_graph.ndata['adj_counts'] = adj_counts[adj_order]
    new_graph.ndata['nfeat_counts'] = nfeat_counts[adj_order]

    mlog(f'finish preprocessing, time elapsed: {time.time() - tic:.2f}s')
    return new_graph, n_classes


def my_iter(args, train_idx):
    pm = torch.randperm(train_idx.shape[0]).to(train_idx.device)
    local_train_idx = train_idx[pm]
    length = train_idx.shape[0] // args.bs
    for i in range(length):
        st = i * args.bs
        ed = (i+1) * args.bs
        yield local_train_idx[st:ed]

def generate_stats(args, graph, train_idx):
    #mlog("start calculate counts")
    fanouts = [int(x) for x in args.fanouts.split(",")]
    sampler = dgl.dataloading.NeighborSampler(fanouts)
    nfeat_counts = torch.zeros(graph.num_nodes()).cuda()
    adj_counts = torch.zeros(graph.num_nodes()).cuda()
    tic = time.time()
    for _ in range(args.pre_epochs):
        it = my_iter(args, train_idx)
        for seeds in it:
            input_nodes, output_nodes, blocks = sampler.sample(graph, seeds)
            # for nfeat, each iteration we only need to prepare the input layer
            nfeat_counts[input_nodes] += 1
            # for adj, each iteration we need to access each block's dst nodes
            for block in blocks:
                dst_num = block.dstnodes().shape[0]
                cur_touched_adj = block.ndata[dgl.NID]['_N'][:dst_num]
                adj_counts[cur_touched_adj] += 1
    #mlog(f"pre-sampling {args.pre_epochs} epochs time: {time.time()-tic:.3f}s")
    #mlog(f"adj counts' min, max, mean, nnz ratio: {adj_counts.min()}, {adj_counts.max()}, {adj_counts.mean():.2f}, {(adj_counts>0).sum()/adj_counts.shape[0]:.2f}")
    #mlog(f"nfeat counts' min, max, mean, nnz ratio: {nfeat_counts.min()}, {nfeat_counts.max()}, {nfeat_counts.mean():.2f}, {(nfeat_counts>0).sum()/nfeat_counts.shape[0]:.2f}")
    adj_counts = adj_counts.cpu()
    nfeat_counts = nfeat_counts.cpu()
    return adj_counts, nfeat_counts

if __name__ == '__main__':
    #g, _ = load_diffusion('ogbn-products')
    #mlog(f"{g.num_nodes()}, {g.num_edges()}")
    pass
