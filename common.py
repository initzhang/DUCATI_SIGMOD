import dgl
import math
import torch
import random
import numpy as np

from mylog import get_logger
mlog = get_logger()

def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        dgl.seed(seed)
        dgl.random.seed(seed)
    else:
        mlog(f"warning: no cuda available, seeds may not be properly set")

def get_seeds_list(args, train_idx):
    seeds_list = []
    for _ in range(args.batches):
        idxs = torch.randint(0, train_idx.shape[0], (args.bs,))
        cur_seed = train_idx[idxs].to(train_idx.device)
        seeds_list.append(cur_seed)

    size = args.batches * args.bs * idxs.element_size() / (1024**3)
    mlog(f"get {args.batches} seeds, {size:.2f}GB on {train_idx.device}")

    return seeds_list

def fast_reorder(graph, nodes_perm):
    if isinstance(graph, tuple):
        src, dst = graph
    else:
        assert isinstance(graph, dgl.DGLHeteroGraph)
        src, dst = graph.adj_sparse(fmt='coo')
    mmap = torch.zeros(nodes_perm.shape[0], dtype=torch.int64)
    mmap[nodes_perm] = torch.arange(nodes_perm.shape[0])
    src = mmap[src]
    dst = mmap[dst]
    new_graph = dgl.graph((src, dst))
    del src, dst
    return new_graph
