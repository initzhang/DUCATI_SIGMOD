import dgl
import time
import torch

from common import set_random_seeds, get_seeds_list
from mylog import get_logger
mlog = get_logger()

def form_nfeat_cache(args, all_data, counts):
    if args.nfeat_budget == 0:
        return None, None, [None]*len(all_data), 0

    # get probs and order
    nfeat_counts = counts[1]
    nfeat_probs = nfeat_counts / nfeat_counts.sum()
    nfeat_probs, nfeat_order = nfeat_probs.sort(descending=True)

    # calculate current cache
    SINGLE_LINE_SIZE = 0
    for data in all_data:
        SINGLE_LINE_SIZE += (data.shape[1] if len(data.shape) > 1 else 1) * data[torch.arange(1)].element_size()
    
    cache_nums = int(args.nfeat_budget * (1024**3) / SINGLE_LINE_SIZE)
    cache_nids = nfeat_order[:cache_nums]
    accum_hit = nfeat_probs[:cache_nums].sum().item()

    # prepare flag
    gpu_flag = torch.zeros(all_data[0].shape[0], dtype=torch.bool)
    gpu_flag[cache_nids] = True
    gpu_flag = gpu_flag.cuda()

    # prepare cache
    all_cache = [data[cache_nids].to('cuda') for data in all_data]

    # prepare map in GPU
    # for gpu feature retrieve, input -(gpu_flag)-> gpu_mask --> gpu_nids -(gpu_map)-> gpu_local_id -> features
    gpu_map = torch.zeros(nfeat_probs.shape[0], dtype=torch.int32).fill_(-1)
    gpu_map[cache_nids] = torch.arange(cache_nids.shape[0]).int()
    gpu_map = gpu_map.cuda()

    return gpu_flag, gpu_map, all_cache, accum_hit


def form_adj_cache(args, graph, counts):
    # given cache budget (in GB), derive the number of adj lists to be saved
    cache_bytes = args.adj_budget*(1024**3)
    if graph.idtype == torch.int64:
        cache_elements = cache_bytes // 8
        graph_bytes = (graph.num_edges()+graph.num_nodes()+1) * 8.
    else:
        cache_elements = cache_bytes // 4
        graph_bytes = (graph.num_edges()+graph.num_nodes()+1) * 4.

    # search break point
    indptr, indices, _ = graph.adj_sparse(fmt='csc')
    acc_size = indptr[1:] + torch.arange(1,graph.num_nodes()+1) + 1 # accumulated cache size in theory
    cache_size = torch.searchsorted(acc_size, cache_elements).item()

    # prepare cache tensor
    cached_indptr = indptr[:cache_size+1].cuda()
    cached_indices = indices[:indptr[cache_size]].cuda()

    # calculate theoretical gains
    adj_counts = counts[0]
    adj_probs = adj_counts / adj_counts.sum()
    accum_hit = adj_probs[:cache_size].sum()

    return cached_indptr, cached_indices

def separate_features_idx(args, graph):
    separate_tic = time.time()
    train_idx = torch.nonzero(graph.ndata.pop("train_mask")).reshape(-1)
    adj_counts = graph.ndata.pop('adj_counts')
    nfeat_counts = graph.ndata.pop('nfeat_counts')

    # cleanup
    graph.ndata.clear()
    graph.edata.clear()

    # we prepare fake input for all datasets
    fake_nfeat = dgl.contrib.UnifiedTensor(torch.rand((graph.num_nodes(), args.fake_dim), dtype=torch.float), device='cuda')
    fake_label = dgl.contrib.UnifiedTensor(torch.randint(args.n_classes, (graph.num_nodes(), ), dtype=torch.long), device='cuda')

    mlog(f'finish generating random features with dim={args.fake_dim}, time elapsed: {time.time()-separate_tic:.2f}s')
    return graph, [fake_nfeat, fake_label], train_idx, [adj_counts, nfeat_counts]


