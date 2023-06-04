import dgl
import time
import torch
import numpy as np

from mylog import get_logger
mlog = get_logger()

from NeighborSamplerWithCache import NeighborSamplerWithCache
from CacheConstructor import form_adj_cache, form_nfeat_cache, separate_features_idx

def allocate_dual_cache(args, graph, all_data, counts):
    dual_tic = time.time()
    assert args.total_budget >= 0
    mlog(f"total cache budget: {args.total_budget}GB")
    total_cache_bytes = args.total_budget*(1024**3)
    assert graph.idtype == torch.int64
    graph_bytes = (graph.num_edges()+graph.num_nodes()+1) * 8.
    SINGLE_LINE_SIZE = 0
    for data in all_data:
        SINGLE_LINE_SIZE += (data.shape[1] if len(data.shape) > 1 else 1) * data[torch.arange(1)].element_size()
    mlog(f"total adj size: {graph_bytes/(1024**3):.3f}GB, total nfeat size: {SINGLE_LINE_SIZE*graph.num_nodes()/(1024**3):.3f}GB")

    adj_counts, nfeat_counts = counts
    adj_size_array = (graph.in_degrees() + 1) * 8
    adj_density_array = (adj_counts / adj_counts.sum()) * args.adj_slope / (graph.in_degrees() + 1) / 8
    adj_id_array = -torch.arange(graph.num_nodes())-1
    nfeat_size_array = torch.ones(graph.num_nodes()) * SINGLE_LINE_SIZE
    nfeat_density_array = (nfeat_counts / nfeat_counts.sum()) * args.nfeat_slope / SINGLE_LINE_SIZE
    nfeat_id_array = torch.arange(graph.num_nodes())
    mlog("finish constructing density and size array")

    adj_info = torch.stack((adj_density_array, adj_size_array, adj_id_array), dim=-1) # in shape Nx3
    nfeat_info = torch.stack((nfeat_density_array, nfeat_size_array, nfeat_id_array), dim=-1) # in shape Nx3
    whole_info = torch.cat((adj_info, nfeat_info)) # in shape 2Nx3
    whole_info = whole_info[whole_info[:, 0].argsort(descending=True)] # sort according to the density
    accum_size = torch.cumsum(whole_info[:,1], 0)
    separate_point = torch.searchsorted(accum_size, total_cache_bytes)
    mlog(f"find the separate point {separate_point}")

    # find the ratio in dual cache
    all_cached_ids = whole_info[:separate_point, 2].long()
    cached_adj_ids = -all_cached_ids[all_cached_ids < 0] - 1
    cached_nfeat_ids = all_cached_ids[all_cached_ids >= 0]
    adj_bytes = adj_size_array[cached_adj_ids].sum()
    nfeat_bytes = nfeat_size_array[cached_nfeat_ids].sum()
    args.nfeat_budget = nfeat_bytes/(1024**3)
    args.adj_budget = adj_bytes/(1024**3)
    mlog(f"nfeat entries: {cached_nfeat_ids.shape[0]}, adj entries: {cached_adj_ids.shape[0]}")
    mlog(f"nfeat size: {args.nfeat_budget:.3f} GB, adj size: {args.adj_budget:.3f} GB")

    # prepare adj cache tensor
    # note that adj has already been sorted by density, so cached_adj_ids = torch.arange(cached_adj_ids.shape[0])
    indptr, indices, _ = graph.adj_sparse(fmt='csc')
    adj_cache_size = cached_adj_ids.shape[0]
    cached_indptr = indptr[:adj_cache_size+1].cuda()
    cached_indices = indices[:indptr[adj_cache_size]].cuda()

    # prepare nfeat cache
    gpu_flag = torch.zeros(graph.num_nodes(), dtype=torch.bool)
    gpu_flag[cached_nfeat_ids] = True
    gpu_flag = gpu_flag.cuda()
    all_cache = [data[cached_nfeat_ids].to('cuda') for data in all_data]
    gpu_map = torch.zeros(graph.num_nodes(), dtype=torch.int32).fill_(-1)
    gpu_map[cached_nfeat_ids] = torch.arange(cached_nfeat_ids.shape[0]).int()
    gpu_map = gpu_map.cuda()
    mlog(f"dual cache allocation done, time_elapsed: {time.time()-dual_tic:.2f}s")

    return cached_indptr, cached_indices, gpu_flag, gpu_map, all_cache


def get_slope(args, graph, counts, seeds_list, all_data):
    mlog(f"start profiling and calculating slope")
    slope_tic = time.time()
    fanouts = [int(x) for x in args.fanouts.split(",")]
    adj_counts, nfeat_counts = counts
    adj_ratio_steps = [x/10 for x in range(0,10,1)]
    nfeat_ratio_steps = [x/10 for x in range(0,10,1)]

    ###################
    ### get adj slope
    ###################
    adj_nnz = (adj_counts > 0).sum()
    indptr, indices, _ = graph.adj_sparse(fmt='csc')
    adj_stats = []
    for cached_nnz_adj_ratio in adj_ratio_steps:
        cache_size = int(cached_nnz_adj_ratio * adj_nnz)
        cur_accum_hits = adj_counts[:cache_size].sum()/adj_counts.sum()
        if cache_size == 0:
            sampler = dgl.dataloading.NeighborSampler(fanouts)
        else:
            try:
                cached_indptr = indptr[:cache_size+1].cuda()
                cached_indices = indices[:indptr[cache_size]].cuda()
                sampler = NeighborSamplerWithCache(cached_indptr, cached_indices, fanouts)
            except:
                mlog('early stop at adj due to OOM')
                break
        # warmup
        for seeds in seeds_list[-10:]:
            input_nodes, output_nodes, blocks = sampler.sample(graph, seeds)
        # measure
        torch.cuda.synchronize()
        tic = time.time()
        for seeds in seeds_list[:args.pre_batches]:
            input_nodes, output_nodes, blocks = sampler.sample(graph, seeds)
        torch.cuda.synchronize()
        avg_duration = 1000*(time.time() - tic)/args.pre_batches
        adj_stats.append((cur_accum_hits.item(), avg_duration))

    input_nodes_list = []
    for seeds in seeds_list[:args.pre_batches]:
        input_nodes_list.append(sampler.sample(graph, seeds)[0].cpu())

    ###################
    ### get nfeat slope
    ###################
    def retrieve_data(cpu_data, gpu_data, idx, out_buf):
        nonlocal gpu_map, gpu_flag
        if gpu_map is None:
            cur_res = cpu_data[idx]
        else:
            gpu_mask = gpu_flag[idx]
            gpu_nids = idx[gpu_mask]
            gpu_local_nids = gpu_map[gpu_nids].long()
            cpu_nids = idx[~gpu_mask]

            cur_res = out_buf[:idx.shape[0]]
            cur_res[gpu_mask] = gpu_data[gpu_local_nids]
            cur_res[~gpu_mask] = cpu_data[cpu_nids]
        return cur_res

    def run_one_list(input_list):
        nonlocal gpu_flag, gpu_map, all_cache, all_data
        for input_nodes in input_list:
            input_nodes = input_nodes.cuda()
            cur_nfeat = retrieve_data(all_data[0], all_cache[0], input_nodes, nfeat_buf) # fetch nfeat
            cur_label = retrieve_data(all_data[1], all_cache[1], input_nodes[:args.bs], label_buf) # fetch label

    nfeat_nnz = (nfeat_counts > 0).sum()
    nfeat_stats = []
    nfeat_budget_backup = args.nfeat_budget
    for cached_nnz_nfeat_ratio in nfeat_ratio_steps: 
        cache_size = int(cached_nnz_nfeat_ratio * nfeat_nnz)
        args.nfeat_budget = cache_size * args.fake_dim * 4 / (1024**3)
        if args.nfeat_budget > args.total_budget:
            break

        nfeat_buf = label_buf = None
        if args.nfeat_budget > 0:
            estimate_max_batch = int(1.2*input_nodes_list[0].shape[0])
            nfeat_buf = torch.zeros((estimate_max_batch, args.fake_dim), dtype=torch.float).cuda()
            label_buf = torch.zeros((args.bs, ), dtype=torch.long).cuda()

        try:
            gpu_flag, gpu_map, all_cache, accum_hit = form_nfeat_cache(args, all_data, counts)
            # warmup
            run_one_list(input_nodes_list[-10:])
            # measure total
            torch.cuda.synchronize()
            tic = time.time()
            run_one_list(input_nodes_list)
            torch.cuda.synchronize()
            avg_duration = 1000*(time.time() - tic)/args.pre_batches
        except:
            mlog('early stop at nfeat due to OOM')
            break
        # measure transfer
        torch.cuda.synchronize()
        tic = time.time()
        for input_nodes in input_nodes_list:
            input_nodes = input_nodes.cuda()
        torch.cuda.synchronize()
        transfer_avgs = 1000*(time.time() - tic)/args.pre_batches
        nfeat_stats.append((accum_hit, avg_duration-transfer_avgs))

        del gpu_flag, gpu_map, all_cache, nfeat_buf, label_buf, input_nodes
        torch.cuda.empty_cache()
    args.nfeat_budget = nfeat_budget_backup

    adj_stats = np.array(adj_stats)
    nfeat_stats = np.array(nfeat_stats)
    adj_slope = -np.polyfit(adj_stats[:,0], adj_stats[:,1], 1)[0]
    nfeat_slope = -np.polyfit(nfeat_stats[:,0], nfeat_stats[:,1], 1)[0]
    assert nfeat_slope > 0
    assert adj_slope > 0
    mlog(f"finish calculating slope: adj({adj_slope:.2f}) nfeat({nfeat_slope:.2f}), time elapsed: {time.time() - slope_tic:.2f}s")

    return adj_slope, nfeat_slope

