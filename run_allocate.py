"""
use this script to validate the dual-cache allocation results
"""
import dgl
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mylog import get_logger
mlog = get_logger()

import DUCATI
from model import SAGE
from load_graph import load_dc_realtime_process
from common import set_random_seeds, get_seeds_list

def entry(args, graph, all_data, seeds_list, counts):
    fanouts = [int(x) for x in args.fanouts.split(",")]

    # dual cache allocator
    adj_slope, nfeat_slope = DUCATI.DualCacheAllocator.get_slope(args, graph, counts, seeds_list, all_data)
    args.adj_slope = adj_slope
    args.nfeat_slope = nfeat_slope
    cached_indptr, cached_indices, gpu_flag, gpu_map, all_cache = DUCATI.DualCacheAllocator.allocate_dual_cache(args, graph, all_data, counts)

    mlog(f"current allocation plan: {args.adj_budget:.3f}GB adj cache & {args.nfeat_budget:.3f}GB nfeat cache")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset params
    parser.add_argument("--dataset", type=str, choices=['ogbn-papers100M', 'uk', 'uk-union', 'twitter'],
                        default='ogbn-papers100M')
    parser.add_argument("--pre-epochs", type=int, default=2) # PreSC params

    # running params
    parser.add_argument("--nfeat-budget", type=float, default=0) # in GB
    parser.add_argument("--adj-budget", type=float, default=0) # in GB
    parser.add_argument("--bs", type=int, default=8000)
    parser.add_argument("--fanouts", type=str, default='15,15,15')
    parser.add_argument("--batches", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--fake-dim", type=int, default=100)

    # dual cache allocator params
    parser.add_argument("--total-budget", type=float, default=1)
    parser.add_argument("--pre-batches", type=int, default=100)
    parser.add_argument("--nfeat-slope", type=float, default=1)
    parser.add_argument("--adj-slope", type=float, default=1)

    args = parser.parse_args()
    mlog(args)
    set_random_seeds(0)

    graph, n_classes = load_dc_realtime_process(args)
    args.n_classes = n_classes
    graph, all_data, train_idx, counts = DUCATI.CacheConstructor.separate_features_idx(args, graph)
    train_idx = train_idx.cuda()
    graph.pin_memory_()
    mlog(graph)

    # get seeds candidate
    seeds_list = get_seeds_list(args, train_idx)
    del train_idx

    entry(args, graph, all_data, seeds_list, counts)
