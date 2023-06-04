from ogb.nodeproppred import DglNodePropPredDataset
import torch as th
import time
import dgl


def load_ogb(name, root='.'):
    tic = time.time()
    print(f'load {name}')
    data = DglNodePropPredDataset(root=root, name=name)
    print(f'finish loading {name}')
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata['label'] = labels
    in_feats = graph.ndata['feat'].shape[1]
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    print(f"finish loading {name}, time elapsed: {time.time() - tic:.2f}s")
    return graph, num_labels

name = 'ogbn-papers100M'
graph, _ = load_ogb(name=name, root='/home/data/USERNAME/datasets')
mask = graph.ndata.pop('train_mask')
indptr, indices, _ = graph.adj_sparse(fmt='csc')
new_graph = dgl.graph(('csc', (indptr, indices, th.tensor([]))), num_nodes=graph.num_nodes())
new_graph.ndata['train_mask'] = mask
dgl.save_graphs(f'dgl_{name}.bin', [new_graph])
