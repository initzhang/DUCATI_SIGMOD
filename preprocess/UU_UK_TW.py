import dgl
import pandas
import torch

"""
get the *_coo.txt files from GNNLab & WebGraph
each line of such file is "src_id\tdst_id" representing an edge, namely the COO format
"""

for name in ['uk-union', 'uk', 'twitter']:
    print('reading csv')
    df = pandas.read_csv(f'{name}_coo.txt', sep='\t', header=None, names=['src', 'dst'])
    src = df['src'].values
    dst = df['dst'].values

    print('construct the graph')
    g = dgl.graph((src, dst))

    print('transform')
    g = g.formats('csc')
    indptr, indices, _ = g.adj_sparse(fmt='csc')
    new_graph = dgl.graph(('csc', (indptr, indices, torch.tensor([]))), num_nodes=g.num_nodes())

    print('save')
    dgl.save_graphs(f"dgl_{name}.bin", [new_graph])
