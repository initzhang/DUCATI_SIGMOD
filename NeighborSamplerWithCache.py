import dgl

class NeighborSamplerWithCache(dgl.dataloading.Sampler):
    def __init__(self, cached_indptr, cached_indices, fanouts):
        super().__init__()
        self.cached_indptr= cached_indptr
        self.cached_indices= cached_indices
        self.fanouts = fanouts

    def sample(self, g, seed_nodes):
        output_nodes = seed_nodes
        subgs = []
        for fanout in reversed(self.fanouts):
            # customized API for sampling with gpu cache
            sg = dgl.sampling.sample_neighbors_with_cache(
                    g, self.cached_indptr, self.cached_indices, seed_nodes, fanout)
            sg = dgl.to_block(sg, seed_nodes)
            seed_nodes = sg.srcdata[dgl.NID]
            subgs.insert(0, sg)
            input_nodes = seed_nodes
        return input_nodes, output_nodes, subgs
