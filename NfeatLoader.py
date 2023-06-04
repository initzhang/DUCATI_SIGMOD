class NfeatLoader(object):
    def __init__(self, cpu_data, gpu_data, gpu_map, gpu_flag):
        self.cpu_data = cpu_data
        self.gpu_data = gpu_data
        self.gpu_map = gpu_map
        self.gpu_flag = gpu_flag

    def load(self, idx, out_buf):
        gpu_mask = self.gpu_flag[idx]
        gpu_nids = idx[gpu_mask]
        gpu_local_nids = self.gpu_map[gpu_nids].long()
        cpu_nids = idx[~gpu_mask]

        cur_res = out_buf[:idx.shape[0]]
        cur_res[gpu_mask] = self.gpu_data[gpu_local_nids]
        cur_res[~gpu_mask] = self.cpu_data[cpu_nids]
        return cur_res

