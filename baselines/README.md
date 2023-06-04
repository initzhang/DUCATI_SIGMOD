here presents some details about the additional comparisons in Section 5.6 and Section 5.7

Quiver: 
* prepare environment as in https://github.com/quiver-team/torch-quiver
* modified training script is `ogbn_sage_quiver.py`

MariusGNN: 
* prepare environment as in https://github.com/marius-team/marius/tree/eurosys_2023_artifact
* training config file is `config_marius.yaml`

Sancus: 
* prepare environment as in https://github.com/chenzhao/light-dist-gnn 
* align hyperparameters like fanouts and run sancus

By default we use 8 CPU threads for single-gpu training by setting `OMP_NUM_THREADS=8`.
