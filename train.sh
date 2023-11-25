python prepare_dataset_spacenet.py
python prepare_dataset_alabama.py
python -m torch.distributed.launch --nproc_per_node=2 pretrain.py 2>&1 | tee logs/pretrain_convnextv2_base_256_e04_00.txt
# python -m torch.distributed.launch --nproc_per_node=2 train.py 2>&1 | tee logs/train_convnextv2_base_256_e04_full.txt
