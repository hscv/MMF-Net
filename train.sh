CUDA_VISIBLE_DEVICES=1 nohup python -u main/train.py --config experiments/stmtrack/train/got10k/stmtrack-googlenet-trn.yaml --gpu_num 1 > nohup.train.log 2>&1 &
