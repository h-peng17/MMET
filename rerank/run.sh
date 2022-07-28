python3 -m torch.distributed.launch --nproc_per_node 8 train.py \
    --n_gpu 8 \
    --max_seq_length 80 \
    --recall_count 64 \
    --batch_size_per_gpu 1 \
    --num_train_epochs 100 \
    --save_epoch 100 \
    --record_step 100 \
    --model v1 \
    --loss focal \
    --encoder br \
    --ckpt_dir ckpt/text \
    --data_prefix train_ \
    --tcp tcp://127.0.0.1:12341 \
    --optim adamw \

