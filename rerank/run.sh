# for ckpt in {1..19..2}
# do
#     echo $ckpt
#     CUDA_VISIBLE_DEVICES=$1 python3 inferv2.py \
#         --max_seq_length 64 \
#         --recall_count 64 \
#         --batch_size_per_gpu 1 \
#         --num_train_epochs 10 \
#         --save_epoch 10 \
#         --record_step 100 \
#         --model biencoder \
#         --loss focal \
#         --encoder br \
#         --data_prefix dev_ \
#         --ckpt ckpt/text_biencoder/ckpt_$ckpt.pth 
# done 

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


# CUDA_VISIBLE_DEVICES=$1 python3 inferv2.py \
#     --max_seq_length 64 \
#     --recall_count 64 \
#     --batch_size_per_gpu 1 \
#     --num_train_epochs 10 \
#     --save_epoch 10 \
#     --record_step 100 \
#     --model biencoder \
#     --loss focal \
#     --encoder br \
#     --data_prefix dev_ \
#     --expert_count 3 \
#     --image_expert_count 1 \
#     --ckpt ckpt/text_biencoder/ckpt_17.pth \
#     --ckpt None \
    
# CUDA_VISIBLE_DEVICES=$1 python3 train.py \
#     --n_gpu 1 \
#     --max_seq_length 77 \
#     --recall_count 64 \
#     --batch_size_per_gpu 1 \
#     --num_train_epochs 20 \
#     --save_epoch 10 \
#     --model clip \
#     --data_prefix train_ 

# CUDA_VISIBLE_DEVICES=$1 python3 inferv2.py \
#     --max_seq_length 32 \
#     --recall_count 64 \
#     --batch_size_per_gpu 1 \
#     --num_train_epochs 10 \
#     --save_epoch 10 \
#     --model mmt \
#     --ckpt ckpt/mmt/ckpt_9.pth \
#     --data_prefix test_ 