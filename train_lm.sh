uv run python -m cs336_basics.train_transformer_lm \
  --train_text data/owt_train.txt \
  --val_text data/owt_valid.txt \
  --train_tokens data/owt_train_tokens.npy \
  --val_tokens data/owt_valid_tokens.npy \
  --data_dtype uint16 \
  --bpe_vocab artifacts/owt_vocab.pkl \
  --bpe_merges artifacts/owt_merges.pkl \
  --vocab_size 32000 \
  --context_length 256 \
  --num_layers 4 --d_model 512 --num_heads 16 --d_ff 1344 --rope_theta 10000.0 \
  --lr 3e-3 --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decay 0.01 \
  --warmup_iters 800 --cosine_cycle_iters 200000 --lr_min 1e-5 \
  --batch_size 32 --max_iters 400000 --grad_clip 1.0 \
  --eval_interval 1000 --eval_batches 50 --log_interval 100 \
  --device cuda:0 \
  --seed 42 \
  --wandb \
  --wandb_project lm-from-scratch \
  --wandb_run owt_train \
  --checkpoint_path ./checkpoints/checkpoint.pt --save_every 2000 \
  --force_regen

# uv run python -m cs336_basics.train_transformer_lm \
#   --train_text data/TinyStoriesV2-GPT4-train.txt \
#   --val_text data/TinyStoriesV2-GPT4-valid.txt \
#   --train_tokens data/TinyStoriesV2-GPT4-train_tokens.npy \
#   --val_tokens data/TinyStoriesV2-GPT4-valid_tokens.npy \
#   --data_dtype uint16 \
#   --bpe_vocab artifacts/TinyStoriesV2-GPT4_vocab.pkl \
#   --bpe_merges artifacts/TinyStoriesV2-GPT4_merges.pkl \
#   --vocab_size 10000 \
#   --context_length 256 \
#   --num_layers 4 --d_model 512 --num_heads 16 --d_ff 1344 --rope_theta 10000.0 \
#   --lr 3e-3 --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decay 0.01 \
#   --warmup_iters 800 --cosine_cycle_iters 200000 --lr_min 1e-5 \
#   --batch_size 32 --max_iters 400000 --grad_clip 1.0 \
#   --eval_interval 1000 --eval_batches 50 --log_interval 100 \
#   --device cuda:0 \
#   --seed 42 \
#   --wandb \
#   --wandb_project lm-from-scratch \
#   --wandb_run exp_7_2 \
#   --checkpoint_path ./checkpoints/checkpoint.pt --save_every 2000
  # --force_regen \
  # --resume_from ./checkpoint.pt