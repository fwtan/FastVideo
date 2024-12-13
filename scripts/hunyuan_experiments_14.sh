export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_DIR="$HOME"
export WANDB_MODE=online
export WANDB_API_KEY=4f6de3765d6464f43e0506ec7d785641af645e73
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_PROTO=simple

DATA_DIR=/data
IP=10.4.139.86

torchrun --nnodes 2 --nproc_per_node 8\
    --node_rank=0 \
    --rdzv_id=456 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$IP:29500 \
    fastvideo/distill_adv.py\
    --seed 42\
    --pretrained_model_name_or_path $DATA_DIR/hunyuan\
    --dit_model_name_or_path $DATA_DIR/hunyuan/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt\
    --model_type "hunyuan" \
    --cache_dir "$DATA_DIR/.cache"\
    --data_json_path "$DATA_DIR/Hunyuan-Distill-Data/videos2caption.json"\
    --validation_prompt_dir "$DATA_DIR/Hunyuan-Distill-Data/validation"\
    --gradient_checkpointing\
    --train_batch_size=1\
    --num_latent_t 24\
    --sp_size 1\
    --train_sp_batch_size 1\
    --dataloader_num_workers 4\
    --gradient_accumulation_steps=2\
    --max_train_steps=640\
    --learning_rate=1e-6\
    --mixed_precision="bf16"\
    --checkpointing_steps=64\
    --validation_steps 64\
    --validation_sampling_steps "2,4,8" \
    --checkpoints_total_limit 3\
    --allow_tf32\
    --ema_start_step 0\
    --cfg 0.0\
    --log_validation\
    --output_dir="$DATA_DIR/outputs/hy_phase1_shift17_bs_32_adv"\
    --tracker_project_name Hunyuan_Distill \
    --num_frames  93 \
    --shift 17 \
    --validation_guidance_scale "1.0" \
    --num_euler_timesteps 50 \
    --multi_phased_distill_schedule "4000-1" \
    --not_apply_cfg_solver 