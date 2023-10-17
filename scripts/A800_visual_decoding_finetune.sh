export CUDA_VISIBLE_DEVICES=0,1,2,7
python image_inference.py  --train  --load_ckpt  --batch_size 20  --lr_base 0.00005 --epoch 100 --log_dir results/visual_decoding_finetune_exp/lrb{0.00005}-epo{100}-bs{20}-norm{wo}-ckpt{w}-lora{w}-lrd{w}-cs{3.}    --lora --data_path ../emu_data_subtask/ --mlt_emu --bf16 
