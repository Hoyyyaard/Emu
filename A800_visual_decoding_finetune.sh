export CUDA_VISIBLE_DEVICES=0,1,2,3
python inference.py  --train  --load_ckpt  --batch_size 4 --lr_base 0.0001 --epoch 100 --log_dir results/visual_decoding_finetune_exp/lrb{0.0001}-epo{100}-bs{4}-norm{wo}-ckpt{w}-lora{w}-lrd{w}    --lora --data_path ../emu_data_subtask/ --mlt_emu 
