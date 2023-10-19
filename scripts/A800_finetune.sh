export CUDA_VISIBLE_DEVICES=0,1
python inference.py  --train  --load_ckpt  --batch_size 4 --lr_base 0.00001 --epoch 100 --log_dir results/finetune_exp/Exp1-2 --bf16   --lora --data_path ../emu_data_subtask/ --mlt_emu 
