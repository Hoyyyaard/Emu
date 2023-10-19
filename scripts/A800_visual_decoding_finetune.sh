export CUDA_VISIBLE_DEVICES=0,1,2,7
python image_inference.py  --train  --load_ckpt  --batch_size 8  --emu_ckpt results/finetune_exp/Exp1-1/ckpt/finetune_20_cls0.11_reg0.10.bin --lr_base 0.0001 --epoch 100 --classify_scale 10 --log_dir results/visual_decoding_finetune_exp/Exp2-1    --lora --mlt_emu --bf16  --data_path ../emu_data_subtask/ 
