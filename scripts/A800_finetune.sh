export CUDA_VISIBLE_DEVICES=0,1,2
python inference.py  --train  --load_ckpt  --batch_size 4 --lr_base 0.00001 --epoch 100 --log_dir results/finetune_exp/lrb{0.00001}-epo{100}-bs{4}-gpu{3}-norm{wo}-ckpt{w}-loss{all}-lora{w}-lrd{w}-bf16{w}-data{all} --bf16   --lora --data_path ../emu_data_subtask/ --mlt_emu 
