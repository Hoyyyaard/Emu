export CUDA_VISIBLE_DEVICES=5,6,7
python inference.py --lora_finetune --train --mlt_emu --data_path /root/Projects/emu_data_subtask/  --val_data_path /root/Projects/emu_data_subtask_val/ 