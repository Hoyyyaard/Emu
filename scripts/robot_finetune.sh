export CUDA_VISIBLE_DEVICES=0,1,2,3
python cliport_benchmark.py  --instruction_tuning --batch_size 8 --lr_base 0.00001 --epoch 100 --log_dir results/robot_pretrain/1-4 --bf16   --instruct --data_path ../dataset/CLIPORT/emu_data_subtask_l 
