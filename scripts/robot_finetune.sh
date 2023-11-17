export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
python cliport_benchmark.py \
 --instruct  --instruction_tuning --pretrain_predict_task\
 --ckpt_path /root/Projects/Emu/ckpts/Emu-instruct.pt \
 --detail_task --batch_size 4 --lr_base 0.00001 --epoch 1000 \
 --save_ckpt_epoch 5 \
 --log_dir results/robot_pretrain/4-2-6/ --bf16  \
 --data_path ../dataset/CLIPORT/emu_data_subtask_l 

#  --pretrain_predict_task \
#  --pretrain_lora_ckpt /root/Projects/Emu/results/robot_pretrain/4-1-3/robot_ckpt/finetune_8_13984 \