CUDA_VISIBLE_DEVICES=1 python run.py --cuda --do_train --do_test --do_valid \
  --data_path data/FB15k-237-2 --model BoxTransE -n 128 -b 512 -d 400 -g 24 -a 1.0 \
  -lr 0.0001 --max_steps 300001 --cpu_num 5 --test_batch_size 16 --center_reg 0.2 \
  --geo box --stepsforpath 300001  --offset_deepsets inductive \
  --center_deepsets eleattention --print_on_screen --note q2b_caqr -sd 108 \
  --use_hyperrel --rel_neigh_sample 120 \
  --PE --RE --SE --table_normalization --seed 42 \
  --anchor_not_center_trans


CUDA_VISIBLE_DEVICES=0 python run.py --cuda --do_train --do_test --do_valid \
  --data_path data/NELL-2 --model BoxTransE -n 128 -b 512 -d 400 -g 24 -a 1.0 \
  -lr 0.0001 --max_steps 300001 --cpu_num 5 --test_batch_size 16 --center_reg 0.2 \
  --geo box --stepsforpath 300001  --offset_deepsets inductive \
  --center_deepsets eleattention --print_on_screen --note q2b_caqr -sd 108 \
  --use_hyperrel --rel_neigh_sample 120 \
  --PE --RE --SE --table_normalization --seed 42 \
  --anchor_not_center_trans


