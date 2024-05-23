CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_test \
--data_path ../data/FB15k-237-betae -n 128 -b 512 -d 400 -g 60 \
-lr 0.0001 --max_steps 450001 --cpu_num 5 --geo beta --valid_steps 60000 \
-betam "(1600,2)" --print_on_screen --note betae_caqr \
--use_hyperrel --PE --RE --SE --table_normalization -sd 400 \
--fusion_all

CUDA_VISIBLE_DEVICES=1 python main.py --cuda --do_train --do_test \
--data_path ../data/NELL-betae -n 128 -b 512 -d 400 -g 60 \
-lr 0.0001 --max_steps 500000 --cpu_num 5 --geo beta --valid_steps 60000 \
-betam "(1600,2)" --print_on_screen --note betae_caqr \
--use_hyperrel --PE --RE --SE --table_normalization -sd 108 \
--var_lambda 0.0001


