CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_test \
--print_on_screen --data_path ../data/FB15k-237-betae -n 128 -b 512 -d 800 -g 30 \
--data FB237 -lr 0.00005 --max_steps 300001 --cpu_num 5 --valid_steps 60000 \
--test_batch_size 4 --seed 42 -cenr 0.2 \
--RE --PE --SE --table_normalization -sd 800 \
--use_hyperrel --rel_neigh_sample 120 --fusion_layer 0 \
--note cone_caqr --do_valid

CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_test \
--print_on_screen --data_path ../data/NELL-betae -n 128 -b 512 -d 800 -g 30 \
--data NELL -lr 0.00005 --max_steps 300001 --cpu_num 5 --valid_steps 60000 \
--test_batch_size 4 --seed 42 -cenr 0.2 --drop 0.1 \
--RE --PE --SE --table_normalization -sd 800 \
--use_hyperrel --rel_neigh_sample 120 --fusion_layer 2 \
--regularization 0.00001 \
--note cone_caqr --do_valid
