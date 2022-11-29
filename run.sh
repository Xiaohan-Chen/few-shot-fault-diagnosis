# python3 Prototypical.py --n_train 800 --lr_scheduler 'exp' --showstep 5

# python3 penalty_prototypical.py --n_train 300 --lr_scheduler step --showstep 20

# python3 Prototypical.py --n_train 800 --lr_scheduler 'step' --showstep 5

# python3 moprototypical.py --n_train 800 --m 0.99 --lr_scheduler 'stepLR' --showstep 5 --lr 1e-3 --max_epoch 700 --split 1

python3 moprototypical.py --n_train 1000 --m 0.99 --lr_scheduler 'stepLR' --step '30' --target_dataname 'CWRU' --t_load 2 --showstep 5 --lr 1e-3 --max_epoch 300 --split 1