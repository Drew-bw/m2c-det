C='configs/dsec/dfine_s_cross.py'
W='work_dirs/dsec/dfine_s_cross/25_avg'
WND='2-4-1-1'
CUDA_VISIBLE_DEVICES=5 python train_kd.py  --config=$C --work-dir=$W --wnd=$WND
