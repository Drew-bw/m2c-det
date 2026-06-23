C='configs/dsec/gfl_cross.py'
W='work_dirs/dsec/gfl_baseline'
WND='0-0-0-0'
CUDA_VISIBLE_DEVICES=6 python train_kd.py  --config=$C --work-dir=$W --wnd=$WND
