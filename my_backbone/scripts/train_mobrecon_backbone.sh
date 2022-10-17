exp_name='densestack'

CUDA_VISIBLE_DEVICES=0 python -m my_backbone.main \
    --exp_name $exp_name \
    --config_file my_backbone/configs/densestack.yml
