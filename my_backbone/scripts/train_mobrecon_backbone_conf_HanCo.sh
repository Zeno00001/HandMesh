exp_name='densestack_conf_HanCo'

CUDA_VISIBLE_DEVICES=0 python -m my_backbone.main \
    --exp_name $exp_name \
    --config_file my_backbone/configs/densestack_conf_HanCo.yml

# remember to change shape in runner-train, -eval