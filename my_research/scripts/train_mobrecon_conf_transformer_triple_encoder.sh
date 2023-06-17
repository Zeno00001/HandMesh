exp_name='triple_encoder'

CUDA_VISIBLE_DEVICES=0 python -m my_research.seq_main \
    --exp_name $exp_name \
    --config_file my_research/configs/mobrecon_ds_conf_transformer_triple_encoder.yml
