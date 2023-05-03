exp_name='score_b33_normT_RegEnc_FR70FF_RegDec05_scale_confWW_remove2D_50'

CUDA_VISIBLE_DEVICES=0 python -m my_research.main \
    --exp_name $exp_name \
    --config_file my_research/configs/mobrecon_ds_conf_transformer_single.yml
