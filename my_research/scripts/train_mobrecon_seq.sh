exp_name='score_mobrecon_densestack'
# exp_name=''

CUDA_VISIBLE_DEVICES=0 python -m my_research.seq_main \
    --exp_name $exp_name \
    --config_file my_research/configs/mobrecon_ds_seq.yml
