exp_name='mrc_rs'
CUDA_VISIBLE_DEVICES=0
python -m my_research.main \
    --exp_name $exp_name \
    --config_file my_research/configs/mobrecon_rs.yml