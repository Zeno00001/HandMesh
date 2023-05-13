# exp_name='mrc_ds_angle'
# exp_name='mrc_ds_angle_2_head'
# exp_name='mrc_ds_angle_4_head_fake'       # total 50 epoch, bad mesh
# exp_name='mrc_ds_angle_4_head'            # bad negative pred
# exp_name='mrc_ds_angle_4_head_train_GCN'  # bad negative pred
exp_name='mrc_ds_angle_4_head_freezeAll30'  # bad negative pred

CUDA_VISIBLE_DEVICES=0 python -m my_research.main \
    --exp_name $exp_name \
    --config_file my_research/configs/mobrecon_ds_angle.yml
