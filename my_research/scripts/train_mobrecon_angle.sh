# exp_name='mrc_ds_angle'
# exp_name='mrc_ds_angle_2_head'
# exp_name='mrc_ds_angle_4_head_fake'       # total 50 epoch, bad mesh
# exp_name='mrc_ds_angle_4_head'            # bad negative pred
# exp_name='mrc_ds_angle_4_head_train_GCN'  # bad negative pred

# exp_name='mrc_ds_angle_2_head_freezeAll30'  # bad negative pred
# exp_name='mrc_ds_angle_2_head_freezeUntilUpsample_30'  # a little bit bad negative pred
# exp_name='mrc_ds_angle_2_head_scratch'  # good
# exp_name='mrc_ds_angle_4_head_scratch'  # good


# exp_name='mrc_ds_angle_4_head_freezeBackbone_40_correct'  # <- 229 negs, 80/100
# exp_name='mrc_ds_angle_4_head_scratch_correct'  # 45/60

exp_name='mrc_ds_angle_1_head_pretrained_correct'  # 38/ 90, from 50
# exp_name='mrc_ds_angle_1_relu_head_pretrained_correct'  # 38/ 90, freeze in first 15 epochs


CUDA_VISIBLE_DEVICES=0 python -m my_research.main \
    --exp_name $exp_name \
    --config_file my_research/configs/mobrecon_ds_angle.yml
