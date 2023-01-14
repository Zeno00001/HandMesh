# exp_name='test_conf_backbone'
# exp_name='test_conf_backbone_freeze_at_10'
# exp_name='test_conf_backbone_freezed_uv_embed'  # same
# exp_name='test_conf_backbone_nofreeze'  # same
# exp_name='test_conf_backbone_all_connected'  # fail
# exp_name='test_conf_backbone_nofreeze__restart_from_pretrained'  # fail
# exp_name='test_conf_backbone_nofreeze__restart_from_pretrained_2'  # OK
# exp_name='test_conf_backbone_nofreeze_new_norm'  # OK
# exp_name='test_conf_backbone_nofreeze_decoder'  # OK
# exp_name='test_conf_backbone_norm'
# exp_name='test_conf_backbone_batchnorm'
# exp_name='test_conf_backbone_norm__nodetach'
# exp_name='base_decoder_layer1'
# exp_name='base_decoder_layer6'
# exp_name='base_decoder_layer3'
# exp_name='enc_only_layer3'
# exp_name='dec_only_layer3'
# exp_name='base_enc4_dec6'
# exp_name='base_enc5_dec6'
# exp_name='base_enc3_dec6_normfrist'
# exp_name='dec_only3_diagmask'
# exp_name='base_enc3_dec6_J_loss'
# exp_name='base_enc3_dec6_J_loss_onEncOut'
# exp_name='base_enc3_dec6_joint_mask'
# exp_name='base_enc3_dec3_normtwice'
# exp_name='base_enc3_dec3_normfirst'
# exp_name='base_enc3_dec3_normtwice_reg_at_index'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc'
# exp_name='base_enc3_dec3_normfirst_reg_at_enc'
# exp_name='base_enc3_dec3_normtwice_reg_at_dec'
# exp_name='base_enc3_dec3_normfirst_reg_at_dec'

# exp_name='decoder_only3_normfirst'
# exp_name='base_enc3_dec3_normonce_reg_at_enc'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_jointmask'

# exp_name='base_enc3_dec3_normtwice_reg_at_enc_diag_p01p01p01'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_diag_p01f10p01'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_49verts'

# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_AR21F'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_AR21Z_50epoch'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_AR21Z_weightConf'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_AR21Z_50epoch_38decay'
exp_name='base_enc3_dec3_normtwice_reg_at_each_enc_DF_AR21Z'

CUDA_VISIBLE_DEVICES=0 python -m my_research.seq_main \
    --exp_name $exp_name \
    --config_file my_research/configs/mobrecon_ds_conf_transformer.yml
