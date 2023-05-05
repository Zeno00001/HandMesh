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
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_AR21Z_50epoch_38decay'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_AR21Z_weightConf'
# exp_name='base_enc3_dec3_normtwice_reg_at_each_enc_DF_AR21Z'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FR21Z'  # overfit while A -> F

# exp_name='decoder_only3_normfirst_DF_FR21Z'

# exp_name='base_enc3_dec3_normtwice_reg_at_dec_DF_FR70F'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FR70F'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_AR70F'  # 2, append bad
# exp_name='base_enc3_dec6_normtwice_reg_at_enc_DF_FR70F'  # 3, dec layer 6, bad bad
# exp_name='base_enc3_dec6_normtwice_reg_at_enc_DF_FR70F_again'  # 3 again, bad bad
# exp_name='base_enc3_dec1_normtwice_reg_at_enc_DF_FR70F'  # 4, dec layer 1, bad

# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FX49F'  # 49 verts with Feature version

# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FR70FF'  # 70 verts with feature, feature version

# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_AR21F_norm'  # not softmax but norm(0, 1)
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_AR21F_norm_weightConf'

# Check useage?
# ? Replace or NOT ?
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FR49F_with_verts_mem'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FX49F_with_verts_mem'

# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FR70FF_15reg_norm'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FR70FF_15reg'

# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FR21F'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FX21F'

# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FR21F_embedonce'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_dec_DF_FR70FF'

# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FX49F_without_FF'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FX49F_head4'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FX21Z'

# TODO cross image feature
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FX49F_cross_image'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FR70FF_15reg_cross_image'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FX49F_ENC_cross_image'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FX49F_ENC_cross_image_nodetach'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FX49F_ENC_cross_image8x8'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FX49F_ENC_cross_image16x16'


# exp_name='base_enc3_dec3_normtwice_reg_at_enc_dec_DF_FR70FZ'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_dec_DF_FR70FZ_norm'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FR70FZ_norm'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FX49Z_norm'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FX49F'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FX49F_norm_conf_ww'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FX49F_norm_conf_wx'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FX49F_norm_conf_wx_50epoch'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FX49F_norm_conf_xw_50epoch'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FX49F_norm_conf_ww_50epoch'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FX49F_conf_ww_50epoch'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FX49F_norm_conf_ww_50epoch_again'
# exp_name='z_prev/base_decoder_layer3'

# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FR70ZZ_reg_mesh'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FX49F_norm_conf_wx_16frame_50epoch'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FR70FF_15reg_norm_conf_ww'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FR70FF_15reg_norm_conf_ww_again'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FR70FF_15reg_norm_conf_ww_stdize'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FR70FF_15reg_norm_conf_ww_stdize_remove2D'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FR70FF_15reg_norm_conf_ww_stdize_remove2D_diagMask_p02n0n0'
# exp_name='base_enc3_dec3_normtwice_reg_at_enc_DF_FR70FF_15reg_norm_conf_ww_stdize_remove2D_diagMask_p10n0n0'

# Rotation Augment Related
# exp_name='rot_base_enc3_dec3_normtwice_reg_at_enc_DF_FR21F'

# exp_name='rot_diff_norm_256'
# exp_name='rot_same_norm_256'  # not implemented yet

# {every one use base33} _ {normTwice} _ {RegEnc} _ {Decoder Forward} _ {Additional RegDec, partial}
#   {scale instead of softmax} _ {apply confidence} _ {standardize encodings...} _ {remove 2D encoding} _ {epoches}
# exp_name='prerequire_mobrecon_ds_layernorm_15'

# exp_name='ablation_b33_normT_RegEnc_FR70FF_RegDec05_scale_confWW_remove2D_50'

# exp_name='score_b33_normT_RegEnc_FR70FF_RegDec05_scale_confWW_50'
exp_name='score_b33_normT_RegEnc_FR70FF_RegDec05_scale_confWW_remove2D_50'
# exp_name='score_b33_normT_RegEnc_FX49F_RegDec05_scale_confWW_remove2D_50'
# exp_name='ablation_b33_normL_RegEnc_FR70FF_RegDec05_scale_confWW_remove2D_50'
# exp_name='ablation_b33_normT_RegEnc_FR70FF_RegDec05__confWW_remove2D_50'

# exp_name='ablation_e30_normT_RegEnc_FR70FF_RegDec05_scale_confWW_remove2D_50'
# much edits here, 1. b33-> e30, 2. FR21F,
# 3. ReturnEncoderOutput='no', 4. # pred_joint += [self.joint_head(out_joint)]
# diff in model, no additional reg head

# exp_name='remove_conf'

# exp_name='score_b33_normT_RegEnc_FR70FF_RegDec05_scale_confWW_50_mergedTemporalEncoding'
# exp_name='score_b33_normT_RegEnc_FR70FF_RegDec05_scale_confWW_50_mergedTemporalEncoding_stdize'
# exp_name='score_b33_normT_RegEnc_FR70FF_RegDec05_scale_confWW_50'
# exp_name='score_b33_normT_RegEnc_FR70FF_RegDec05_scale_confWW_50_remove2D_p02n0n0'
# exp_name='score_b33_normT_RegEnc_FR70FF_RegDec05_scale_confWW_50_remove2D_p02n0n0_stdize'
# exp_name='score_a_temp_BS30'

# exp_name='score_mobrecon_densestack_wo_LN_50'

# exp_name='score_b33_normT_RegEnc_FR70FF_RegDec05_scale_confWW_stdize_remove2D_50'
# exp_name='ablation_'



CUDA_VISIBLE_DEVICES=0 python -m my_research.seq_main \
    --exp_name $exp_name \
    --config_file my_research/configs/mobrecon_ds_conf_transformer.yml
