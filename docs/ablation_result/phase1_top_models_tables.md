# Ablation Phase 1 - Top 10 Models

**Total Experiments:** 1392

**Generated:** 2026-01-27 14:11:29.475075

---

### Top 10 by ROC-AUC

| experiment | roc_auc | f1_score | pa_20_roc_auc | pa_20_f1 | pa_50_roc_auc | pa_50_f1 | pa_80_roc_auc | pa_80_f1 | disc_ratio | disc_ratio_disturbing | recon_ratio | disc_cohens_d_normal_vs_anomaly | disc_cohens_d_disturbing_vs_anomaly | recon_cohens_d_normal_vs_anomaly | recon_cohens_d_disturbing_vs_anomaly | inference_mode | scoring_mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 007_patch_20_mask_before_default_all | 0.9624 | 0.8464 | 0.987 | 0.7721 | 0.9801 | 0.7702 | 0.9598 | 0.7297 | 1.7369 | 1.3128 | 6.0038 | 0.9682 | 0.3716 | 1.9495 | 1.0819 | all_patches | default |
| 009_w500_p20_mask_before_default_last | 0.9586 | 0.8826 | 0.9793 | 0.8161 | 0.9689 | 0.8072 | 0.9257 | 0.7336 | 1.6887 | 1.6905 | 4.5918 | 0.727 | 0.4901 | 2.083 | 1.2418 | last_patch | default |
| 009_w500_p20_mask_before_default_all | 0.9578 | 0.7401 | 0.9942 | 0.7866 | 0.9893 | 0.771 | 0.9652 | 0.7109 | 2.1526 | 2.0167 | 4.4999 | 1.1793 | 0.8029 | 3.7745 | 2.3125 | all_patches | default |
| 023_d_model_256_mask_before_default_last | 0.9575 | 0.8477 | 0.9808 | 0.8281 | 0.9689 | 0.8254 | 0.9209 | 0.7659 | 1.3095 | 1.0174 | 6.3155 | 0.4113 | 0.021 | 2.1105 | 1.1996 | last_patch | default |
| 019_decoder_t4s2_mask_before_default_all | 0.9572 | 0.8121 | 0.9932 | 0.8177 | 0.9878 | 0.8129 | 0.9603 | 0.7319 | 1.9363 | 1.401 | 5.6922 | 0.9007 | 0.3514 | 2.2982 | 1.3616 | all_patches | default |
| 016_decoder_t4s1_mask_before_default_all | 0.9564 | 0.8046 | 0.9941 | 0.8065 | 0.9883 | 0.8029 | 0.9588 | 0.7212 | 1.971 | 1.4467 | 5.6021 | 0.8183 | 0.3376 | 2.3083 | 1.368 | all_patches | default |
| 022_d_model_128_mask_before_default_last | 0.9557 | 0.8636 | 0.9828 | 0.8306 | 0.9692 | 0.8279 | 0.9297 | 0.7867 | 2.1606 | 1.1313 | 6.597 | 0.8269 | 0.1164 | 2.1526 | 1.2389 | last_patch | default |
| 015_decoder_t3s1_mask_before_default_all | 0.9555 | 0.8145 | 0.9932 | 0.8226 | 0.9863 | 0.8178 | 0.9583 | 0.7156 | 1.7313 | 1.2763 | 5.6653 | 0.7312 | 0.2534 | 2.2317 | 1.3111 | all_patches | default |
| 018_decoder_t3s2_mask_before_default_all | 0.9532 | 0.8083 | 0.9927 | 0.8215 | 0.9869 | 0.8158 | 0.9568 | 0.7129 | 1.6681 | 1.1478 | 5.7237 | 0.6544 | 0.1403 | 2.1957 | 1.2869 | all_patches | default |
| 054_lr_0.005_mask_before_default_last | 0.9521 | 0.8601 | 0.9809 | 0.8366 | 0.9661 | 0.8306 | 0.9241 | 0.7866 | 1.3475 | 0.8784 | 6.1224 | 0.408 | -0.1534 | 2.1888 | 1.2421 | last_patch | default |


### Top 10 by Discrepancy Ratio

| experiment | roc_auc | f1_score | pa_20_roc_auc | pa_20_f1 | pa_50_roc_auc | pa_50_f1 | pa_80_roc_auc | pa_80_f1 | disc_ratio | disc_ratio_disturbing | recon_ratio | disc_cohens_d_normal_vs_anomaly | disc_cohens_d_disturbing_vs_anomaly | recon_cohens_d_normal_vs_anomaly | recon_cohens_d_disturbing_vs_anomaly | inference_mode | scoring_mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 050_k_4.0_mask_after_normalized_all | 0.8719 | 0.5812 | 0.9422 | 0.5475 | 0.9144 | 0.5318 | 0.8704 | 0.4602 | 4.2588 | 1.6971 | 1.5627 | 1.8689 | 0.7131 | 0.5329 | -0.1385 | all_patches | normalized |
| 050_k_4.0_mask_after_adaptive_all | 0.8793 | 0.5773 | 0.9458 | 0.5327 | 0.9212 | 0.52 | 0.8819 | 0.4705 | 4.2548 | 1.6958 | 1.5633 | 1.8669 | 0.7117 | 0.5318 | -0.1396 | all_patches | adaptive |
| 050_k_4.0_mask_after_default_all | 0.8123 | 0.5169 | 0.9048 | 0.4728 | 0.8476 | 0.4624 | 0.7727 | 0.3827 | 4.247 | 1.6915 | 1.5619 | 1.8658 | 0.7096 | 0.5315 | -0.1411 | all_patches | default |
| 057_dropout_0.3_mask_after_normalized_all | 0.8506 | 0.5364 | 0.9248 | 0.4907 | 0.8908 | 0.47 | 0.845 | 0.4211 | 4.2349 | 1.5351 | 1.3339 | 1.7531 | 0.5859 | 0.3284 | -0.2746 | all_patches | normalized |
| 162_d128_dropout0.3_mask_after_normalized_all | 0.8506 | 0.5364 | 0.9248 | 0.4907 | 0.8908 | 0.47 | 0.845 | 0.4211 | 4.2349 | 1.5351 | 1.3339 | 1.7531 | 0.5859 | 0.3284 | -0.2746 | all_patches | normalized |
| 057_dropout_0.3_mask_after_default_all | 0.7368 | 0.4182 | 0.8637 | 0.3568 | 0.7795 | 0.3389 | 0.6549 | 0.2865 | 4.2235 | 1.5322 | 1.3284 | 1.7452 | 0.5813 | 0.3231 | -0.2802 | all_patches | default |
| 162_d128_dropout0.3_mask_after_default_all | 0.7368 | 0.4182 | 0.8637 | 0.3568 | 0.7795 | 0.3389 | 0.6549 | 0.2865 | 4.2235 | 1.5322 | 1.3284 | 1.7452 | 0.5813 | 0.3231 | -0.2802 | all_patches | default |
| 057_dropout_0.3_mask_after_adaptive_all | 0.8613 | 0.5634 | 0.9297 | 0.5294 | 0.9029 | 0.5073 | 0.8622 | 0.4434 | 4.2199 | 1.5285 | 1.3245 | 1.7414 | 0.5777 | 0.3182 | -0.2852 | all_patches | adaptive |
| 162_d128_dropout0.3_mask_after_adaptive_all | 0.8613 | 0.5634 | 0.9297 | 0.5294 | 0.9029 | 0.5073 | 0.8622 | 0.4434 | 4.2199 | 1.5285 | 1.3245 | 1.7414 | 0.5777 | 0.3182 | -0.2852 | all_patches | adaptive |
| 118_d128_patch25_mask_after_default_all | 0.8041 | 0.5199 | 0.9018 | 0.4068 | 0.8436 | 0.3804 | 0.741 | 0.3078 | 4.0919 | 1.0801 | 1.8957 | 1.4148 | 0.1078 | 1.0545 | 0.4809 | all_patches | default |


### Top 10 by Teacher Reconstruction Ratio

| experiment | roc_auc | f1_score | pa_20_roc_auc | pa_20_f1 | pa_50_roc_auc | pa_50_f1 | pa_80_roc_auc | pa_80_f1 | disc_ratio | disc_ratio_disturbing | recon_ratio | disc_cohens_d_normal_vs_anomaly | disc_cohens_d_disturbing_vs_anomaly | recon_cohens_d_normal_vs_anomaly | recon_cohens_d_disturbing_vs_anomaly | inference_mode | scoring_mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 022_d_model_128_mask_before_default_last | 0.9557 | 0.8636 | 0.9828 | 0.8306 | 0.9692 | 0.8279 | 0.9297 | 0.7867 | 2.1606 | 1.1313 | 6.597 | 0.8269 | 0.1164 | 2.1526 | 1.2389 | last_patch | default |
| 022_d_model_128_mask_before_adaptive_last | 0.9277 | 0.771 | 0.966 | 0.7484 | 0.9436 | 0.7274 | 0.8882 | 0.6558 | 2.1606 | 1.1313 | 6.597 | 0.8269 | 0.1164 | 2.1526 | 1.2389 | last_patch | adaptive |
| 022_d_model_128_mask_before_normalized_last | 0.9182 | 0.7539 | 0.9629 | 0.7424 | 0.9368 | 0.7171 | 0.8664 | 0.6196 | 2.1606 | 1.1313 | 6.597 | 0.8269 | 0.1164 | 2.1526 | 1.2389 | last_patch | normalized |
| 030_ffn_512_mask_before_default_last | 0.9473 | 0.8423 | 0.9795 | 0.8161 | 0.9618 | 0.8083 | 0.9147 | 0.759 | 1.3785 | 0.8388 | 6.3706 | 0.4016 | -0.1954 | 1.9971 | 1.1001 | last_patch | default |
| 030_ffn_512_mask_before_adaptive_last | 0.8971 | 0.7232 | 0.955 | 0.7392 | 0.914 | 0.6898 | 0.8347 | 0.5553 | 1.3785 | 0.8388 | 6.3706 | 0.4016 | -0.1954 | 1.9971 | 1.1001 | last_patch | adaptive |
| 030_ffn_512_mask_before_normalized_last | 0.8559 | 0.6566 | 0.9346 | 0.6965 | 0.8823 | 0.6242 | 0.7708 | 0.4676 | 1.3785 | 0.8388 | 6.3706 | 0.4016 | -0.1954 | 1.9971 | 1.1001 | last_patch | normalized |
| 029_ffn_128_mask_before_default_last | 0.9479 | 0.8483 | 0.9797 | 0.8199 | 0.9626 | 0.804 | 0.9168 | 0.7667 | 1.3119 | 0.8839 | 6.3323 | 0.3553 | -0.1466 | 2.0915 | 1.1729 | last_patch | default |
| 029_ffn_128_mask_before_adaptive_last | 0.8954 | 0.7187 | 0.9567 | 0.754 | 0.9112 | 0.676 | 0.8367 | 0.5638 | 1.3119 | 0.8839 | 6.3323 | 0.3553 | -0.1466 | 2.0915 | 1.1729 | last_patch | adaptive |
| 029_ffn_128_mask_before_normalized_last | 0.8566 | 0.6188 | 0.9348 | 0.6193 | 0.8719 | 0.5726 | 0.7821 | 0.4677 | 1.3119 | 0.8839 | 6.3323 | 0.3553 | -0.1466 | 2.0915 | 1.1729 | last_patch | normalized |
| 023_d_model_256_mask_before_default_last | 0.9575 | 0.8477 | 0.9808 | 0.8281 | 0.9689 | 0.8254 | 0.9209 | 0.7659 | 1.3095 | 1.0174 | 6.3155 | 0.4113 | 0.021 | 2.1105 | 1.1996 | last_patch | default |


