Download the data from https://njusteducn-my.sharepoint.com/:f:/g/personal/ironkitty_njust_edu_cn/EopcsxbL_TFJiQYEHkr9Wd8BllWJFvplYco2arbqgOhwMA?e=eqed5n


### Train:



python hsi_denoising_gauss_95.py --gpu-ids 1 -a mscnet_l1 -p cscnet_new_bn0_0.00008_4_16 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 8e-5 --dataroot 'Training Path'  --testroot ''   -gr  '' --conv_num 3 --channel 16


### Test

python hsi_eval.py --gpu-ids 1 --bn 0 -a mscnet_l1 --unfolding 4    --num_half_layer 5  -r -rp checkpoints/mscnet_l1/15/model_latest.pth -tr 'Testing Path'  -gr  'GT Path' --noise_level 95_20_6 --channels 16

