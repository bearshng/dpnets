


python hsi_eval.py --gpu-ids 1 --bn 0 -a cscnet --conv_num 2  --unfolding 4 --num_half_layer 6 --no-cuda -r -rp checkpoints/cscnet/cscnet_l1_bn0_0.00008_4_95_2_128/model_latest.pth   -tr ../../data/ICVL/test/  -gr  ../../data/ICVL/test_crop/ --noise_level 95 --channels 128


python hsi_eval.py --gpu-ids 1 --bn 0 -a cscnet --conv_num 2  --unfolding 4 --num_half_layer 6 --no-cuda -r -rp checkpoints/cscnet/cscnet_l1_bn0_0.00008_4_55_2_128/model_latest.pth   -tr ../../data/ICVL/test/  -gr  ../../data/ICVL/test_crop/ --noise_level 55 --channels 128


python hsi_eval.py --gpu-ids 1 --bn 0 -a cscnet --conv_num 2  --unfolding 4 --num_half_layer 6 --no-cuda -r -rp checkpoints/cscnet/cscnet_l1_bn0_0.00008_4_15_2_128/model_latest.pth   -tr ../../data/ICVL/test/  -gr  ../../data/ICVL/test_crop/ --noise_level 15 --channels 128
