conda activate pytorch_gpu
#python hsi_eval.py --gpu-ids 0 --bn 0 -a mscnet_l1 --unfolding 4 --num_half_layer 3 -r -rp checkpoints/mscnet_l1/mscnet_l1_bn0_16_0.001_5_v2/model_latest.pth  -tr ../../data/ICVL/test/  -gr  ../../data/ICVL/test_crop/


#python hsi_eval.py --gpu-ids 0 --bn 0 -a mscnet_l1 --unfolding 4 --num_half_layer 3 -r -rp checkpoints/mscnet_l1/mscnet_l1_bn0_16_0.0005_5_v2/model_latest.pth  -tr ../../data/ICVL/test/  -gr  ../../data/ICVL/test_crop/



#python hsi_eval.py --gpu-ids 0 --bn 0 -a mscnet_l1 --unfolding 4 --num_half_layer 3 -r -rp checkpoints/mscnet_l1/mscnet_l1_bn0_0.001_5_v2/model_latest.pth  -tr ../../data/ICVL/test/  -gr  ../../data/ICVL/test_crop/


#python hsi_eval.py --gpu-ids 0 --bn 0 -a mscnet_l1 --unfolding 4 --num_half_layer 3 -r -rp checkpoints/mscnet_l1/mscnet_l1_bn0_0.0005_5_v2/model_latest.pth  -tr ../../data/ICVL/test/  -gr  ../../data/ICVL/test_crop/


#python hsi_eval.py --gpu-ids 0 --bn 0 -a mscnet_l1 --unfolding 4 --num_half_layer 3 -r -rp checkpoints/mscnet_l1/mscnet_l1_bn0_0.0001_5_v2/model_latest.pth  -tr ../../data/ICVL/test/  -gr  ../../data/ICVL/test_crop/
python hsi_eval.py --gpu-ids 0 --bn 0 -a mscnet --unfolding 4 --num_half_layer 3 -r -rp checkpoints/mscnet/prename_bn0_16_0.001_v2/model_latest.pth -tr ../../data/ICVL/test/  -gr  ../../data/ICVL/test_crop/


python hsi_eval.py --gpu-ids 0 --bn 0 -a mscnet --unfolding 4 --num_half_layer 3 -r -rp checkpoints/mscnet/prename_bn0_16_0.0001_v2/model_latest.pth -tr ../../data/ICVL/test/  -gr  ../../data/ICVL/test_crop/


python hsi_eval.py --gpu-ids 0 --bn 0 -a mscnet --unfolding 4 --num_half_layer 3 -r -rp checkpoints/mscnet/prename_bn0_16_0.0005_v2/model_latest.pth -tr ../../data/ICVL/test/  -gr  ../../data/ICVL/test_crop/
