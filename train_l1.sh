conda activate pytorch_gpu 
#python hsi_denoising_gauss.py --gpu-ids 0 -a mscnet_l1 -p mscnet_l1_bn0_0.001_5_v2 --bn 0 --unfolding 4 --num_half_layer 3 --batchSize 8 --lr 1e-3 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   -gr  ../../data/ICVL/test_crop/
#
##python hsi_denoising_gauss.py --gpu-ids 0 -a mscnet_l1 -p mscnet_l1_bn0_0.005 --bn 0 --unfolding 4 --num_half_layer 3 --batchSize 8 --lr 5e-3 --dataroot ./datasets/ICVL64_31_2mats.db
#python hsi_denoising_gauss.py --gpu-ids 0 -a mscnet_l1 -p mscnet_l1_bn0_0.0005_5_v2 --bn 0 --unfolding 4 --num_half_layer 3 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   -gr  ../../data/ICVL/test_crop/
#
#python hsi_denoising_gauss.py --gpu-ids 0 -a mscnet_l1 -p mscnet_l1_bn0_0.0001_5_v2 --bn 0 --unfolding 4 --num_half_layer 3 --batchSize 8 --lr 1e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/    -gr  ../../data/ICVL/test_crop/
#
#python hsi_denoising_gauss.py --gpu-ids 0 -a mscnet_l1 -p mscnet_l1_bn0_16_0.001_5_v2 --bn 0 --unfolding 4 --num_half_layer 3 --batchSize 16 --lr 1e-3 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   -gr  ../../data/ICVL/test_crop/
##python hsi_denoising_gauss.py --gpu-ids 0 -a mscnet_l1 -p mscnet_l1_bn0_16_0.005_v2 --bn 0 --unfolding 4 --num_half_layer 3 --batchSize 16 --lr 5e-3 --dataroot ./datasets/ICVL64_31_2mats.db
#
#python hsi_denoising_gauss.py --gpu-ids 0 -a mscnet_l1 -p mscnet_l1_bn0_16_0.0001_5_v2 --bn 0 --unfolding 4 --num_half_layer 3 --batchSize 16 --lr 1e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   -gr  ../../data/ICVL/test_crop/
#
#python hsi_denoising_gauss.py --gpu-ids 0 -a mscnet_l1 -p mscnet_l1_bn0_16_0.0005_5_v2 --bn 0 --unfolding 4 --num_half_layer 3 --batchSize 16 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_sep.py --gpu-ids 1 -a mscnet_l1 -p mscnet_l1_bn0_0.001_95 --bn 0 --unfolding 4 --num_half_layer 3 --batchSize 8 --lr 1e-3 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_sep.py --gpu-ids 1 -a mscnet_l1 -p mscnet_l1_bn0_0.0005_95 --bn 0 --unfolding 4 --num_half_layer 3 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_sep.py --gpu-ids 1 -a mscnet_l1 -p mscnet_l1_bn0_0.0001_95 --bn 0 --unfolding 4 --num_half_layer 3 --batchSize 8 --lr 1e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   -gr  ../../data/ICVL/test_crop/

#-r -rp checkpoints/mscnet/prename/model_latest.pth  for resume
#--dataroot ./datasets/ICVL64_31.db from 100mats



#python hsi_denoising_gauss_95.py --gpu-ids 1 -a mscnet_l1 -p mscnet_l1_bn0_0.0005_4_95_1 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --conv_num 1  -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_55.py --gpu-ids 1 -a mscnet_l1 -p mscnet_l1_bn0_0.0005_4_55_1 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   --conv_num 1 -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_15.py --gpu-ids 1 -a mscnet_l1 -p mscnet_l1_bn0_0.0005_4_15_1 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --conv_num 1  -gr  ../../data/ICVL/test_crop/
#
#
#python hsi_denoising_gauss_95.py --gpu-ids 1 -a mscnet_l1 -p mscnet_l1_bn0_0.0005_4_95_2 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --conv_num 2  -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_55.py --gpu-ids 1 -a mscnet_l1 -p mscnet_l1_bn0_0.0005_4_55_2 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   --conv_num 2 -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_15.py --gpu-ids 1 -a mscnet_l1 -p mscnet_l1_bn0_0.0005_4_15_2 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --conv_num 2  -gr  ../../data/ICVL/test_crop/
#
#
#
#python hsi_denoising_gauss_95.py --gpu-ids 1 -a mscnet_l1 -p mscnet_l1_bn0_0.0005_4_95_4 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --conv_num 4  -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_55.py --gpu-ids 1 -a mscnet_l1 -p mscnet_l1_bn0_0.0005_4_55_4 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   --conv_num 4 -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_15.py --gpu-ids 1 -a mscnet_l1 -p mscnet_l1_bn0_0.0005_4_15_4 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --conv_num 4  -gr  ../../data/ICVL/test_crop/
#


#python hsi_denoising_gauss_95.py --gpu-ids 0 -a mscnet_l1 -p cscnet_l1_bn0_0.0005_4_95_5 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --conv_num 5  -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_55.py --gpu-ids 0 -a mscnet_l1 -p cscnet_l1_bn0_0.0005_4_55_5 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   --conv_num 5 -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_15.py --gpu-ids 0 -a mscnet_l1 -p cscnet_l1_bn0_0.0005_4_15_5 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --conv_num 5  -gr  ../../data/ICVL/test_crop/

#python hsi_denoising_gauss_95.py --gpu-ids 0 -a mscnet_l1 -p cscnet_l1_bn0_0.0005_4_95_1_128 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --conv_num 1 --channels 128 -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_55.py --gpu-ids 0 -a mscnet_l1 -p cscnet_l1_bn0_0.0005_4_55_1_128 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   --conv_num 1 --channels 128 -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_15.py --gpu-ids 0 -a mscnet_l1 -p cscnet_l1_bn0_0.0005_4_15_1_128 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --conv_num 1  --channels 128 -gr  ../../data/ICVL/test_crop/

python hsi_denoising_gauss_95.py --gpu-ids 1 -a mscnet_l1 -p mscnet_l1_bn0_0.0005_4_95_2_128 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --conv_num 2 --channels 64 -gr  ../../data/ICVL/test_crop/
python hsi_denoising_gauss_55.py --gpu-ids 1 -a mscnet_l1 -p mscnet_l1_bn0_0.0005_4_55_2_128 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   --conv_num 2 --channels 64 -gr  ../../data/ICVL/test_crop/
python hsi_denoising_gauss_15.py --gpu-ids 1 -a mscnet_l1 -p mscnet_l1_bn0_0.0005_4_15_2_128 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --conv_num 2  --channels 64 -gr  ../../data/ICVL/test_crop/


python hsi_denoising_gauss_95.py --gpu-ids 1 -a mscnet_l1 -p mscnet_l1_bn0_0.0005_4_95_3_128 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --conv_num 3 --channels 32 -gr  ../../data/ICVL/test_crop/
python hsi_denoising_gauss_55.py --gpu-ids 1 -a mscnet_l1 -p mscnet_l1_bn0_0.0005_4_55_3_128 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   --conv_num 3 --channels 32 -gr  ../../data/ICVL/test_crop/
python hsi_denoising_gauss_15.py --gpu-ids 1 -a mscnet_l1 -p mscnet_l1_bn0_0.0005_4_15_3_128 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --conv_num 3  --channels 32 -gr  ../../data/ICVL/test_crop/
