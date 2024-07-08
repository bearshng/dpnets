#conda activate torch_37_2
#python hsi_denoising_gauss.py --gpu-ids 1,2 -a mscnet_l1 -p mscnet_l1_bn0_0.001 --bn 0 --unfolding 4 --num_half_layer 3 --batchSize 8 --lr 1e-3 --dataroot ./datasets/ICVL64_31_2mats.db
#python hsi_denoising_gauss.py --gpu-ids 1,2 -a mscnet_l1 -p mscnet_l1_bn0_0.0005 --bn 0 --unfolding 4 --num_half_layer 3 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db
##python hsi_denoising_gauss.py --gpu-ids 1,2 -a mscnet_l1 -p mscnet_l1_bn0_16_0.001 --bn 0 --unfolding 4 --num_half_layer 3 --batchSize 16 --lr 1e-3 --dataroot ./datasets/ICVL64_31_2mats.db
##python hsi_denoising_gauss.py --gpu-ids 0 -a mscnet_l1 -p mscnet_l1_bn0_16_0.005 --bn 0 --unfolding 4 --num_half_layer 3 --batchSize 16 --lr 5e-3 --dataroot ./datasets/ICVL64_31_2mats.db
#python hsi_denoising_gauss.py --gpu-ids 1,2 -a mscnet -p prename_bn0_0.0005 --bn 0 --unfolding 4 --num_half_layer 3 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db
#
#python hsi_denoising_gauss.py --gpu-ids 1,2 -a mscnet -p prename_bn0_0.001 --bn 0 --unfolding 4 --num_half_layer 3 --batchSize 8 --lr 1e-3 --dataroot ./datasets/ICVL64_31_2mats.db
##-r -rp checkpoints/mscnet/prename/model_latest.pth  for resume
##--dataroot ./datasets/ICVL64_31.db from 100mats

#
#python hsi_denoising_gauss_95.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.0005_4 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_95.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.0001_4 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 1e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_95.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00005_4 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_95.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00001_4 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 1e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   -gr  ../../data/ICVL/test_crop/


#python hsi_denoising_gauss_95.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00002_4_95 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 2e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   -gr  ../../data/ICVL/test_crop/
#
#python hsi_denoising_gauss_55.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00002_4_55 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 2e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   -gr  ../../data/ICVL/test_crop/
#

#python hsi_denoising_gauss_95.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00008_4_95_1 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 8e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --conv_num 1  -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_55.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00008_4_55_1 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 8e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   --conv_num 1 -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_15.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00008_4_15_1 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 8e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --conv_num 1  -gr  ../../data/ICVL/test_crop/
#
#
#python hsi_denoising_gauss_95.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00008_4_95_2 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 8e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --conv_num 2  -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_55.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00008_4_55_2 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 8e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   --conv_num 2 -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_15.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00008_4_15_2 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 8e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --conv_num 2  -gr  ../../data/ICVL/test_crop/
#
#
#
#python hsi_denoising_gauss_95.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00008_4_95_4 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 8e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --conv_num 4  -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_55.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00008_4_55_4 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 8e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   --conv_num 4 -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_15.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00008_4_15_4 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 8e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --conv_num 4  -gr  ../../data/ICVL/test_crop/
#
#
#
#python hsi_denoising_gauss_95.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00008_4_95_5 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 8e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --conv_num 5  -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_55.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00008_4_55_5 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 8e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   --conv_num 5 -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_15.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00008_4_15_5 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 8e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --conv_num 5  -gr  ../../data/ICVL/test_crop/

#python hsi_denoising_gauss_55.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.0001_4_55 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 1e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_55.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00005_4_55 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_55.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00001_4_55 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 1e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   -gr  ../../data/ICVL/test_crop/

#python hsi_denoising_gauss_15.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00002_4_15 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 2e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   -gr  ../../data/ICVL/test_crop/

python hsi_denoising_gauss_95.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00008_4_95_1_128 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 8e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --channels 128 --conv_num 1  -gr  ../../data/ICVL/test_crop/
python hsi_denoising_gauss_55.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00008_4_55_1_128 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 8e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --channels 128 --conv_num 1 -gr  ../../data/ICVL/test_crop/
python hsi_denoising_gauss_15.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00008_4_15_1_128 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 8e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --channels 128 --conv_num 1  -gr  ../../data/ICVL/test_crop/


python hsi_denoising_gauss_95.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00008_4_95_2_128 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 8e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --channels 64 --conv_num 1  -gr  ../../data/ICVL/test_crop/
python hsi_denoising_gauss_55.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00008_4_55_2_128 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 8e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --channels 64 --conv_num 1 -gr  ../../data/ICVL/test_crop/
python hsi_denoising_gauss_15.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00008_4_15_2_128 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 8e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/  --channels 64 --conv_num 1  -gr  ../../data/ICVL/test_crop/


#python hsi_denoising_gauss_15.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.0001_4_15 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 1e-4 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_15.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00005_4_15 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 5e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   -gr  ../../data/ICVL/test_crop/
#python hsi_denoising_gauss_15.py --gpu-ids 0 -a cscnet -p cscnet_l1_bn0_0.00001_4_15 --bn 0 --unfolding 4 --num_half_layer 5 --batchSize 8 --lr 1e-5 --dataroot ./datasets/ICVL64_31_2mats.db  --testroot ../../data/ICVL/test/   -gr  ../../data/ICVL/test_crop/
