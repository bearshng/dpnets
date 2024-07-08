import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse

from utility import *
from hsi_setup import Engine, train_options, make_dataset
from utility import dataloaders_hsi_test ###modified###

if __name__ == '__main__':
    """Training settings"""
    parser = argparse.ArgumentParser(
        description='Hyperspectral Image Denoising (Gaussian Noise)')
    opt = train_options(parser)
    print(opt)

    """Setup Engine"""
    engine = Engine(opt)

    """Dataset Setting"""
    HSI2Tensor = partial(HSI2Tensor, use_2dconv=engine.get_net().use_2dconv)

    common_transform_1 = lambda x: x

    common_transform_2 = Compose([
        partial(rand_crop, cropx=32, cropy=32),
    ])

    target_transform = HSI2Tensor()
    train_transform_0 = Compose([
        AddNoise(50),
        HSI2Tensor()
    ])
    train_transform_1 = Compose([
        AddNoiseDynamic(15),
        HSI2Tensor()
    ])
    train_transform_2 = Compose([
        AddNoiseDynamic(55),
        HSI2Tensor()
    ])
    train_transform_3 = Compose([
        AddNoiseDynamic(95),
        HSI2Tensor()
    ])
    train_transform_4 = Compose([
        AddNoiseDynamicList((15,35,95)),
        HSI2Tensor()
    ])
    '''
    train_transform_1 = Compose([
        AddNoise(50),
        HSI2Tensor()
    ])
    
    train_transform_2 = Compose([
        AddNoiseBlind([10, 30, 50, 70]),
        HSI2Tensor()
    ])
    '''
    print('==> Preparing data..')
    
    icvl_64_31_TL_0 = make_dataset(
        opt, train_transform_0,
        target_transform, common_transform_1,opt.batchSize )
    icvl_64_31_TL_1 = make_dataset(
        opt, train_transform_1,
        target_transform, common_transform_1, opt.batchSize)
    icvl_64_31_TL_2 = make_dataset(
        opt, train_transform_2,
        target_transform, common_transform_1,opt.batchSize)
    icvl_64_31_TL_3 = make_dataset(
        opt, train_transform_3,
        target_transform, common_transform_1, opt.batchSize)
    icvl_64_31_TL_4 = make_dataset(
        opt, train_transform_4,
        target_transform, common_transform_2, opt.batchSize*4)
    icvl_64_31_TL_5 = make_dataset(
        opt, train_transform_4,
        target_transform, common_transform_1, opt.batchSize)
    '''
    icvl_64_31_TL_2 = make_dataset(
        opt, train_transform_2,
        target_transform, common_transform_2, 64)
    '''
    """Test-Dev"""

    ###modified###
    basefolder = opt.testroot
    mat_names = ['icvl_dynamic_512_15','icvl_dynamic_512_55','icvl_dynamic_512_95']
    #mat_names = ['icvl_512_30', 'icvl_512_50']
    mat_loaders = []
    for noise in (15,55,95):
        test_path = os.path.join(basefolder, str(noise)+'/')
        print('noise:   ',noise,end='')
        mat_loaders.append(dataloaders_hsi_test.get_dataloaders([test_path],verbose=True,grey=False)['test'])
    ###modified###

    #print(icvl_64_31_TL_0.__len__())
    if icvl_64_31_TL_0.__len__()*opt.batchSize > 22470:
        max_epoch = 50
        if_100 = 0
        epoch_per_save = 5
    else:
        max_epoch = 50
        if_100 = 0
        epoch_per_save = 10
    """Main loop"""
    base_lr = opt.lr   
    testsize = 10 ###modified###
    stages=[0, 15, 30, 45, 60, 75]
    while engine.epoch < max_epoch:
        if if_100:
            epoch = engine.epoch * 2
        else:
            epoch = engine.epoch
        display_learning_rate(engine.optimizer)
        np.random.seed() # reset seed per epoch, otherwise the noise will be added with a specific pattern
        if epoch % 10 == 0 and epoch>0 :
            adjust_learning_rate(engine.optimizer, opt.lr*0.5)
            opt.lr = opt.lr*0.5
        # if epoch == stages[0]:
        #     adjust_learning_rate(engine.optimizer, opt.lr)
        # elif epoch == stages[1]:
        #     adjust_learning_rate(engine.optimizer, base_lr*0.1)
        # elif epoch == (stages[1]+(stages[2]-stages[1])//2):
        #     adjust_learning_rate(engine.optimizer, base_lr*0.01)
        # elif epoch ==stages[2]:
        #     adjust_learning_rate(engine.optimizer, base_lr*0.1)
        # elif epoch == (stages[2]+(stages[3]-stages[2])//2):
        #     adjust_learning_rate(engine.optimizer, base_lr*0.01)
        # elif epoch ==stages[3]:
        #     adjust_learning_rate(engine.optimizer, base_lr*0.1)
        # elif epoch == (stages[3]+(stages[4]-stages[3])//2):
        #     adjust_learning_rate(engine.optimizer, base_lr*0.01)
        # elif epoch == stages[4]:
        #     adjust_learning_rate(engine.optimizer, base_lr*0.001)
        '''
        elif engine.epoch % 30 == 1 and engine.epoch != 1:
            adjust_learning_rate(engine.optimizer, base_lr)

        elif engine.epoch % 30 == 0 and engine.epoch != 0:
            adjust_learning_rate(engine.optimizer, base_lr*0.01)
        '''
        #print(if_100)
        # if epoch < stages[1]:
            #engine.validate(mat_loaders[0], 'icvl-validate-15')
        engine.train(icvl_64_31_TL_1)
        # if epoch % 5==0:
        #     engine.validate(mat_loaders[0], 'icvl-validate-15')###modified###
        #     if if_100:
        #         engine.validate(mat_loaders[1], 'icvl-validate-55')###modified###
        #         engine.validate(mat_loaders[2], 'icvl-validate-95')###modified###
        #     #engine.validate(mat_loaders[1], 'icvl-validate-50')
        # elif epoch < stages[2]:
        #     engine.train(icvl_64_31_TL_1)
        #     if epoch % 5 ==0:
        #         engine.validate(mat_loaders[0], 'icvl-validate-15')###modified###
        #     if if_100:
        #         engine.validate(mat_loaders[1], 'icvl-validate-55')###modified###
        #         engine.validate(mat_loaders[2], 'icvl-validate-95')###modified###
        #     #engine.validate(mat_loaders[0], 'icvl-validate-15')
        #     #engine.validate(mat_loaders[0], 'icvl-validate-30')
        #     #engine.validate(mat_loaders[1], 'icvl-validate-50')
        # elif epoch < stages[3]:
        #     engine.train(icvl_64_31_TL_2)
        #     if epoch % 5 ==0:
        #         engine.validate(mat_loaders[0], 'icvl-validate-15')###modified###
        #     if if_100:
        #         engine.validate(mat_loaders[1], 'icvl-validate-55')###modified###
        #         engine.validate(mat_loaders[2], 'icvl-validate-95')###modified###
        # elif epoch < stages[4]:
        #     engine.train(icvl_64_31_TL_3)
        #     if epoch % 5 ==0:
        #         engine.validate(mat_loaders[0], 'icvl-validate-15')###modified###
        #     if if_100:
        #         engine.validate(mat_loaders[1], 'icvl-validate-55')###modified###
        #         engine.validate(mat_loaders[2], 'icvl-validate-95')###modified###
        # else:
        #     engine.train(icvl_64_31_TL_5)
        #     if epoch % 5 ==0:
        #         engine.validate(mat_loaders[0], 'icvl-validate-15')###modified###
        #     if if_100:
        #         engine.validate(mat_loaders[1], 'icvl-validate-55')###modified###
        #         engine.validate(mat_loaders[2], 'icvl-validate-95')###modified###
        
        print('\nLatest Result Saving...')
        model_latest_path = os.path.join(engine.basedir, engine.prefix, 'model_latest.pth')
        engine.save_checkpoint(
            model_out_path=model_latest_path
        )

        
        if engine.epoch % epoch_per_save == 0:###modified###
            engine.save_checkpoint()
