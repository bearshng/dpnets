import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from utility import dataloaders_hsi_test
from utility import *
from hsi_setup import Engine, train_options
import models
import  scipy.io as scio


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


prefix = 'test'

if __name__ == '__main__':
    """Training settings"""
    parser = argparse.ArgumentParser(
        description='Hyperspectral Image Denoising')
    opt = train_options(parser)
    print(opt)

    cuda = not opt.no_cuda
    opt.no_log = True

    """Setup Engine"""
    engine = Engine(opt)
    ###modified###
    basefolder = opt.testroot
    # for noise in (15, 55, 95):
    test_path = os.path.join(basefolder)    
    #print(test_path)
    test = dataloaders_hsi_test.get_dataloaders([test_path], verbose=True, grey=False)
    print(test)
    engine.test_real(test['test'], savedir=None)
    # _, _, res_arr = engine.validate(, '')
    # scio.savemat(str(noise) + 'dB_' + 'Res.mat',
    #              {'res_arr': res_arr})    

    # for noise in (15,55,95):
    #     test_path = os.path.join(basefolder, str(noise)+'dB/')
    #     print('noise:   ',noise,end='')
    #     test = dataloaders_hsi_test.get_dataloaders([test_path],verbose=True,grey=False)
    #
    #     # res_arr, input_arr = engine.test_develop(mat_loader, savedir=resdir, verbose=True)
    #     # print(res_arr.mean(axis=0))
    #     _,_,res_arr=engine.validate(test['test'], '')

    ###modified###
