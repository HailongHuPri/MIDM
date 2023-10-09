import argparse,random
import h5py


import numpy as np

import os

import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage


from utils import restore_checkpoint


import models
from models import utils as mutils
from models import ncsnpp
from models import ddpm as ddpm_model

import sampling
from sde_lib import VESDE, VPSDE
from sampling import ReverseDiffusionPredictor, LangevinCorrector
import datasets


def main(args):
    random_seed=args.seed
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(random_seed)
    
    # @title Load the score-based model
    # sde = 'VPSDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
    diff_types = args.diff_types
    if diff_types == 'ddpm':
        sde = 'VPSDE'
        if sde.lower() == 'vpsde':
          from configs.vp.ddpm import ffhq_ddpm as configs  
          ckpt_filename = args.model_path
          config = configs.get_config()
          sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
          sampling_eps = 1e-3
    elif diff_types == 'vpsde':
        sde = 'VPSDE'
        if sde.lower() == 'vpsde':
          from configs.vp import ffhq_vp_cont as configs  
          ckpt_filename = args.model_path
          config = configs.get_config()
          sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
          sampling_eps = 1e-3
    elif diff_types == 'smld':
        sde = 'VESDE'
        if sde.lower() == 'vesde':
          from configs.ve import ffhq_ve_discr as configs
          ckpt_filename = args.model_path
          config = configs.get_config()  
          sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
          sampling_eps = 1e-5
    elif diff_types == 'vesde':  
        sde = 'VESDE'
        if sde.lower() == 'vesde':
          from configs.ve import ffhq_ve_cont as configs
          ckpt_filename = args.model_path
          config = configs.get_config()  
          sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
          sampling_eps = 1e-5
    
    num_imgs=args.num_imgs
    batch_size =   args.batch_size  #@param {"type":"integer"}
    config.training.batch_size = batch_size
    config.eval.batch_size = batch_size
    # import ipdb;ipdb.set_trace() # debugging starts here
    # random_seed = 0 #@param {"type": "integer"}
    

    inverse_scaler = datasets.get_data_inverse_scaler(config)
    score_model = mutils.create_model(config)
    
    optimizer = get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
    state = dict(step=0, optimizer=optimizer,
                 model=score_model, ema=ema)
    
    state = restore_checkpoint(ckpt_filename, state, config.device)
    ema.copy_to(score_model.parameters())
    
    
    #@title PC sampling
    img_size = config.data.image_size
    channels = config.data.num_channels
    shape = (batch_size, channels, img_size, img_size)
    predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
    corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
    snr = 0.16 #@param {"type": "number"}
    n_steps =  1#@param {"type": "integer"}
    probability_flow = False #@param {"type": "boolean"}
    
    all_x = []
    
    
    sampling_fn = sampling.get_pc_sampler(sde, shape, predictor, corrector,
                                          inverse_scaler, snr, n_steps=n_steps,
                                          probability_flow=probability_flow,
                                          continuous=config.training.continuous,
                                          eps=sampling_eps, device=config.device)
    
    # import timeit
    
    for i in range(0, num_imgs, batch_size):
        # start = timeit.default_timer()
        print(i)
        x, n = sampling_fn(score_model)
        all_x.append(x.detach().cpu().numpy())
        # stop = timeit.default_timer()
        # print("Time elapses: {}s".format(stop - start))
    all_x = np.vstack(all_x)
    fake_images = np.clip(np.rint(all_x * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
    

    if args.save_dir!='None':
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        h5py_file = f'{args.save_dir}/query_ffhq_1k_{args.diff_types}_{random_seed}.h5py'
        with h5py.File(h5py_file, "w") as f:
            f.create_dataset('images', data = fake_images, dtype='uint8')
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE')
    
   
    parser.add_argument('--seed', default=0, type=int, help='')
    parser.add_argument('--batch_size', default=100, type=int, help='')
    parser.add_argument('--num_imgs', default=1000, type=int, help='')
    

    parser.add_argument('--save_dir', default='./', type=str, help='') 
    parser.add_argument('--model_path', default='./', type=str, help='')   
    
    parser.add_argument('--diff_types', default='ddpm', type=str, help='') 

       
    args, other_args = parser.parse_known_args()
    
    print('Settings:')
    for arg in vars(args):
        print('\t{}: {}'.format(arg, getattr(args, arg)))
    ### RUN
    main(args)
    
    
    