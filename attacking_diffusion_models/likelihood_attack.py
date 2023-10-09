# -*- coding: utf-8 -*-



import os
import h5py
import random
import timeit
import argparse
import numpy as np

import torch
import datasets

# Keep the import below for registering all model definitions
import losses
import sde_lib
import likelihood
from models import utils as mutils
from models import ddpm, ncsnv2, ncsnpp
from models.ema import ExponentialMovingAverage


from compute_metrics import get_metrics






def load_checkpoint(ckpt_dir, state, device):

    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state

def load_target_dataset(data_path, random_idx_path, memb_idx, num_images = 10000, seed = 1000):
    with h5py.File(data_path, "r") as f:
        images = f['images'][:]   
        print(images.shape)    
    
    shuffled_idx = np.load(random_idx_path)  # load training data as members
    all_member_images = images[shuffled_idx[:memb_idx]]
    all_nonmember_images = images[shuffled_idx[memb_idx:]]
    
    np.random.seed(seed)       
    shuffled_idx = np.arange(all_member_images.shape[0])
    np.random.shuffle(shuffled_idx)    
    member_images = all_member_images[shuffled_idx[:num_images]]
    print('member_images', member_images.shape)
    
    np.random.seed(seed)     
    shuffled_idx = np.arange(all_nonmember_images.shape[0])
    np.random.shuffle(shuffled_idx)
    nonmember_images = all_nonmember_images[shuffled_idx[:num_images]]  
    print('nonmember_images', nonmember_images.shape)    
    
    all_images=np.concatenate((member_images,nonmember_images),axis=0)
    return all_images


def main(args):

    seed=args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    
    
    # Initialize model.
    diff_types = args.diff_types
    if diff_types == 'ddpm':
        from configs.vp.ddpm import ffhq_ddpm as configs  
    elif diff_types == 'vpsde':
        from configs.vp import ffhq_vp_cont as configs  
    elif diff_types == 'smld':
        from configs.ve import ffhq_ve_discr as configs
    elif diff_types == 'vesde':  
        from configs.ve import ffhq_ve_cont as configs
    
    config = configs.get_config()
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)


    # load a trained model
    checkpoint_meta_dir = args.model_path
    state = load_checkpoint(checkpoint_meta_dir, state, config.device)
    ema.copy_to(score_model.parameters()) 
    

    
    # load a dataset (including member and nonmember data)
    memb_idx=args.memb_idx
    num_images=args.num_images
    random_idx_path=args.shuffled_idx_file
    data_path=args.data_path
    images = load_target_dataset(data_path, random_idx_path, memb_idx, num_images, seed = 1000)
    print(images.shape)
    images = np.clip(images/ 255.0, 0.0, 1.0).astype(np.float32) # [0,255] ==> [0,1]


    if config.training.sde.lower() == 'vpsde':
      sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
      sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
      sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
      sampling_eps = 1e-5
    else:
      raise NotImplementedError(f"SDE {config.training.sde} unknown.")   
    
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    
    n_images = images.shape[0]
    im_batch_size=1
    all_likelihoods=[]
    
    if diff_types == 'ddpm' or diff_types == 'vpsde':   
        if diff_types == 'ddpm':
            likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler, continuous=False)
        elif diff_types == 'vpsde':
            likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler, continuous=True)            
        for i_batch in range(0, n_images, im_batch_size):
            if i_batch*im_batch_size%100==0:
                print(i_batch)
            batch = torch.from_numpy(images[i_batch : i_batch + im_batch_size, :]).to(config.device).float()
            batch = (batch * 2. - 1.)
            
            bpd = likelihood_fn(score_model, batch)[0]
            bpd = bpd.detach().cpu().numpy().reshape(-1)
            all_likelihoods.extend(bpd)
    
            
            if i_batch*im_batch_size%100==0:
                all_likelihoods_np = np.array(all_likelihoods)
                if args.save_dir!='None':
                    if not os.path.exists(args.save_dir):
                        os.makedirs(args.save_dir)
                    h5py_file = f'{args.save_dir}/likelihood_ffhq_{args.num_images}_{args.diff_types}.h5py'
                    with h5py.File(h5py_file, "w") as f:
                        f.create_dataset('likelihood', data = all_likelihoods_np, dtype='float32')   
                        
    elif diff_types == 'smld' or diff_types == 'vesde': 
        if diff_types == 'smld':
            likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler, continuous=False)
        elif diff_types == 'vesde':
            likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler, continuous=True)  
        for i_batch in range(0, n_images, im_batch_size):
            if i_batch*im_batch_size%100==0:
                print(i_batch)
            batch = torch.from_numpy(images[i_batch : i_batch + im_batch_size, :]).to(config.device).float()
            
            bpd = likelihood_fn(score_model, batch)[0]
            bpd = bpd.detach().cpu().numpy().reshape(-1)
            all_likelihoods.extend(bpd)
    

            if i_batch*im_batch_size%100==0:
                all_likelihoods_np = np.array(all_likelihoods)
                if args.save_dir!='None':
                    if not os.path.exists(args.save_dir):
                        os.makedirs(args.save_dir)
                    h5py_file = f'{args.save_dir}/likelihood_ffhq_{args.num_images}_{args.diff_types}.h5py'
                    with h5py.File(h5py_file, "w") as f:
                        f.create_dataset('likelihood', data = all_likelihoods_np, dtype='float32')   

        
    all_likelihoods_np = np.array(all_likelihoods)                
    if args.save_dir!='None':
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        h5py_file = f'{args.save_dir}/likelihood_ffhq_{args.num_images}_{args.diff_types}.h5py'
        with h5py.File(h5py_file, "w") as f:
            f.create_dataset('likelihood', data = all_likelihoods_np, dtype='float32')   

    # attack performance: TFP@FPR
    likelihood_attack_performance(h5py_file,args.num_images)

def likelihood_attack_performance(data_path,num_images):

    with h5py.File(data_path, "r") as f:
        resluts = f['likelihood'][:]    # likelihood, larger is better
    print(resluts.shape)
        
    
    labels = np.concatenate((np.zeros(num_images), np.ones(num_images))) 


    all_steps_tpr_at_low_fpr=[]


    tpr_at_low_fpr_1 =get_metrics(labels, resluts, fixed_fpr=0.1)
    tpr_at_low_fpr_2 =get_metrics(labels, resluts, fixed_fpr=0.01)
    tpr_at_low_fpr_3 =get_metrics(labels, resluts, fixed_fpr=0.001)
    tpr_at_low_fpr_4 =get_metrics(labels, resluts, fixed_fpr=0.0001)

        

    all_steps_tpr_at_low_fpr.append([tpr_at_low_fpr_1,tpr_at_low_fpr_2,tpr_at_low_fpr_3,tpr_at_low_fpr_4])

    print('finished!')    

    all_steps_tpr_at_low_fpr_ = np.reshape(all_steps_tpr_at_low_fpr,(-1,4))
    

    print('[TPR@10%FPR,TPR@1%FPR,TPR@0.1%FPR,TPR@0.01%FPR]')
    print(all_steps_tpr_at_low_fpr_*100)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loss attack')
    
   
    parser.add_argument('--seed', default=2022, type=int, help='')
    parser.add_argument('--model_path', default='./', type=str, help='Path to the target model.')   
    parser.add_argument('--diff_types', default='ddpm', type=str, help='') 
    parser.add_argument('--shuffled_idx_file', default='./', type=str, help='The index file for training images (e.g. ffhq_1000_idx.npy).')   
    parser.add_argument('--data_path', default='./', type=str, help='Path to the file containing all images in the h5py format (e.g. ffhq_all.h5py).') 
    parser.add_argument('--memb_idx', default=1000, type=int, help='The index for member(training) samples in a dataset.')
    parser.add_argument('--num_images', default=1000, type=int, help='The number of images chosen for inference.')
    

    parser.add_argument('--save_dir', default='./', type=str, help='Path to save the results of the attack.')   
    

     
    args, other_args = parser.parse_known_args()
    
    print('Settings:')
    for arg in vars(args):
        print('\t{}: {}'.format(arg, getattr(args, arg)))
    ### RUN
    main(args)