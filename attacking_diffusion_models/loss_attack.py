# -*- coding: utf-8 -*-


import os
import h5py
import timeit
import random
import argparse
import numpy as np

import torch

# Keep the import below for registering all model definitions
import losses
import sde_lib
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
 
def get_loss_ddpm(state, batch, train=False,reduce_mean=True):
    model = state['model']
    #  training hyperparameters
    N = 1000
    beta_min=0.1
    beta_max=20
    discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    alphas = 1. - discrete_betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    
    all_loss=[]
    im_batch_size=200
    with torch.no_grad():
        model_fn = mutils.get_model_fn(model, train=train)
            
        labels = torch.arange(0, N, device=batch.device).reshape(1,-1).repeat_interleave(batch.size(0),dim=0).reshape(-1)
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(batch.device)
        sqrt_1m_alphas_cumprod = sqrt_1m_alphas_cumprod.to(batch.device)
        
        batch = batch.repeat_interleave(N,dim=0)
        noise = torch.randn_like(batch)
        perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                         sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
        
        for i_batch in range(0, N, im_batch_size): 
            
            score = model_fn(perturbed_data[i_batch : i_batch + im_batch_size, :], labels[i_batch : i_batch + im_batch_size])
            losses = torch.square(score - noise[i_batch : i_batch + im_batch_size, :])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
            all_loss += losses.cpu().numpy().tolist()
    
    return all_loss

def get_loss_vpsde(state, batch, sde, t, train=False,reduce_mean=True,eps=1e-5):
    model = state['model']
    
    #  training hyperparameters
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    
    all_loss=[]
    im_batch_size=200
    N=1000 # here N is noise times
    with torch.no_grad():
        
        score_fn = mutils.get_score_fn(sde, model, train=train, continuous=True)
        t = t.to(batch.device).reshape(1,-1).repeat_interleave(batch.size(0),dim=0).reshape(-1)
        batch = batch.repeat_interleave(N,dim=0)
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z
        
        
        for i_batch in range(0, N, im_batch_size): 
            score = score_fn(perturbed_data[i_batch : i_batch + im_batch_size, :], t[i_batch : i_batch + im_batch_size])
            losses = torch.square(score * std[i_batch : i_batch + im_batch_size, None, None, None] + z[i_batch : i_batch + im_batch_size])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
            
            all_loss += losses.cpu().numpy().tolist()

    return all_loss

    
def get_loss_smld(state, batch, train=False,reduce_mean=True):
    model = state['model']
    #  training hyperparameters
    N = 1000
    sigma_min=0.001
    sigma_max=50
  
    
    discrete_sigmas = torch.exp(torch.linspace(np.log(sigma_min), np.log(sigma_max), N))
    smld_sigma_array = torch.flip(discrete_sigmas, dims=(0,))
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    all_loss=[]
    im_batch_size=200
    with torch.no_grad():
        model_fn = mutils.get_model_fn(model, train=train)
            
        labels = torch.arange(0, N, device=batch.device).reshape(1,-1).repeat_interleave(batch.size(0),dim=0).reshape(-1)
        sigmas = smld_sigma_array.to(batch.device)[labels]
        noise = torch.randn_like(batch) * sigmas[:, None, None, None]
        perturbed_data = noise + batch
         
        for i_batch in range(0, N, im_batch_size): 
            
            score = model_fn(perturbed_data[i_batch : i_batch + im_batch_size, :], labels[i_batch : i_batch + im_batch_size])
            target = -noise[i_batch : i_batch + im_batch_size, :] / (sigmas[i_batch : i_batch + im_batch_size] ** 2)[:, None, None, None]
            losses = torch.square(score - target)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas[i_batch : i_batch + im_batch_size] ** 2
            
            all_loss += losses.cpu().numpy().tolist()
    
    return all_loss

def get_loss_vesde(state, batch, sde, t, train=False,reduce_mean=True,eps=1e-5):
    model = state['model']
    #  training hyperparameters
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    
    all_loss=[]
    im_batch_size=200
    N=1000 # here N is noise times
    with torch.no_grad():
        
        score_fn = mutils.get_score_fn(sde, model, train=train, continuous=True)
        t = t.to(batch.device).reshape(1,-1).repeat_interleave(batch.size(0),dim=0).reshape(-1)
        batch = batch.repeat_interleave(N,dim=0)
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z
        
    
        for i_batch in range(0, N, im_batch_size): 
            score = score_fn(perturbed_data[i_batch : i_batch + im_batch_size, :], t[i_batch : i_batch + im_batch_size])
            losses = torch.square(score * std[i_batch : i_batch + im_batch_size, None, None, None] + z[i_batch : i_batch + im_batch_size])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
            
            all_loss += losses.cpu().numpy().tolist()

    return all_loss

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
      
    # compute loss values
    n_images = images.shape[0]
    im_batch_size=1
    all_losses=[]
    if diff_types == 'ddpm':     
        for i_batch in range(0, n_images, im_batch_size):
            if i_batch*im_batch_size%100==0:
                print(i_batch)
            batch = torch.from_numpy(images[i_batch : i_batch + im_batch_size, :]).to(config.device).float()
            batch = (batch * 2. - 1.)
            loss = get_loss_ddpm(state, batch)
            all_losses.append(loss)

            
            if i_batch*im_batch_size%100==0:
                all_losses_np = np.array(all_losses)
                if args.save_dir!='None':
                    if not os.path.exists(args.save_dir):
                        os.makedirs(args.save_dir)
                    h5py_file = f'{args.save_dir}/loss_ffhq_{args.num_images}_{args.diff_types}.h5py'
                    with h5py.File(h5py_file, "w") as f:
                        f.create_dataset('losses', data = all_losses_np, dtype='float32')   
                        
    elif diff_types == 'vpsde':
        t = torch.rand(1000) * (1 - 1e-5) + 1e-5
        for i_batch in range(0, n_images, im_batch_size):
            if i_batch*im_batch_size%100==0:
                print(i_batch)
            batch = torch.from_numpy(images[i_batch : i_batch + im_batch_size, :]).to(config.device).float()
            batch = (batch * 2. - 1.)

            loss = get_loss_vpsde(state, batch, sde, t)
            all_losses.append(loss) 

            if i_batch*im_batch_size%100==0:
                all_losses_np = np.array(all_losses)
                if args.save_dir!='None':
                    if not os.path.exists(args.save_dir):
                        os.makedirs(args.save_dir)
                    h5py_file = f'{args.save_dir}/loss_ffhq_{args.num_images}_{args.diff_types}.h5py'
                    with h5py.File(h5py_file, "w") as f:
                        f.create_dataset('losses', data = all_losses_np, dtype='float32')  
                        f.create_dataset('t', data = t, dtype='float32')  
                        
                        
    elif diff_types == 'smld':
        for i_batch in range(0, n_images, im_batch_size):
            if i_batch*im_batch_size%100==0:
                print(i_batch)
            batch = torch.from_numpy(images[i_batch : i_batch + im_batch_size, :]).to(config.device).float()

            loss = get_loss_smld(state, batch)
            all_losses.append(loss)


            if i_batch*im_batch_size%100==0:
                all_losses_np = np.array(all_losses)
                if args.save_dir!='None':
                    if not os.path.exists(args.save_dir):
                        os.makedirs(args.save_dir)
                    h5py_file = f'{args.save_dir}/loss_ffhq_{args.num_images}_{args.diff_types}.h5py'
                    with h5py.File(h5py_file, "w") as f:
                        f.create_dataset('losses', data = all_losses_np, dtype='float32')   
                        
                        
    elif diff_types == 'vesde':      
        t = torch.rand(1000) * (1 - 1e-5) + 1e-5
        for i_batch in range(0, n_images, im_batch_size):
            if i_batch*im_batch_size%100==0:
                print(i_batch)
            batch = torch.from_numpy(images[i_batch : i_batch + im_batch_size, :]).to(config.device).float()
            loss = get_loss_vesde(state, batch, sde, t)
            all_losses.append(loss)  

            if i_batch*im_batch_size%100==0:
                all_losses_np = np.array(all_losses)
                if args.save_dir!='None':
                    if not os.path.exists(args.save_dir):
                        os.makedirs(args.save_dir)
                    h5py_file = f'{args.save_dir}/loss_ffhq_{args.num_images}_{args.diff_types}.h5py'
                    with h5py.File(h5py_file, "w") as f:
                        f.create_dataset('losses', data = all_losses_np, dtype='float32')   
                        f.create_dataset('t', data = t, dtype='float32')  
                        
                        
    
    all_losses_np = np.array(all_losses)                
    if args.save_dir!='None':
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        h5py_file = f'{args.save_dir}/loss_ffhq_{args.num_images}_{args.diff_types}.h5py'
        if diff_types == 'ddpm' or diff_types == 'smld':     
            with h5py.File(h5py_file, "w") as f:
                f.create_dataset('losses', data = all_losses_np, dtype='float32')   
        elif diff_types == 'vpsde' or diff_types == 'vesde':     
            t = t.numpy()  
            with h5py.File(h5py_file, "w") as f:
                f.create_dataset('losses', data = all_losses_np, dtype='float32')       
                f.create_dataset('t', data = t, dtype='float32')  
                
    # attack performance: TFP@FPR
    attack_performance(h5py_file, diff_types,args.num_images)
    
def attack_performance(data_path, diff_types,num_images):
    if diff_types == 'ddpm' or diff_types == 'smld':   
        with h5py.File(data_path, "r") as f:
            resluts = f['losses'][:]    # losses, smaller is better
        print(resluts.shape)
        
    elif diff_types == 'vpsde' or diff_types == 'vesde':         
        with h5py.File(data_path, "r") as f:
            _resluts = f['losses'][:]    # losses, smaller is better
            _t=f['t'][:]
        print(_resluts.shape)

        sort_idx= np.argsort(_t)  
        t = _t[sort_idx]
        resluts = _resluts[:,sort_idx] # sorting by t
        print(resluts.shape)
        print(t.shape)
    
    labels = np.concatenate((np.zeros(num_images), np.ones(num_images))) 


    all_steps_tpr_at_low_fpr=[]

    print(resluts.shape[1])
    for i in range(resluts.shape[1]):
        tpr_at_low_fpr_1 =get_metrics(labels, resluts[:,i], fixed_fpr=0.1)
        tpr_at_low_fpr_2 =get_metrics(labels, resluts[:,i], fixed_fpr=0.01)
        tpr_at_low_fpr_3 =get_metrics(labels, resluts[:,i], fixed_fpr=0.001)
        tpr_at_low_fpr_4 =get_metrics(labels, resluts[:,i], fixed_fpr=0.0001)

        

        all_steps_tpr_at_low_fpr.append([tpr_at_low_fpr_1,tpr_at_low_fpr_2,tpr_at_low_fpr_3,tpr_at_low_fpr_4])

    print('finished!')    

    all_steps_tpr_at_low_fpr_ = np.reshape(all_steps_tpr_at_low_fpr,(-1,4))
    
    diffusion_steps = args.diffusion_steps.split(',')
    print('step, [TPR@10%FPR,TPR@1%FPR,TPR@0.1%FPR,TPR@0.01%FPR]')
    for i in diffusion_steps:
        print('t=',i,' ', all_steps_tpr_at_low_fpr_[int(i)]*100)



            
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
    
    parser.add_argument('--diffusion_steps', default='0,200,500', type=str, help='The diffusion steps, separate in comma [0-999].')
     
    args, other_args = parser.parse_known_args()
    
    print('Settings:')
    for arg in vars(args):
        print('\t{}: {}'.format(arg, getattr(args, arg)))
    ### RUN
    main(args)