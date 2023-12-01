# Membership Inference of Diffusion Models

In this repository, we propose two membership inference attacks against diffusion models, namely loss-based attack and likelihood-based attack, to infer whether a sample was used to train the model.

[[Paper (arXiv)]](https://arxiv.org/abs/2301.09956) [Paper (ISC2023)](https://orbilu.uni.lu/handle/10993/58604)

## Table of Contents

- [Environment](#Environment)
- [Dataset Preparation](#Dataset-Preparation)
- [Target Model Preparation](#Target-Model-Preparation)
- [Membership Inference Attacks](#Membership-inference-attacks)
  - [Loss-based Attack](##loss-based-attacks)
  - [Likelihood-based Attack](##likelihood-based-attacks)

## Environment

We recommend using Anaconda to manage the Python environment. In this work, we use conda version 4.13.0, and install the following packages in the virtual environment *midm*.

```shell
conda create -n midm python=3.8.13
conda activate midm

pip install tensorflow-gpu==2.4.0
pip install tensorflow-gan==2.0.0
pip install tensorflow_io
pip install tensorflow-datasets==3.1.0
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

pip install scikit-learn==1.2.0
pip install scipy==1.9.3
pip install pillow==9.3.0
pip install ninja==1.10.2.4

pip install jupyter
pip install ipykernel
pip install ipyparallel
pip install jupyter_server
python -m ipykernel install --user --name midm

pip install h5py==2.10.0
pip install ipdb
pip install tqdm
pip install matplotlib==3.6.2
pip install ml-collections==0.1.1

pip install numpy==1.22.1
pip install jax==0.3.23
pip install jaxlib==0.3.22
pip install seaborn==0.12.1

conda install -c conda-forge cudatoolkit-dev==11.7.0
```

## Dataset Preparation

1. Download and unzip the FFHQ dataset: we choose the thumbnails128x128 version --- [FFHQ Dataset.](https://github.com/NVlabs/ffhq-dataset)
  
2. Prepare our dataset through `data_process.py`.
  

```shell
data_process.py:
    --img_dir: Path to the folder containing all original images.(such as Dataset/thumbnails128x128)
    --base_dir: Path to the folder where all results will be saved.
    --num_images: The number of member(training) samples. 
```

The output of `data_process.py` consists of 3 files and 1 folder.

- `ffhq_all.h5py`: This file contains all images in the h5py format.
  
- `ffhq_1000.h5py`: This file contains training images in the h5py format. The number of training samples is 1000, i.e. `--num_images 1000`.
  
- `ffhq_1000_idx.npy`: This file contains the indices of the training images within the `ffhq_all.h5py`. It can be used to split member(training) and nonmember samples.
  
- `tfrecord_dir_ffhq_1000`: This folder contains training images in the tfrecords format at different resolutions. The file name containing 'r06', for example, corresponds to images with a resolution of $64 \times 64$ pixels.
  

## Target Model Preparation

Before training diffusion models, please ensure that we have the correct version of `tensorflow_probability`, which is version 0.12.0. We can install this specific version using the following command:

`pip install tensorflow_probability==0.12.0`

After installing the required version, follow these steps:

1. Enter the `diffusion_models`folder.
  
2. Set `tfrecords_path` in the configuration file under the folder `configs`.
  
3. Train a diffusion model via `main.py`
  

```shell
main.py
    --config: path for a training configuration file.
    --workdir: path for the folder saving training results.
    --mode: 'train' or 'eval'
    --eval_folder: path for the folder saving evaluating results.
```

For more detailed information, please consult [this repository.](https://github.com/yang-song/score_sde_pytorch)

Additionally, we offer a pre-trained model [DDPM](https://drive.google.com/file/d/1b69vT1dWzseXIFSz--2n8dsd_Zxiipu2/view?usp=drive_link) as an illustrative example.

## Membership Inference Attacks

Given a trained diffusion model and a dataset containing member (training) samples and nonmember samples, we can perform membership inference.

### Loss-based attack

Run the loss-based attack via `loss_attack.py`

```shell
loss_attack.py 
    --save_dir: Path to save the results of the attack.
    --model_path: Path to the target model. 
    --shuffled_idx_file: The index file for training images (e.g. ffhq_1000_idx.npy).
    --data_path: Path to the file containing all images in the h5py format (e.g. ffhq_all.h5py).
    --diff_types: Choose from ['ddpm', 'smld', 'vpsde', 'vesde']  to specify the diffusion model type.
    --memb_idx 1000: The index for member(training) samples in a dataset. 
    --num_images 1000: The number of images chosen for inference.
    --diffusion_steps: The diffusion steps (e.g. '0, 200, 500').
```

The output of `loss_attack.py` contains loss values of all samples.

### Likelihood-based attack

Run the likelihood-based attack via `likelihood_attack.py`

```shell
likelihood_attack.py 
```

The parameter in `likelihood_attack.py` has the same meaning as in `loss_attack.py`. The output of `likelihood_attack.py` contains the likelihood values of all samples.

**As an example,** we summarize our attack results for the target model DDPM in `Example.ipynb`.

## Acknowledgements

Our implementation uses the source code in this repository [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch).
