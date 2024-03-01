# Latte

## Set-up

### All scripts must be run from project root directory

Starting from scratch (create virtual environment, install dependencies, activate venv and set path variables):

`$ source ./scripts/bootstrap`

If venv is already created (just activate venv and set path variables):

`$ source ./scripts/init`

Download data

`$ scripts/download_data`

## Run Experiment
To run an experiment just run the associated bash file.
`bash experiments/bash/run_lm.sh`    
You can also run it from any directory
`bash run_lm.sh`     
The code assumes you have a local directory data in the root with the input and where the output will be saved.
Some datasets are downloaded automatically: Like OpenWebTxt and wiki103,
while others need downloading:``` ./scripts/download_lra```

# Enwiki8
This is a quick guide to run enwik8. To setup the software:   
1. Run ``` source scripts/boostrap. ```
2. ``` source scripts/init ```
3. Install requirements. We did not properly set the requirements.txt since instalation of jax and torch depends on GPU. For now:    
a. install jax: https://jax.readthedocs.io/en/latest/installation.html    
``` pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html ```    
b. Install torch: ```pip install torch```     
c. Reinstall jax: ``` pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html ```    
    - This is a bit buggy since jax install its own cuda kernels, that is why we need to re-run after torch installation.     
d. Install all other libraries: ```pip install transformers datasets wandb tqdm flax```
4. To run enwik8 do:   
    a. Create a folder called data in root.
    b. Create a subfolder named data/logs_latte    
    c. Download data by runing: ```bash scripts/download_chr_data.sh```    
    d. run ``` bash experiments/bash/run_lm_enwik.sh```    
    e. You can configure steps/batch size, etc in ```experiments/config/lm_scale_enwik.yaml```     

