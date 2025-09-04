

## Requirements

A suitable [conda](https://conda.io/) environment named `resshift` can be created and activated with:

```
conda create -n resshift python=3.10
conda activate resshift
pip install -r requirements_new.txt
```

```
## Training

* python realsr_swinunet_realesrgan256_journal.yaml --save_dir

1. Adjust the data path in the [config](configs) file. 
2. Adjust batchsize according your GPUS. 
    + configs.train.batch: [training batchsize, validation batchsize] 
    + configs.train.microbatch: total batchsize = microbatch * #GPUS * num_grad_accumulation

```
## Inference

python inference_resshift.py -i [image folder/image path] -o [result folder] --task realsr --scale 4 --version v3

```


 
## Acknowledgement

This project is based on [ResShift](https://github.com/zsyOAOA/ResShift), [Improved Diffusion Model](https://github.com/openai/improved-diffusion), [LDM](https://github.com/CompVis/latent-diffusion), and [BasicSR](https://github.com/XPixelGroup/BasicSR). 
