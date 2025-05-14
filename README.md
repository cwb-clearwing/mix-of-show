# Mix-of-Show Experiments for Riemannian AdaGrad

## Repository Overview

* [datasets/](datasets) contains the training images.
* [mixofshow/](mixofshow) contains the source code used for models.
* [options/](options) contains training and testing hyperparameters.

## Requirements
```bash
pip install -r requirements.txt
 ```
See the [Mix-of-Show](https://github.com/TencentARC/Mix-of-Show/tree/main) repository for requirement details.

## Quickstart
1. Download pretrained model
 ```bash
mkdir experiments
mkdir experiments/pretrained_models
cd experiments/pretrained_models
git-lfs clone https://huggingface.co/windwhinny/chilloutmix.git
 ```
2. Create model directory
```bash
cd ../..    
mkdir experiments/8101_potter_rada_rank_4
mkdir experiments/8101_potter_rada_rank_4/models
 ```
3. Prepare for datasets
Prepare dataset by yourself or find dataset from other Mix-of-show repos on open source platform:

https://github.com/pilancilab/Riemannian_Preconditioned_LoRA/tree/main/Mix-of-Show/datasets

or

https://github.com/TencentARC/Mix-of-Show/tree/main/datasets



4. Train (specify learning rates in <code>-opt</code> files)
```bash
accelerate launch train_edlora.py -opt options/train/EDLoRA/real/8101_potter_rada_rank_4.yml --optimizer radagrad
```

Here <code>radagrad, radam, rgd </code> are all valid choices for <code>--optimizer</code>. Scaled AdamW is set as dafualt. Trained models will be saved to <code>experiments/8101_potter_rada_rank_4/models</code>.

5. Create image directory
```bash
mkdir results
mkdir results/8101_potter_rada_rank_4
 ```

6. Test
```bash
python test_edlora.py -opt options/test/EDLoRA/human/8101_potter_rada_rank_4.yml
```
Figures will be saved to <code>results/8101_potter_rada_rank_4/visualization</code>.


## Parameter Reference

See our paper for reference.