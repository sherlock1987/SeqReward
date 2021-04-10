This repository is for gan-vae reward function training of paper "Guided Dialog Policy Training without Adversarial Learning in the Loop"

### VAE-GAN training: 
```sh
python -u mwoz_gan_vae.py --max_epoch 400 --data_dir ./data/multiwoz --init_lr 0.0005 --vae_loss bce --l2_lambda 0.000001 --early_stop True --round_for_disc False --gan_type wgan --op rmsprop --gan_ratio 3 
```

Requirements: pytorch-1.0, cuda-9.2, nltk-3.2.5, Python-2.7

* The dataset and pretrained reward function can be found [here](https://drive.google.com/file/d/1qjoLU3RzkI3EUyE8yuhpUpDTD1fmt6Ez/view?usp=sharing). Copy the downloaded dataset to path './data/' and unzip them. 

You can reuse the pretrained vae model: './logs/cl_3_VAE_pre_training_mode', this pretrained vae model give me the best results.

Then you couuld train the multi-GAN network based on this model.

This code is based on the source code of Ziming's work.
