# CDM
Compressive diffusion model.

### Description
This is a repository for the compressive diffusion model (CDM) for image compression.
The models are based on the Variational Diffusion Model (VDM) and the Latent
Diffusion Model (LDM). The models are trained on the MNIST and CIFAR10 datasets. An
additional model was made using the VDM alongside BB-ANS for the MNIST dataset (this
model is lossless). The models are trained using the PyTorch framework.

VDM paper: https://arxiv.org/abs/2107.00630

LDM paper: https://arxiv.org/abs/2112.10752

### Roadmap
- [✓] Create the v0.1 of the model for the MNIST dataset
- [✓] Create the autoencoder model for v0.1 
- [✓] Create the score model for v0.1
- [✓] Train it on the dataset
- [✓] Test and record findings
- [✓] Create the v0.2 of the model for the CIFAR10 dataset
- [✓] Create the advanced autoencoder model for v0.2
- [✓] Create the advanced score model for v0.2
- [✓] Train it on the dataset
- [✓] Test and record findings

### How to use
- Check the notebooks in the model folder for examples on how to use the model.
You might have to do some changes to the code to make it work for you (but should work
out of the box for the MNIST/CIFAR dataset and most systems).
- Checkpoint files are not stored due to the minimal size of the datasets. Of course
these models (BB-ANS and CIFAR10) are scalable to larger datasets, but the training time will increase.

### BB-ANS
BB-ANS is a lossless compression algorithm that is used in the VDM model. This is
possible do with a diffusion model as it can act as a latent model. The BB-ANS
uses this latent model to compress the data. More details can be found in the report
and the sources listed there.
