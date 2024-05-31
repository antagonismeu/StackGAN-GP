## UnetStableDiffusion

**UnetStableDiffusion** is a model incorporating the Structure:**Unet**. Current  code version: **4.1.2**

Run the **SCRIPT** : 

**1.**
IF you want to run the code at a fresh start
```python
python3 refined_MultiGPU_UnetStableDiffusion.py restart
```
**2.**
IF you want to continue the code at the last unexpected collapse
```python
python3 refined_MultiGPU_UnetStableDiffusion.py recover
```
**3.** 
IF you seek for the more specified instruction
```python
python3 refined_MultiGPU_UnetStableDiffusion.py --help[-h]
```

**Model Information**
Here is the brief summary of the number of parameters in both the DiffusionModel and VAEModel:

### VAE

| Layer (type)               | Output Shape | Param #     |
|----------------------------|--------------|-------------|
| conv2d                     | multiple     | 896         |
| down_block                 | multiple     | 166,720     |
| down_block_1               | multiple     | 665,216     |
| down_block_2               | multiple     | 2,657,536   |
| vae_residual_block_6       | multiple     | 1,181,184   |
| vae_residual_block_7       | multiple     | 1,181,184   |
| mid_block2d                | multiple     | 2,559,744   |
| gsc_20                     | multiple     | 590,592     |
| flatten                    | multiple     | 0           |
| dense                      | multiple     | 536,872,960 |
| dense_1                    | multiple     | 268,697,600 |
| reshape                    | multiple     | 0           |
| conv2d_28                  | multiple     | 590,080     |
| mid_block2d_1              | multiple     | 2,559,744   |
| up_block                   | multiple     | 1,182,080   |
| up_block_1                 | multiple     | 296,128     |
| up_block_2                 | multiple     | 74,336      |
| vae_residual_block_21      | multiple     | 18,624      |
| vae_residual_block_22      | multiple     | 18,624      |
| vae_residual_block_23      | multiple     | 18,624      |
| gsc_49                     | multiple     | 9,312       |
| conv2d_transpose           | multiple     | 867         |
| **Total params**           |              | **819,342,051** |
| Trainable params           |              | 819,342,051 |
| Non-trainable params       |              | 0           |

### Text2ImageModel

| Layer (type)                | Output Shape | Param #     |
|-----------------------------|--------------|-------------|
| text_encoder (TextEncoder)  | multiple     | 1688192     |
| flatten_1 (Flatten)         | multiple     | 0           |
| vae (VAE)                   | multiple     | 819342051    |
| u_net_diffusion_module (UNetDiffusionModule) | multiple | 950255424   |
| **Total params**            |              | **1771293347** |
| Trainable params            |              | 1771279715   |
| Non-trainable params        |              | 13632       |


**Notable**: The total size of the utilized model is **1,771,293,347 (approximate 1.77B parameters) (6.60 GB)**, dependent on the StableDiffusionModel(***SD***) and Variational Auto-Encoder(***VAE***).