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
Here is the brief summary of the number of parameters in both the DiffusionModel and DecoderModel:

### VAE

| Layer (type)               | Output Shape | Param #     |
|----------------------------|--------------|-------------|
| conv2d                     | multiple      | 896         |
| batch_normalization        | multiple      | 128         |
| leaky_re_lu                | multiple      | 0           |
| residual_block             | multiple      | 18624       |
| conv2d_3                   | multiple      | 18496       |
| batch_normalization_2      | multiple      | 256         |
| leaky_re_lu_3              | multiple      | 0           |
| residual_block_1           | multiple      | 74112       |
| conv2d_6                   | multiple      | 73856       |
| batch_normalization_4      | multiple      | 512         |
| leaky_re_lu_6              | multiple      | 0           |
| residual_block_2           | multiple      | 295680      |
| conv2d_9                   | multiple      | 295168      |
| batch_normalization_6      | multiple      | 1024        |
| leaky_re_lu_9              | multiple      | 0           |
| residual_block_3           | multiple      | 1181184     |
| conv2d_12                  | multiple      | 1180160     |
| batch_normalization_8      | multiple      | 2048        |
| leaky_re_lu_12             | multiple      | 0           |
| residual_block_4           | multiple      | 4721664     |
| flatten                    | multiple      | 0           |
| dense                      | multiple      | 33555456    |
| dense_1                    | multiple      | 33587200    |
| reshape                    | multiple      | 0           |
| conv2d_transpose           | multiple      | 1179904     |
| batch_normalization_10     | multiple      | 1024        |
| leaky_re_lu_15             | multiple      | 0           |
| residual_block_5           | multiple      | 1181184     |
| conv2d_transpose_1         | multiple      | 295040      |
| batch_normalization_12     | multiple      | 512         |
| leaky_re_lu_18             | multiple      | 0           |
| residual_block_6           | multiple      | 295680      |
| conv2d_transpose_2         | multiple      | 73792       |
| batch_normalization_14     | multiple      | 256         |
| leaky_re_lu_21             | multiple      | 0           |
| residual_block_7           | multiple      | 74112       |
| conv2d_transpose_3         | multiple      | 18464       |
| batch_normalization_16     | multiple      | 128         |
| leaky_re_lu_24             | multiple      | 0           |
| residual_block_8           | multiple      | 18624       |
| conv2d_transpose_4         | multiple      | 4624        |
| batch_normalization_18     | multiple      | 64          |
| leaky_re_lu_27             | multiple      | 0           |
| residual_block_9           | multiple      | 4704        |
| conv2d_transpose_5         | multiple      | 435         |
| **Total params**           |              | **78155011**|
| Trainable params           |              | 78149059    |
| Non-trainable params       |              | 5952        |


### Text2ImageModel

| Layer (type)                | Output Shape | Param #     |
|-----------------------------|--------------|-------------|
| text_encoder (TextEncoder) | multiple      | 1688192     |
| embedding_1 (Embedding)    | multiple      | 2048        |
| embedding_2 (Embedding)    | multiple      | 64          |
| flatten_1 (Flatten)        | multiple      | 0           |
| vae (VAE)                   | multiple      | 78155011    |
| u_net_diffusion_module (UNetDiffusionModule) | multiple | 917700416   |
| **Total params**           |              | **997545731** |
| Trainable params           |              | 997529987   |
| Non-trainable params       |              | 15744        |


**Notable**: The total size of the utilized model is **997,545,731 (approximate 1B parameters) (3.72 GB)**, dependent on the StableDiffusionModel(***SD***) and Variational Auto-Encoder(***VAE***).