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

### Text2ImageDiffusionModel

| Layer                | Output Shape | Param Count |
|----------------------|--------------|-------------|
| text_encoder         | multiple     | 27,865,600  |
| image_encoder        | multiple     | 44,642,432  |
| embedding_1          | multiple     | 8,192       |
| u_net_diffusion_module | multiple   | 64,232,432  |
| **Total**            |              | **109,161,712** |
| Trainable params     |              | 109,154,160 |
| Non-trainable params |              | 7,552       |

### ImageDecoder

| Layer                | Output Shape | Param Count |
|----------------------|--------------|-------------|
| conv2d_transpose_1533 | multiple   | 2,097,408   |
| batch_normalization_11242 | multiple | 1,024     |
| conv2d_transpose_1534 | multiple   | 524,416     |
| batch_normalization_11243 | multiple | 512       |
| conv2d_transpose_1535 | multiple   | 131,136     |
| batch_normalization_11244 | multiple | 256       |
| conv2d_transpose_1536 | multiple   | 32,800      |
| batch_normalization_11245 | multiple | 128       |
| conv2d_transpose_1537 | multiple   | 1,539       |
| batch_normalization_11246 | multiple | 12        |
| **Total**            |              | **2,789,231** |
| Trainable params     |              | 2,788,265   |
| Non-trainable params |              | 966         |


**Notable**: the generation just depend on **DiffusionModel** and **ImageDecoder** so the size of the utilized model is **111,950,943(112M)(427.06MB)**