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

| Layer                 | Output Shape | Param Count |
|-----------------------|--------------|-------------|
| text_encoder          | multiple     | 27,865,600  |
| image_encoder         | multiple     | 44,642,432  |
| embedding_1           | multiple     | 2,048       |
| embedding_2           | multiple     | 64          |
| flatten_1             | multiple     | 0           |
| u_net_diffusion_module| multiple     | 59,797,216  |
| **Total**             |              | **104720416** |
| Trainable params      |              | 104717472 |
| Non-trainable params |              | 2,944       |

### ImageDecoder

| Layer                   | Output Shape | Param Count |
|-------------------------|--------------|-------------|
| conv2d_transpose_3      | multiple     | 2,097,408   |
| batch_normalization_11  | multiple     | 1,024       |
| conv2d_transpose_4      | multiple     | 524,416     |
| batch_normalization_12  | multiple     | 512         |
| conv2d_transpose_5      | multiple     | 131,136     |
| batch_normalization_13  | multiple     | 256         |
| conv2d_transpose_6      | multiple     | 32,800      |
| batch_normalization_14  | multiple     | 128         |
| conv2d_transpose_7      | multiple     | 1,539       |
| batch_normalization_15  | multiple     | 12          |
| **Total**               |              | **2,789,231** |
| Trainable params        |              | 2,788,265   |
| Non-trainable params    |              | 966         |

**Notable**: The total size of the utilized model is **107,509,647 (107.5M) (410.12 MB)**, dependent on the DiffusionModel and ImageDecoder.