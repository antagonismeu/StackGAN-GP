from PIL import Image
import numpy as np


def add_noise(image, noise_level=0.1):
    noise = np.random.normal(loc=0.0, scale=noise_level, size=image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, -1.0, 1.0)  


num_images = int(input("Enter the number of noisy images to generate: "))


height = 256
width = 256
channels = 3


noise_level = 0.1  

for i in range(num_images):
    
    clean_image = np.random.rand(height, width, channels)
    
    
    noisy_image = add_noise(clean_image, noise_level=noise_level)
    
    
    noisy_image_pil = Image.fromarray(((noisy_image + 1) * 127.5).astype(np.uint8))
    
    
    image_index = i + 1025  
    image_filename = f"images/{image_index}.jpg"
    noisy_image_pil.save(image_filename)

    print(f"Saved image {image_filename}")

print("All images saved successfully.")