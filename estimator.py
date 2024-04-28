import os
try:
    import tensorflow as tf
    if tf.__version__.startswith('1'):
        raise ImportError("Please upgrade your TensorFlow to version 2.x")
    from refined_MultiGPU_UnetStableDifffusion import Text2ImageDiffusionModel, load_dataset, ImageDecoder
except :
    requirements = ['numpy', 'tensorflow', 'pandas', 'Pillow', 'transformers']
    for item in requirements :
        os.system(f'pip3 install {item}')
        print('Done!')


global WIDTH, HEIGHT, CHANNEL
width, height = 256, 256                                  #INFERIOR BOUNDARY : width, height = 128, 128  
WIDTH , HEIGHT = width, height
BATCH_SIZE = 4
channel = 3
CHANNEL = channel
GLOBAL_BATCH_SIZE = BATCH_SIZE * tf.distribute.MirroredStrategy().num_replicas_in_sync


def main():
    print('''
        -----------------------------
        ---Estimation Begins--------- 
        -----------------------------

    ''')
    alpha = 0.828
    time_embedding_dim = 512
    csv_path = 'descriptions.csv'
    images_path = './images'

    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of available GPUs: {strategy.num_replicas_in_sync}')


    _, vocab_size, _ = load_dataset(csv_path, images_path, GLOBAL_BATCH_SIZE, height, width)
    model = Text2ImageDiffusionModel(vocab_size, BATCH_SIZE, width, height, channel, alpha)

    text_inputs_example = tf.random.uniform(shape=(BATCH_SIZE, vocab_size))
    image_inputs_example = tf.random.uniform(shape=(BATCH_SIZE, width, height, channel))
    time_steps_example = tf.random.uniform(shape=(512,))

    _ = model(text_inputs_example, image_inputs_example, time_steps_example)

    model.summary()
    

    model2 = ImageDecoder(output_channels=3)

    latent_image_example = tf.random.uniform(shape=(BATCH_SIZE, time_embedding_dim))

    _ = model2(latent_image_example)

    model2.summary()
    print('''
        -----------------------------
        ---Estimation Terminates--------- 
        -----------------------------

    ''')



if __name__ == '__main__' :
    main()