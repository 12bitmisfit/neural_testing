import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow.keras import layers
import time
from IPython import display
from skimage.transform import rescale as skrescale
from tiler import Tiler, Merger

load = True
EPOCHS = 200
batch_size = 384
scale = 0.25
tu_scale = 128
train_dir = "C:/videos/clean/4k"
tf.keras.mixed_precision.set_global_policy('mixed_float16')


def load_image(filename):
    im = Image.open(filename)
    im = im.convert("RGB")
    im = np.array(im)
    im = im / 255
    return im


# Must be called on a 3 dimensional ndarray (2160, 3840, 3)
def resize_image(image, scale):
    rim = skrescale(image, scale, anti_aliasing=False, channel_axis=2)
    return rim


def train_generator(batch_size):
    count = 0
    train_lr = []
    train_hr = []
    for v in os.listdir(train_dir):
        for f in os.listdir(train_dir + "/" + v):
            original_image = load_image(train_dir + "/" + v + "/" + f)
            tiler = Tiler(data_shape=original_image.shape,
                          tile_shape=(tu_scale, tu_scale, 3),
                          channel_dimension=2)
            for tile_id, tile in tiler.iterate(original_image):
                train_hr.append(tile)
                lr = resize_image(tile, scale)
                train_lr.append(lr)
                count += 1
                if count % batch_size == 0:
                    print(count / batch_size)
                    train_lr = np.stack([i for i in train_lr])
                    train_hr = np.stack([i for i in train_hr])
                    yield train_lr, train_hr
                    train_lr = []
                    train_hr = []


trainer = train_generator


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(64, (3, 3), use_bias=False, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(64, (3, 3), use_bias=False, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.UpSampling2D(size=4))

    model.add(layers.Conv2D(128, (3, 3), use_bias=False, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, (3, 3), use_bias=False, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(64, (3, 3), use_bias=False, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(64, (3, 3), use_bias=False, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(3, (3, 3), use_bias=False, padding='same', activation="tanh"))
    assert model.output_shape == (None, 128, 128, 3)

    return model


generator = make_generator_model()
print(generator.summary())


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[128, 128, 3]))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


discriminator = make_discriminator_model()
print(discriminator.summary())
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam()
discriminator_optimizer = tf.keras.optimizers.Adam()

checkpoint_dir = 'C:/Users/lilfiend/PycharmProjects/tensor/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# images is a tuple of low resolution images [0] and high resolution images [1]
@tf.function
def train_step(images):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(images[0], training=True)
        print(images[0].shape)
        print(images[1].shape)
        real_output = discriminator(images[1], training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset(batch_size):
            train_step(image_batch)

        print("test")
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1
                                 )

        # Save the model every epoch
        if (epoch + 1) % 1 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epochs
                             )


def generate_and_save_images(model, epoch):
    checkpoint.save(file_prefix=checkpoint_prefix)
    test_image = load_image("C:/videos/clean/480/2/00001.png")
    tiler = Tiler(data_shape=test_image.shape,
                  tile_shape=(128, 128, 3),
                  channel_dimension=2)
    merger = Merger(tiler)
    test_hr = []
    test_lr = []
    for tile_id, tile in tiler.iterate(test_image):
        test_hr.append(tile)
        lr = resize_image(tile, scale)
        test_lr.append(lr)
    test_hr = np.stack([i for i in test_hr])
    test_lr = np.stack([i for i in test_lr])
    test_sr = []
    for lr in test_lr:
        lr = np.expand_dims(lr, axis=0)
        sr = model(lr, training=False)
        sr = np.squeeze(sr, axis=0)
        print(sr.shape)
        test_sr.append(sr)
    for tile_id, tile in enumerate(test_hr):
        merger.add(tile_id, tile)
    final_hr = merger.merge(unpad=True)
    merger.reset()
    for tile_id, tile in enumerate(test_sr):
        merger.add(tile_id, tile)
    final_sr = merger.merge(unpad=True)
    final_hr = Image.fromarray((final_hr * 255).astype(np.uint8))
    final_sr = Image.fromarray((final_sr * 255).astype(np.uint8))

    final_hr.save('original_image_at_epoch_{:04d}.png'.format(epoch))
    final_sr.save('super_resolution_image_at_epoch_{:04d}.png'.format(epoch))

if load == True:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print("Loaded latest checkpoint")
train(trainer, EPOCHS)
