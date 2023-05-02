import tensorflow as tf
import os
import time
import datetime
from matplotlib import pyplot as plt
import numpy as np

from pix2pix_model import *
from utils import EarlyStopping


# Limiting GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[1], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


# Setup the hyperparameters
PATH = '/home/user1/Documents/andylui/DeepEnergy/pix2pix-floorplans-dataset/dataset/Augmented_FSL_to_BFP'
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
LAMBDA = 100  # for generator loss
LR = 1e-5
ADAM_BETA_1 = 0.5
EPOCHS = 1000
EARLY_STOP_PATIENCE = 1000


# Helper functions
def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  w = tf.shape(image)[1]

  # 'w' should be 256:
  w = w // 2

  # Reversing to match our image formatting:
  input_image = image[:, :w, :3]
  real_image = image[:, w:, :3]

  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]

# normalizing the images to [-1, 1]

def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
  # resizing to 286 x 286 x 3
  input_image, real_image = resize(input_image, real_image, 286, 286)

  # randomly cropping to 256 x 256 x 3
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image

def load_image_train(image_file):
  input_image, real_image = load(image_file)

  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def generate_images(model, test_input, tar, out_fpath, is_training=True):
  prediction = model(test_input, training=is_training)
  plt.figure(figsize=(15,15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  
  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  
  plt.savefig(out_fpath)
  plt.show()


@tf.function
def train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, loss_object, input_image, target, epoch, summary_writer):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target, loss_object, LAMBDA)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output, loss_object)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)
  
  return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss


def main():
    # create datasets
    train_dataset = tf.data.Dataset.list_files(PATH+'/Train/*.png')
    train_dataset = train_dataset.map(load_image_train,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.list_files(PATH+'/Test/*.png')
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # create models
    generator = Generator(OUTPUT_CHANNELS)
    tf.keras.utils.plot_model(generator, to_file="gen_model.png", show_shapes=True, dpi=64)

    discriminator = Discriminator()
    tf.keras.utils.plot_model(discriminator, to_file="disc_model.png", show_shapes=True, dpi=64)

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator_optimizer = tf.keras.optimizers.Adam(LR, beta_1=ADAM_BETA_1)
    discriminator_optimizer = tf.keras.optimizers.Adam(LR, beta_1=ADAM_BETA_1)

    # setup the logging
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)
    log_dir="logs/"
    summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # train the model
    early_stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta=0.0001)

    for epoch in range(EPOCHS):
        start = time.time()
        for n, (input_image, target) in train_dataset.enumerate():
            print('.', end='')
            if (n+1) % 100 == 0:
                print()

            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(
              generator, discriminator, generator_optimizer, discriminator_optimizer, loss_object, input_image, target, epoch, summary_writer
            )
        print()
        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))

        # log the metrics
        avg_loss = np.mean([gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss])

        # stop training if the loss has not decreased after {patience} of epochs
        early_stopper(avg_loss)
        if early_stopper.early_stop is True:
           break

        # save the model if the performance gets improved
        if early_stopper.counter == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

    # Save the model in local:
    generator.save('generator_model_{}.h5'.format(datetime.date.today().strftime('%Y%m%d-%H%M')))

    # restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # Run the trained model on a few examples from the test dataset
    for i, (inp, tar) in enumerate(test_dataset.take(5)):
        generate_images(generator, inp, tar, out_fpath=f'{i}.png', is_training=False)


if __name__ == '__main__':
    main()

    