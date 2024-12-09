import tensorflow as tf
import matplotlib.pyplot as plt
import os
from vae import Encoder, Decoder, VAE

# Global Parameters
image_dir = r'C:/Users/youss/anime_faces/images'  # Directory containing images
image_size = 64
latent_dim = 512
batch_size = 64
epochs = 50

# 1. Preprocess Function
def preprocess(image_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (image_size, image_size))
    image = tf.cast(image, tf.float32) / 255.0
    return image

# 2. Load Dataset
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(".jpg")]
dataset = tf.data.Dataset.from_tensor_slices(image_paths)
dataset = dataset.map(preprocess).shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Visualize a Batch
def visualize_batch(dataset, num_images=25):
    batch = next(iter(dataset.unbatch().batch(num_images)))
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(batch[i].numpy())
        ax.axis("off")
    plt.show()

visualize_batch(dataset)

# 3. Instantiate Models
encoder = Encoder(latent_dim)
decoder = Decoder(latent_dim)
vae = VAE(encoder, decoder)

# 4. Training
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

def compute_loss(batch, vae):
    reconstructed, mu, log_var = vae(batch)
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(batch - reconstructed), axis=[1, 2, 3]))
    kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=-1)
    return tf.reduce_mean(reconstruction_loss + kl_loss)

@tf.function
def train_step(batch, vae, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(batch, vae)
    grads = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(grads, vae.trainable_variables))
    return loss

for epoch in range(epochs):
    epoch_loss = 0
    for step, batch in enumerate(dataset):
        loss = train_step(batch, vae, optimizer)
        epoch_loss += loss
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss.numpy():.4f}")










