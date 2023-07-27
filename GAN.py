import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Example data: 1000 samples, each with 10 features, and 1 target label
num_samples = 1000
num_features = 10
X_real = np.random.rand(num_samples, num_features)

# Creating generator network
generator = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(num_features,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_features)  
])

# Creating discriminator network
# Binary Classification
discriminator = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(num_features,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  
])

# Combining generator and discriminator to GAN
gan = models.Sequential([generator, discriminator])

# Compiling discriminator 
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Compile the GAN with a custom loss function and optimizer
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Training the GAN
num_epochs = 1000
batch_size = 32

for epoch in range(num_epochs):
   # Generating data with generator
    noise = np.random.rand(num_samples, num_features)
    generated_data = generator.predict(noise)

    # Combine real and generated data for discriminator
    X_combined = np.vstack((X_real, generated_data))
    y_combined = np.vstack((np.ones((num_samples, 1)), np.zeros((num_samples, 1))))

    # Train discriminator 
    discriminator_loss = discriminator.train_on_batch(X_combined, y_combined)

    # Train  generator 
    noise = np.random.rand(num_samples, num_features)
    y_fake = np.ones((num_samples, 1))  # Set labels to 1 since we want the discriminator to think the data is real
    gan_loss = gan.train_on_batch(noise, y_fake)

    # Print the progress
    print(f"Epoch: {epoch+1}/{num_epochs}, Discriminator Loss: {discriminator_loss[0]}, GAN Loss: {gan_loss}")
  
# Save the generator model to an .h5 file after training
generator.save('trained_generator.h5')
