import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(X_train, y_train), (_, _) = mnist.load_data()

# Preprocess the data
X_train = X_train.astype('float32') / 255.0  # Normalize the pixel values
X_train = X_train.reshape(X_train.shape[0], 784)  # Flatten the images
y_train = to_categorical(y_train, 10)  # Convert labels to one-hot encoding

# Define the generator
generator = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(100,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Define the discriminator
discriminator = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the generator and discriminator
generator.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the generator and discriminator
for i in range(100):
    # Sample a random noise vector
    noise = tf.random.normal(shape=(X_train.shape[0], 100))

    # Generate a new data sample using the generator
    generated_data = generator.predict(noise)

    # Train the discriminator on the generated data
    discriminator.fit(generated_data, np.zeros((X_train.shape[0], 1)), epochs=1)

    # Train the generator on the discriminator's output
    generator.fit(noise, y_train, epochs=1)
