import librosa
import glob
import time


# instrument = 'string_acoustic'
# instrument = 'bass_acoustic'
instrument = 'vocal_synthetic'
epochs = 150
noise_level = 0.5
learning_rate = 0.0005

# Set the directory where the audio files are stored
# directory = './train/'
# directory = './train/vocal_synthetic/'
directory = f'./train/{instrument}/'
sample_rate = 16000

# Create a list of all the file paths
clean_files = glob.glob(directory + '*.wav')

def load_audio(file_path):
    audio, _ = librosa.load(file_path, sr=16000)
    return audio

import numpy as np

def add_noise(clean_audio, noise_level=0.5):
    noise = np.random.normal(0, noise_level, clean_audio.shape)
    noisy_audio = clean_audio + noise
    return noisy_audio

from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_wave_u_net():
    input_audio = Input(shape=(None, 1))

    # Encoder
    conv1 = Conv1D(16, 15, activation='relu', padding='same')(input_audio)
    pool1 = MaxPooling1D(4, padding='same')(conv1)
    
    conv2 = Conv1D(32, 15, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling1D(4, padding='same')(conv2)

    # Decoder
    up3 = UpSampling1D(4)(pool2)
    concat3 = Concatenate(axis=2)([up3, conv2])
    conv3 = Conv1D(16, 15, activation='relu', padding='same')(concat3)
    
    up4 = UpSampling1D(4)(conv3)
    concat4 = Concatenate(axis=2)([up4, conv1])
    conv4 = Conv1D(1, 15, activation='tanh', padding='same')(concat4)

    model = Model(inputs=input_audio, outputs=conv4)
    optimizer = Adam(learning_rate=learning_rate)
    # model.compile(optimizer='adam', loss='mse') # This will use the default learning rate of 0.001
    model.compile(optimizer=optimizer, loss='mse') # This will use the default learning rate of 0.001

    return model

model = build_wave_u_net()

model.summary()

X_train, Y_train = [], []

for file_path in clean_files:
    clean_audio = load_audio(file_path)
    noisy_audio = add_noise(clean_audio, noise_level)

    # Reshape for training, e.g., add channel dimension
    clean_audio = clean_audio.reshape(-1, 1)
    noisy_audio = noisy_audio.reshape(-1, 1)

    X_train.append(noisy_audio)
    Y_train.append(clean_audio)

# Convert to numpy arrays
X_train = np.array(X_train)
Y_train = np.array(Y_train)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# Record the current time before training
start_time = time.time()

# Assume X_train, Y_train are your training data and labels
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=20, validation_split=0.2)

# Record the current time after training
end_time = time.time()

# Compute the difference
training_time = end_time - start_time

model_name = f'{instrument}_epochs_{epochs}_nl_{noise_level}_lr_{learning_rate}'
# Print the result
print(f"{model_name} Model training took {training_time} seconds")

# model.save(f'./models/vocal_synthetic_epochs_{epochs}')
model.save(f'./models/{model_name}')

train_loss = history.history['loss']
val_loss = history.history['val_loss']

import matplotlib.pyplot as plt

plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
