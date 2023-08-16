from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import load_model
import numpy as np
import librosa
import glob

# Load the model
loaded_model = load_model('./models/wave_u_net_model_epochs_50')

# Set the directory where the audio files are stored
directory = './test/vocal_synthetic/'

# Create a list of all the file paths
clean_files = glob.glob(directory + '*.wav')

def load_audio(file_path):
    audio, _ = librosa.load(file_path, sr=None)
    return audio

def add_noise(clean_audio, noise_level=0.05):
    noise = np.random.normal(0, noise_level, clean_audio.shape)
    noisy_audio = clean_audio + noise
    return noisy_audio

X_test, Y_test = [], []

for file_path in clean_files:
    clean_audio = load_audio(file_path)
    noisy_audio = add_noise(clean_audio)

    # Reshape for training, e.g., add channel dimension
    clean_audio = clean_audio.reshape(-1, 1)
    noisy_audio = noisy_audio.reshape(-1, 1)

    X_test.append(noisy_audio)
    Y_test.append(clean_audio)

# Convert to numpy arrays
X_test = np.array(X_test)
Y_test = np.array(Y_test)

def evaluate_model(model, X_test, Y_test):
    # Predicting the denoised audio
    Y_pred = model.predict(X_test)

    # Flatten the arrays
    Y_test_flatten = Y_test.reshape(-1)
    Y_pred_flatten = Y_pred.reshape(-1)

    # Calculating Mean Squared Error
    mse = mean_squared_error(Y_test_flatten, Y_pred_flatten)

    # Calculating Signal-to-Noise Ratio
    snr = 10 * np.log10(np.mean(np.square(Y_test)) / np.mean(np.square(Y_test - Y_pred)))

    print('Mean Squared Error:', mse)
    print('Signal-to-Noise Ratio:', snr)

    return mse, snr, Y_pred

mse, snr, Y_pred = evaluate_model(loaded_model, X_test, Y_test)

sample_rate = 16000

import soundfile as sf

for i, audio in enumerate(X_test):
    sf.write(f'./evaluation/vocal_synthetic/noisy_audio_{i}.wav', audio, sample_rate)

for i, audio in enumerate(Y_pred):
    sf.write(f'./evaluation/vocal_synthetic/denoised_audio_{i}.wav', audio, sample_rate)
