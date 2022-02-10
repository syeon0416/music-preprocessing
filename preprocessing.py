import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np

path = 'audio.wav'
sample_rate = 16000

x, sr = librosa.load('audio.wav')

waveform = librosa.load(path, sample_rate)[0]
melspec = librosa.feature.melspectrogram(waveform, sr=sample_rate, n_mels=128)
log_melspec = librosa.power_to_db(melspec, ref=np.max)
mfcc = librosa.feature.mfcc(S=log_melspec, n_fmcc=20)

delta2_mfcc = librosa.feature.delta(mfcc, order=2)

plt.figure(figsize=(12, 4))
librosa.display.specshow(delta2_mfcc)
plt.ylabel('MFCC coeffs')
plt.xlabel('Time')
plt.title('MFCC')
plt.colorbar()
plt.tight_layout()