import librosa.display
import numpy as np
import matplotlib.pyplot as plt

file_path='dataset/audio/angry/OAF_back_angry.wav'

# y, sr = librosa.load(file_path)
# plt.figure()

# D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
# plt.subplot(2, 1, 1)
# librosa.display.specshow(D, y_axis='linear')
# plt.colorbar(format='%+2.0f dB')
# plt.title('線性頻率功率譜')

# plt.subplot(2, 1, 2)
# librosa.display.specshow(D, y_axis='log')
# plt.colorbar(format='%+2.0f dB')
# plt.title('對數頻率功率譜')

# plt.show()


y, sr = librosa.load(file_path)
# 方法一：時間序列
print(librosa.feature.melspectrogram(y=y, sr=sr))
# 方法二：短時距傅立葉變換
D = np.abs(librosa.stft(y)) ** 2  # 短時距傅立葉變換
S = librosa.feature.melspectrogram(S=D)  # 使用stft頻譜求Mel頻譜

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()