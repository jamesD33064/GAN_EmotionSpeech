import librosa
import soundfile as sf

# Get example audio file
path_1 ='/Users/ash/Documents/vscode/GitHub/GAN_EmotionSpeech/dataset/audio/angry/OAF_back_angry.wav'
path_2 ='/Users/ash/Documents/vscode/GitHub/GAN_EmotionSpeech/dataset/audio/disgust/OAF_back_disgust.wav'

data_1, samplerate = sf.read(path_1, dtype='float64')
data_2, samplerate = sf.read(path_2, dtype='float64')

sf.write('stereo_file.wav', data_1, samplerate, subtype='PCM_24')