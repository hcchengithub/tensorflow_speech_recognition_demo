import librosa
import peforth

# Load a wav file
filename = librosa.util.example_audio_file()
y, sr = librosa.load(filename)
peforth.ok(loc=locals(),glo=globals(),cmd='cr dup :> [0] constant loc :> [1] constant glo')

