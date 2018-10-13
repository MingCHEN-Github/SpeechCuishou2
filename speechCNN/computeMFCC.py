from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import librosa
# Import the Preset class
from presets import Preset

# To use presets, we'll make a dummy import of librosa
# and the display submodule here.
import librosa as _librosa
import librosa.display as _display
filename = 'C:\\Users\\zkycs3\\Desktop\\audio\\audio files\\AnswerGroup5\\wohuitongzhide\\wohuitongzhide035.wav'
y, sr = librosa.load(filename)
# Generate a Mel spectrogram:
M = librosa.feature.melspectrogram(y=y)

# Of course, you can still override the new default manually, e.g.:
M_highres = librosa.feature.melspectrogram(y=y, hop_length=512)

# And plot the results
plt.figure(figsize=(6, 6))
ax = plt.subplot(2, 1, 1)
specshow1=librosa.display.specshow(librosa.power_to_db(M, ref=np.max),
                                   y_axis='mel', x_axis='time')
plt.title('44100/1024/4096')
plt.subplot(2, 1, 2, sharex=ax, sharey=ax)
specshow2=librosa.display.specshow(librosa.power_to_db(M_highres, ref=np.max),
                         hop_length=512,
                         y_axis='mel', x_axis='time')

plt.tight_layout()
#plt.colorbar(format='%+2.0f dB')
plt.show()
