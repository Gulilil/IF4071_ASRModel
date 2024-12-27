'''
BACA INI: https://stackoverflow.com/questions/76618750/how-to-convert-m4a-to-wav-in-python
Harus install ffmpeg dulu
'''

from pydub import AudioSegment
import os

m4a_dir = os.path.join(os.getcwd(), "data", "m4a")
custom_dir = os.path.join(os.getcwd(), "data", "custom")
files = os.listdir(m4a_dir)

for i in range(len(files)):
  filename = files[i]
  m4a_path = os.path.join(m4a_dir, filename)

  filename_wav = filename.replace(".m4a", ".wav")
  wav_path = os.path.join(custom_dir, filename_wav)

  sound = AudioSegment.from_file(m4a_path, format='m4a')
  file_handle = sound.export(wav_path, format='wav')