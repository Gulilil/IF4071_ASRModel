from transformers import pipeline

import pyaudio
import wave
import audioop
import time
 
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
# RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "file.wav"
HOP_SIZE                = CHUNK//2
PERIOD_SIZE_IN_FRAME    = HOP_SIZE
METHOD                  = "default"
SILENCE_TIMER = 3

silence_threshold = 1000
audio = pyaudio.PyAudio()
all_transcriptions = []
whisper = pipeline(model="NY7y32/whisper-tiny-id")
# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

while True : 
    # print ("recording...")
    # frames = []

    # for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    rms = audioop.rms(data, 2)
    if rms > silence_threshold :
        print("Recording")
        frames = []
        last_silence_time = time.time()
        while time.time() - last_silence_time < SILENCE_TIMER :
            data = stream.read(CHUNK)
            frames.append(data)
            rms = audioop.rms(data, 2)
            if rms >= silence_threshold/2 :
                last_silence_time = time.time()
    
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

        print("transcripting")
        transcription = whisper(WAVE_OUTPUT_FILENAME)
        # all_transcriptions.extend(transcription["text"].split(" ")) 
        # if len(all_transcriptions) > 20 :
        #     del all_transcriptions[0:10]
        print(transcription["text"])

# stop Recording
# stream.stop_stream()
# stream.close()
# audio.terminate()