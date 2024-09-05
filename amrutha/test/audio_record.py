import pyaudio
import wave
import numpy as np
import keyboard

# Parameters
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Number of audio channels (1 for mono, 2 for stereo)
RATE = 44100  # Sampling rate (in Hz)
CHUNK = 1024  # Number of frames per buffer
RECORDING_FILENAME = "audio_rec.wav"

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Create a stream
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Recording... Press 's' to stop recording.")

frames = []

try:
    while True:
        # Read audio data from the stream
        data = stream.read(CHUNK)
        
        # Convert data to numpy array for processing
        numpy_data = np.frombuffer(data, dtype=np.int16)
        
        # Simple noise reduction
        # Apply a threshold to remove low amplitude signals (noise)
        numpy_data = np.where(np.abs(numpy_data) < 500, 0, numpy_data)
        
        # Convert numpy array back to bytes and store it
        frames.append(numpy_data.tobytes())
        
        # Check if 's' is pressed to stop recording
        if keyboard.is_pressed('s'):
            print("Stopping...")
            break
finally:
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    
    # Terminate PyAudio
    audio.terminate()

    # Save the recorded audio to a file
    with wave.open(RECORDING_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

print(f"Recording saved to {RECORDING_FILENAME}")
