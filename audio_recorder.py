import pyaudio
import wave
import numpy as np
import keyboard
from scipy.signal import butter, lfilter

# Parameters
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Number of audio channels (1 for mono, 2 for stereo)
RATE = 44100  # Sampling rate (in Hz)
CHUNK = 1024  # Number of frames per buffer
RECORDING_FILENAME = "audio_rec.wav"

# Bandpass filter parameters
LOWCUT = 300.0  # Lower bound of the frequency range (300 Hz)
HIGHCUT = 3400.0  # Upper bound of the frequency range (3400 Hz)

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Create a stream
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Recording... Press 's' to stop recording.")

frames = []

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def smooth(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

try:
    while True:
        # Read audio data from the stream
        data = stream.read(CHUNK)
        
        # Convert data to numpy array for processing
        numpy_data = np.frombuffer(data, dtype=np.int16)
        
        # Apply a bandpass filter (retaining frequencies in human voice range)
        filtered_data = apply_bandpass_filter(numpy_data, LOWCUT, HIGHCUT, RATE)
        
        # Apply a simple smoothing filter to reduce noise spikes
        smoothed_data = smooth(filtered_data)
        
        # Apply a noise gate (optional: more refined than a simple threshold)
        noise_gate_threshold = 500
        gated_data = np.where(np.abs(smoothed_data) < noise_gate_threshold, 0, smoothed_data)
        
        # Convert numpy array back to bytes and store it
        frames.append(gated_data.astype(np.int16).tobytes())
        
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

    # Save the recorded audio to a file (WAV format)
    with wave.open(RECORDING_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

print(f"Recording saved to {RECORDING_FILENAME}")
