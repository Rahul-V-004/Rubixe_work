import whisper
import os
import subprocess

# Load the Whisper model
model = whisper.load_model("base")

def extract_audio_from_video(video_path, output_audio_path):
    """Extracts audio from a video file using ffmpeg."""
    command = f"ffmpeg -i \"{video_path}\" -ar 16000 -ac 1 -f wav \"{output_audio_path}\""
    subprocess.call(command, shell=True)

def transcribe_audio(audio_path):
    """Transcribes audio using Whisper model."""
    result = model.transcribe(audio_path)
    return result["text"]

def process_videos_in_folder(input_folder, output_folder):
    """Extracts text from all video files in a folder and saves each result in the specified output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv')):
            video_path = os.path.join(input_folder, file_name)
            audio_path = os.path.join(input_folder, f"{os.path.splitext(file_name)[0]}.wav")

            print(f"Processing video: {video_path}")
            extract_audio_from_video(video_path, audio_path)
            text = transcribe_audio(audio_path)

            # Save the text to the specified output folder
            output_text_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_transcription.txt")
            with open(output_text_file, "w") as f:
                f.write(text)
            print(f"Saved transcription for {file_name} to {output_text_file}")

            # Remove the audio file after processing to save space
            os.remove(audio_path)

# Example usage:
input_folder = "cvid"  # Replace with the folder containing your videos
output_folder = "audio"  # Replace with the folder where you want to save the transcriptions
process_videos_in_folder(input_folder, output_folder)
