import os
import random
from playsound import playsound  # You may need to install this module

# Install playsound using: pip install playsound

# Set the folder paths
folder1 = '/Users/rahul.v/Downloads/dlq'
folder2 = '/Users/rahul.v/Downloads/mlq'
folder3 = '/Users/rahul.v/Downloads/statq'

# Dictionary to keep track of used files in each folder
used_files = {
    folder1: set(),
    folder2: set(),
    folder3: set()
}

# Initialize current_folder to None
current_folder = None

def get_random_audio(exclude_folder=None):
    # List of all folders
    folders = [folder1, folder2, folder3]
    
    # Exclude the current folder to ensure next selection is from a different folder
    if exclude_folder:
        folders = [folder for folder in folders if folder != exclude_folder]
    
    # Randomly select a new folder
    new_folder = random.choice(folders)
    
    # Get all audio files in the selected folder
    all_files = [f for f in os.listdir(new_folder) if f.endswith(('.mp3', '.wav'))]
    
    # Get the set of used files in this folder
    used = used_files[new_folder]
    
    # Determine available files by excluding used files
    available_files = list(set(all_files) - used)
    
    # If all files have been used, reset the used_files set for this folder
    if not available_files:
        used_files[new_folder] = set()
        available_files = all_files
    
    # Select a random audio file from available files
    selected_file = random.choice(available_files)
    
    # Add the selected file to the used_files set
    used_files[new_folder].add(selected_file)
    
    return new_folder, selected_file

def main():
    global current_folder
    
    print("Welcome to the Random Audio Quiz!")
    while True:
        user_input = input("Do you want to hear a question? (yes/no): ").strip().lower()
        
        if user_input == 'yes':
            try:
                folder, audio_file = get_random_audio(exclude_folder=current_folder)
                audio_path = os.path.join(folder, audio_file)
                print(f"Playing audio: {audio_file} from folder: {os.path.basename(folder)}")
                
                # Play the audio file
                playsound(audio_path)
                
                # Update the current_folder
                current_folder = folder
            except Exception as e:
                print(f"An error occurred: {e}")
        elif user_input == 'no':
            print("Thank you! Goodbye.")
            break
        else:
            print("Invalid input. Please type 'yes' or 'no'.")

if __name__ == "__main__":
    main()
