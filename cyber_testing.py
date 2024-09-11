import os
import gdown
import pandas as pd
import time

# Path to the CSV file
file_path = '/Users/rahul.v/Downloads/LMS.csv'

# Load the CSV file
csv_data = pd.read_csv(file_path)

# Main folder to store all downloaded videos
main_save_folder = '126af'
os.makedirs(main_save_folder, exist_ok=True)

# Log file to keep track of failed downloads
log_file = os.path.join(main_save_folder, 'download_log.txt')

# Function to extract Google Drive file ID from the link
def extract_file_id(drive_link):
    if "drive.google.com" in drive_link:
        return drive_link.split("/d/")[1].split("/")[0]
    return None

# Iterate through each link in the CSV and download the file
for idx, row in csv_data.iterrows():
    video_link = row['Video Link']
    file_id = extract_file_id(video_link)
    
    # Add a delay between requests to avoid rate limits
    time.sleep(5)
    
    if file_id:
        # Construct the correct downloadable Google Drive URL with confirm for large files
        download_url = f'https://drive.google.com/uc?export=download&id={file_id}&confirm=t'
        
        # Create a unique filename for each video in the same folder
        output_file = os.path.join(main_save_folder, f'video_{idx + 1}.mp4')
        
        # Skip downloading if the file already exists
        if os.path.exists(output_file):
            print(f"Video {idx + 1} already downloaded, skipping...")
            continue
        
        # Download the video directly from the Google Drive link
        try:
            gdown.download(download_url, output_file, quiet=False)
            print(f"Downloaded video {idx + 1} to {output_file}")
        except Exception as e:
            print(f"Failed to download video {idx + 1}: {e}")
            with open(log_file, 'a') as log:
                log.write(f"Failed to download video {idx + 1}: {e}\n")
    else:
        print(f"Invalid Google Drive link for video {idx + 1}")
        with open(log_file, 'a') as log:
            log.write(f"Invalid Google Drive link for video {idx + 1}\n")
