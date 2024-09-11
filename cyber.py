import csv
import os
import time
import subprocess
import logging
import speech_recognition as sr
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_driver():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")
    prefs = {"download.default_directory": os.getcwd()}
    chrome_options.add_experimental_option("prefs", prefs)
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        logging.info("Chrome driver set up successfully")
        return driver
    except Exception as e:
        logging.error(f"Failed to set up Chrome driver: {e}")
        raise

def download_video(driver, url):
    try:
        driver.get(url)
        download_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Download')]"))
        )
        download_button.click()
        
        # Wait for download to complete (adjust time as needed)
        time.sleep(30)
        
        file_name = url.split('/')[-1] + '.mp4'
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"Downloaded file not found: {file_name}")
        
        logging.info(f"Video downloaded successfully: {file_name}")
        return file_name
    except TimeoutException:
        logging.error(f"Timeout waiting for download button on {url}")
        raise
    except NoSuchElementException:
        logging.error(f"Download button not found on {url}")
        raise
    except Exception as e:
        logging.error(f"Error downloading video from {url}: {e}")
        raise

def extract_audio(video_path, audio_path):
    try:
        command = f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 44100 -vn {audio_path}"
        subprocess.call(command, shell=True)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Extracted audio file not found: {audio_path}")
        logging.info(f"Audio extracted successfully: {audio_path}")
    except Exception as e:
        logging.error(f"Error extracting audio from {video_path}: {e}")
        raise

def audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        logging.info("Audio transcribed successfully")
        return text
    except sr.UnknownValueError:
        logging.warning("Speech recognition could not understand the audio")
        return "Speech recognition could not understand the audio"
    except sr.RequestError as e:
        logging.error(f"Could not request results from speech recognition service: {e}")
        return f"Could not request results from speech recognition service; {e}"
    except Exception as e:
        logging.error(f"Error transcribing audio {audio_path}: {e}")
        raise

def process_video(driver, url, output_text_path):
    try:
        video_path = download_video(driver, url)
        audio_path = video_path.rsplit('.', 1)[0] + '.wav'
        
        extract_audio(video_path, audio_path)
        
        text = audio_to_text(audio_path)
        
        with open(output_text_path, 'w') as f:
            f.write(text)
        
        # Clean up
        os.remove(video_path)
        os.remove(audio_path)
        logging.info(f"Transcription saved to {output_text_path}")
    except Exception as e:
        logging.error(f"Error processing video from {url}: {e}")

def process_csv(csv_path):
    driver = setup_driver()
    try:
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                url = row[0]  # Assuming the URL is in the first column
                output_text_path = f"transcript_{url.split('/')[-1]}.txt"
                try:
                    process_video(driver, url, output_text_path)
                except Exception as e:
                    logging.error(f"Failed to process {url}: {e}")
                    continue
    except Exception as e:
        logging.error(f"Error processing CSV file {csv_path}: {e}")
    finally:
        driver.quit()
        logging.info("Chrome driver closed")

if __name__ == "__main__":
    csv_path = '/Users/rahul.v/Downloads/Cyber sec Links - Sheet1.csv'  # Replace with your CSV file path
    process_csv(csv_path)