from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
from pydub import AudioSegment

def convert_audio_to_text(audio_file):
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Convert audio file to WAV format if it's not
    audio = AudioSegment.from_file(audio_file)
    audio.export("converted_audio.wav", format="wav")

    # Load the audio file
    with sr.AudioFile("converted_audio.wav") as source:
        audio_data = recognizer.record(source)
        try:
            # Convert audio to text
            text = recognizer.recognize_google(audio_data)
            print(f"Recognized Text: {text}")
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand the audio.")
            return None
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service.")
            return None

def bert_similarity_evaluation(key_answer, student_answer):
    # Load pre-trained BERT model and tokenizer
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the input text and get embeddings
    def get_embedding(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()

    # Get embeddings for both answers
    key_embedding = get_embedding(key_answer)
    student_embedding = get_embedding(student_answer)

    # Compute cosine similarity
    similarity_score = cosine_similarity(key_embedding.unsqueeze(0), student_embedding.unsqueeze(0))
    
    return similarity_score[0][0]

# Example Usage
key_answer = "Artificial intelligence is the simulation of human intelligence by machines."

# Path to the student's audio answer (e.g., 'student_answer.mp3')
audio_file = '/Users/rahul.v/Downloads/audio_rec.mp3'

# Convert the audio answer to text
student_answer_text = convert_audio_to_text(audio_file)

if student_answer_text:
    # Evaluate the similarity between the key answer and the student's answer
    similarity_score = bert_similarity_evaluation(key_answer, student_answer_text)
    print(f"BERT Similarity Score: {similarity_score}")

    # Check if the similarity is above the threshold (e.g., 60%)
    if similarity_score > 0.6:
        print("The answer given by the student is correct.")
    else:
        print("The answer given by the student is incorrect.")
    
    # Print the student's answer in text form
    print(f"Student's Answer (Text): {student_answer_text}")