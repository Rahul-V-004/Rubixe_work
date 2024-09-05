import speech_recognition as sr
from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Function to convert audio to text
def audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand the audio."
        except sr.RequestError:
            return "Error in the Google API request."

# Function for BERT-based similarity evaluation
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

# Main function to handle audio input and evaluate similarity
def evaluate_student_answer(audio_file, key_answer):
    # Convert audio to text
    student_answer = audio_to_text(audio_file)
    print(f"Transcribed Student Answer: {student_answer}")

    # If the transcription was successful
    if "Could not understand the audio" not in student_answer and "Error in the Google API request" not in student_answer:
        # Perform BERT similarity evaluation
        similarity_score = bert_similarity_evaluation(key_answer, student_answer)
        print(f"Similarity Score: {similarity_score}")

        # Check if similarity is above 60%
        if similarity_score >= 0.60:
            print("The answer given by the student is correct.")
        else:
            print("The answer given by the student is incorrect.")
    else:
        print("Transcription failed, unable to evaluate the answer.")

# Example Usage
audio_file = '/Users/rahul.v/Downloads/audio_rec.mp3'  # Replace with the path to the audio file
key_answer = "Machine learning is a subset of artificial intelligence that allows computers to learn from data and improve their performance on a specific task without being explicitly programmed. Insteadof being hand-coded with rules, the algorithm learns from examples."

evaluate_student_answer(audio_file, key_answer)