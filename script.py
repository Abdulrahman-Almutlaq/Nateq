from pydub import AudioSegment
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
import openai
import os
import gradio as gr
import numpy as np
import librosa
from google.cloud.vision_v1 import types
from google.cloud import vision
from PIL import Image
import numpy as np
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
import openai
import os
import gradio as gr
import numpy as np
import librosa


def is_noise(audio, threshold_energy=-30):
    """
    Check if an audio segment is primarily noise.

    Parameters:
        audio (AudioSegment): The input audio segment.
        threshold_energy (float): Energy threshold in decibels. Defaults to -30 dB.

    Returns:
        bool: True if the audio is noise, False otherwise.
    """
    # Convert audio to numpy array
    audio_array = np.array(audio.get_array_of_samples())

    # Calculate the energy of the audio
    energy = 10 * np.log10(np.mean(audio_array ** 2))

    # Compare the energy to the threshold
    return energy < threshold_energy







os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = "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"

def audio_gen_2(message, image=None):
    client = OpenAI()

    response = client.audio.speech.create(
        model="tts-1-hd",
        voice="alloy",
        input=message,
    )
    
    response.stream_to_file("output.mp3")
    
    y, sr = librosa.load('output.mp3')
    
    return (sr, y)

demo_2 = gr.Interface(fn=audio_gen_2, 
                    inputs=["text", gr.Image()],  
                    outputs=["audio", "text"],
                    cache_examples=True,
                   title="""
ناطق هو موقع مبتكر يعتمد على حلول الذكاء الاصطناعي، موجه خصيصًا إلى Arabicthon 2023""",)


import gradio as gr

def record_and_thank_you(word_recording, word):
    mine = f"Thank you for recording!"
    return f"Thank you for recording the word: {word}!"

demo_3 = gr.Interface(fn=record_and_thank_you, 
                         inputs=[gr.Microphone(text="Say a word:"), 'text'],
                         outputs="text")


os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = "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"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "vision-405518-2712c0cc1527.json"

def is_noise(audio, threshold_energy=-30):
    audio_array = np.array(audio.get_array_of_samples())
    energy = 10 * np.log10(np.mean(audio_array ** 2))
    return energy < threshold_energy

from google.cloud import vision_v1 as vision
from google.cloud.vision_v1 import types
from PIL import Image
import numpy as np

# ... (your other imports)

def perform_ocr(image_path):
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    # Specify the language explicitly as Arabic
    image_context = types.ImageContext(language_hints=["ar"])

    response = client.text_detection(image=image, image_context=image_context)

    # Extracting text from response
    texts = response.text_annotations

    # Check confidence level
    confidence = texts[0].confidence if texts else 0.0

    # Filter out low-confidence results
    if confidence >= 0.7:
        extracted_text = texts[0].description
    else:
        extracted_text = "Low confidence result"

    return extracted_text


def audio_gen_1(message, image=None):
    # If an image is provided, use its content to update the message
    if image is not None and np.any(image):
        # Convert the NumPy array to an image
        pil_image = Image.fromarray((image * 255).astype(np.uint8))

        # Save the image as a temporary file
        temp_image_path = "temp_image.png"
        pil_image.save(temp_image_path)

        # Use Google Cloud Vision API to extract text from the image
        extracted_text = perform_ocr(temp_image_path)
        
        print(extracted_text)
        # Update the message with the extracted text
        message = extracted_text

        # Remove the temporary image file
        os.remove(temp_image_path)

    boolean = True
    
    while boolean:
        client = OpenAI()

        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=message,
        )

        response.stream_to_file("output.mp3")

        y, sr = librosa.load('output.mp3')

        file_path = "output.mp3"
        audio = AudioSegment.from_file(file_path, format="mp3")

        if not is_noise(audio):
            boolean = False

    return (sr, y), "Doesn't seem good? Please record your own in the record page. Thank you!"





demo_1 = gr.Interface(fn=audio_gen_1, 
                    inputs=["text", gr.Image()],  
                    outputs=["audio", "text"],
                    cache_examples=True,
                    title="""
ناطق هو موقع مبتكر يعتمد على حلول الذكاء الاصطناعي، موجه خصيصًا إلى Arabicthon 2023""",)


demo = gr.TabbedInterface([demo_1, demo_2, demo_3], ["Faster model", "Better model", "Record!"])

demo.launch(inline = False, server_name="0.0.0.0")