from pydub import AudioSegment
from openai import OpenAI
import openai
import os
import gradio as gr
import numpy as np
import librosa
from PIL import Image
import numpy as np
import os
import gradio as gr
import numpy as np
import librosa
from pathlib import Path


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
                    inputs=["text"],  
                    outputs=["audio", "text"],
                    cache_examples=True,
                   title="""
ناطق هو موقع مبتكر يعتمد على حلول الذكاء الاصطناعي، موجه خصيصًا إلى Arabicthon 2023""",)

def record_and_thank_you(word_recording, word):
    mine = f"Thank you for recording!"
    return f"Thank you for recording the word: {word}!"

demo_3 = gr.Interface(fn=record_and_thank_you, 
                         inputs=[gr.Microphone(), 'text'],
                         outputs="text")


os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = "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"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "vision-405518-2712c0cc1527.json"

def is_noise(audio, threshold_energy=-30):
    audio_array = np.array(audio.get_array_of_samples())
    energy = 10 * np.log10(np.mean(audio_array ** 2))
    return energy < threshold_energy





def audio_gen_1(message):
   

    flag = True

    while flag:
        client = OpenAI()

        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=message,
        )

        response.stream_to_file("output.mp3")

        y, sr = librosa.load('output.mp3')

        file_path = "output.mp3"

        current_path = os.getcwd()
        
        full_path = os.path.join(current_path, file_path)

        audio = AudioSegment.from_file(full_path, format="mp3")

        if not is_noise(audio):
            flag = False

    return (sr, y), "Doesn't seem good? Please record your own in the record page. Thank you!"





demo_1 = gr.Interface(fn=audio_gen_1, 
                    inputs=["text"],  
                    outputs=["audio", "text"],
                    cache_examples=True,
                    title="""
ناطق هو موقع مبتكر يعتمد على حلول الذكاء الاصطناعي، موجه خصيصًا إلى Arabicthon 2023""",)


demo = gr.TabbedInterface([demo_1, demo_2, demo_3], ["Faster model", "Better model", "Record!"])

demo.launch(inline = False, server_name="0.0.0.0")