from pydub import AudioSegment
from transformers import pipeline
import gradio as gr
import ffmpeg
import soundfile as sf
import numpy as np


def process_audio_length(audio_path):
    audio = AudioSegment.from_file(audio_path)
    audio_length_ms = len(audio)  
    chunks = []
    target_length_ms=30000
    if audio_length_ms < target_length_ms:
        padding = AudioSegment.silent(duration=target_length_ms - audio_length_ms)
        padded_audio = audio + padding
        chunks.append(padded_audio)
    else:
        for start in range(0, audio_length_ms, target_length_ms):
            end = min(start + target_length_ms, audio_length_ms)
            chunk = audio[start:end]
            if len(chunk) < target_length_ms:
                padding = AudioSegment.silent(duration=target_length_ms - len(chunk))
                chunk += padding
            chunks.append(chunk)
    
    return chunks

def save_chunks(chunks):
    import os
    output_folder="audio_chunks"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    chunk_files = []
    for i, chunk in enumerate(chunks):
        chunk_name = f"{output_folder}/chunk_{i}.wav"
        chunk.export(chunk_name, format="wav")
        chunk_files.append(chunk_name)
    
    return chunk_files

pipe = pipeline(task="automatic-speech-recognition", model="alisharifi/whisper-farsi")

def extract_audio_from_video(video_file):
    output_audio = "output_audio.wav"
    stream = ffmpeg.input(video_file)
    stream = ffmpeg.output(stream, output_audio, ac=1, ar="16000")
    ffmpeg.run(stream, overwrite_output=True)
    return output_audio

def process_audio(audio):
    if audio is None:
        return "لطفاً صدایی ضبط کنید."    
    chunks = process_audio_length(audio)  
    chunk_files = save_chunks(chunks)
    transcriptions = []
    for chunk_file in chunk_files:
        transcription = pipe(chunk_file)["text"] 
        transcriptions.append(transcription)
    
    return " ".join(transcriptions)

def process_video(video_file):
    audio_path = extract_audio_from_video(video_file)
    return process_audio(audio_path)

def process_audio_file(audio_file):
    return process_audio(audio_file)

def process_microphone(audio_data):
    print(f"Audio data type: {type(audio_data)}")
    print(f"Audio data content: {audio_data}")
    
    if audio_data is None:
        return "هیچ صدایی ضبط نشد. لطفاً دوباره امتحان کنید."
    
    if not isinstance(audio_data, tuple) or len(audio_data) != 2:
        return f"فرمت داده صوتی نادرست است: {type(audio_data)}"
    
    audio_array, sample_rate = audio_data  
    if not isinstance(audio_array, np.ndarray):
        return f"داده صوتی نادرست است: {type(audio_array)}"
    
    if audio_array.ndim == 1:
        audio_array = audio_array[:, np.newaxis]
    
    audio_path = "recorded_audio.wav"
    sf.write(audio_path, audio_array, sample_rate)
    return process_audio(audio_path)


with gr.Blocks() as demo:
    gr.Markdown("## سامانه تبدیل گفتار به متن")
   
    with gr.Tab("آپلود ویدئو"):
        video_input = gr.Video(label="آپلود فایل ویدئو")
        video_output = gr.Textbox(label="متن استخراج شده")
        video_button = gr.Button("پردازش")
        video_button.click(process_video, inputs=video_input, outputs=video_output)
   
    with gr.Tab("آپلود فایل صوتی"):
        audio_input = gr.Audio(label="آپلود فایل صوتی", type="filepath")
        audio_output = gr.Textbox(label="متن استخراج شده")
        audio_button = gr.Button("پردازش")
        audio_button.click(process_audio_file, inputs=audio_input, outputs=audio_output)

    with gr.Tab("ضبط صدا"):
        mic_input = gr.Audio(sources="microphone", type="filepath", label="ضبط صدا")
        mic_output = gr.Textbox(label="متن استخراج شده")
        mic_button = gr.Button("پردازش")
        mic_button.click(process_audio_file, inputs=mic_input, outputs=mic_output)

demo.launch(share=True)
