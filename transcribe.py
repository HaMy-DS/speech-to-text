import numpy as np
import librosa
import json
import argparse
from transformers import pipeline

def librosa_suppress_noise(input_wav_path, noise_reduction_factor=0.15):
    try:
        data, rate = librosa.load(input_wav_path, sr=None)
        print(f"Loaded audio file: {input_wav_path}")
        print(f"Sample rate: {rate}, Audio length: {data.shape}")

        stft_data = librosa.stft(data, n_fft=1024, hop_length=512)
        magnitude, phase = librosa.magphase(stft_data)
        noise_profile = np.mean(magnitude[:, :10], axis=1, keepdims=True)
        magnitude_denoised = np.maximum(magnitude - noise_reduction_factor * noise_profile, 0)
        stft_denoised = magnitude_denoised * phase
        data_denoised = librosa.istft(stft_denoised, hop_length=512)

        return data_denoised, rate
    except Exception as e:
        print(f"An error occurred during noise suppression: {e}")
        return None, None

def extract_transcript(data: np.ndarray, rate: int, output_file: str) -> None:
    target_rate = 16000
    if rate != target_rate:
        data = librosa.resample(data, orig_sr=rate, target_sr=target_rate)

    chunk_duration = 25  
    chunk_size = chunk_duration * target_rate
    num_chunks = int(np.ceil(len(data) / chunk_size))

    transcriber = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        return_timestamps=True
    )

    chunks = []
    current_time = 0.0
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(data))
        chunk_data = data[start:end]

        input_audio = {
            "array": chunk_data,
            "sampling_rate": target_rate
        }

        result = transcriber(input_audio)

        for item in result.get('chunks', []):
            start_time, end_time = item.get('timestamp', (0, 0))
            start_time += current_time
            end_time += current_time
            start_hms = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{start_time % 60:.2f}"
            end_hms = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{end_time % 60:.2f}"

            chunks.append({
                'timestamp': (start_hms, end_hms),
                'text': item.get('text', '')
            })

        current_time += (end - start) / target_rate

    transcript_data = {'chunks': chunks}

    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(transcript_data, json_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio and save as JSON.")
    parser.add_argument("audio_path", type=str, help="Path to the input WAV file.")
    parser.add_argument("json_output", type=str, help="Path to save the JSON output.")
    args = parser.parse_args()

    librosa_audio, rate = librosa_suppress_noise(args.audio_path)
    
    if librosa_audio is not None:
        if len(librosa_audio.shape) > 1 and librosa_audio.shape[1] == 2:
            librosa_audio = np.mean(librosa_audio, axis=1)
        
        extract_transcript(librosa_audio, rate, args.json_output)

