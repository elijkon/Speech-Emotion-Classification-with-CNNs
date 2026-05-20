import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import boto3
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from prefect import task, flow

BUCKET_NAME = 'ravdess-pipeline-ekonkle-2026'
s3 = boto3.client('s3')
emotion_map = {1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}

def process_single_file(file_path, filename):
    try:
        parts = filename.replace('.wav', '').split("-")
        emotion = emotion_map[int(parts[2])]
        intensity = "normal" if int(parts[3]) == 1 else "strong"
        actor_id = int(parts[6])
        gender = "male" if actor_id % 2 != 0 else "female"
        
        y, sr = librosa.load(file_path, sr=None)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        image_name = filename.replace(".wav", ".png")
        temp_image_path = image_name 
        
        plt.figure(figsize=(2.5, 2.5))
        librosa.display.specshow(mel_spec_db, sr=sr, cmap="magma")
        plt.axis("off")
        plt.savefig(temp_image_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        s3_key = f"spectrograms/{emotion}/{image_name}"
        s3.upload_file(temp_image_path, BUCKET_NAME, s3_key)
        s3_url = f"s3://{BUCKET_NAME}/{s3_key}"
        
        os.remove(temp_image_path)
        return (filename, emotion, intensity, actor_id, gender, s3_url)
        
    except Exception as e:
        print(f"Failed on {filename}: {e}")
        return None

@task
def write_to_db(records):
    conn = sqlite3.connect('ravdess_metadata.db')
    cursor = conn.cursor()
    success_count = 0
    for record in records:
        if record:
            cursor.execute('''
            INSERT OR IGNORE INTO processed_audio 
            (filename, emotion, intensity, actor_id, gender, s3_image_url) 
            VALUES (?, ?, ?, ?, ?, ?)
            ''', record)
            success_count += 1
    conn.commit()
    conn.close()
    print(f"Logged {success_count} records to database.")

@flow(name="Dockerized_RAVDESS_Pipeline")
def run_pipeline(input_dir):
    files_to_process = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                files_to_process.append((os.path.join(root, file), file))
    
    print(f"Found {len(files_to_process)} files. Starting parallel workers...")
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(
            process_single_file, 
            [f[0] for f in files_to_process], 
            [f[1] for f in files_to_process]
        ))
    
    write_to_db(results)

if __name__ == "__main__":
    run_pipeline("audio_speech_actors_01-24")