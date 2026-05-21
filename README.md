# End-to-End Speech Emotion Classification Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED) ![AWS S3](https://img.shields.io/badge/AWS-S3-FF9900) ![Prefect](https://img.shields.io/badge/Prefect-Orchestration-0052FF) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C) ![WandB](https://img.shields.io/badge/Weights_&_Biases-Logging-FFBE00)

## Overview
This project is an end-to-end data engineering and deep learning pipeline that classifies human emotion from raw audio files. 

Moving beyond traditional recurrent audio models, this system translates the problem into a Computer Vision task. It orchestrates a parallelized ETL pipeline to convert raw audio into Mel-spectrograms, loads the processed assets into an AWS S3 Data Lake, and trains a Convolutional Neural Network (CNN) via PyTorch to classify the emotional state of the speaker.

## 🏗️ Architecture & Tech Stack

This repository demonstrates proficiency across the modern data and machine learning stack:

* **Data Orchestration & ETL:** Prefect, Python `ProcessPoolExecutor` (Multiprocessing)
* **Audio Processing:** Librosa
* **Cloud Storage:** AWS S3 (via Boto3)
* **Database / Metadata Management:** SQLite
* **Deep Learning:** PyTorch, Torchvision
* **Experiment Tracking:** Weights & Biases (WandB)
* **Infrastructure:** Docker

### The Pipeline Workflow

1. **Extract:** Ingests raw `.wav` audio files from the RAVDESS dataset.
2. **Transform (Parallelized):** Utilizes multi-core processing to efficiently extract 128x128 Mel-spectrograms from the audio signal. Metadata (emotion, intensity, actor ID, gender) is dynamically parsed from the filename nomenclature.
3. **Load:** * Visualized spectrograms are pushed to an **AWS S3 bucket** for scalable storage.
   * Audio metadata and S3 URIs are logged into a **SQLite relational database** for easy querying and dataset splitting.
4. **Train:** A custom PyTorch CNN model consumes the local NVMe-synced dataset, tracked thoroughly with WandB for loss/accuracy metrics across epochs.

## 🚀 Key Features

* **Dockerized Environment:** The entire ETL pipeline is containerized using `python:3.10-slim` with necessary OS-level audio dependencies (`libsndfile1`) for seamless cross-platform execution.
* **Concurrent Processing:** Drastically reduces data transformation time by utilizing Python's `ProcessPoolExecutor` to process audio files in parallel.
* **Robust Orchestration:** Tasks are wrapped in **Prefect** (`@flow`, `@task`) to ensure observability, logging, and fault tolerance during the ETL process.
* **Scalable Storage Design:** Decouples heavy image assets (AWS S3) from relational metadata (SQLite), simulating a production-grade data lake architecture.

## ⚙️ How to Run

### 1. Environment Setup
Create a `.env` file or configure your AWS credentials to allow Boto3 to authenticate with S3:

```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=your_region
```

### 2. Run the ETL Pipeline (Docker)
Ensure your raw audio files are located in the `audio_speech_actors_01-24/` directory.

Build and run the container to execute the database setup and data pipeline:

```bash
docker build -t emotion-pipeline .
docker run --env-file .env emotion-pipeline
```
*This will automatically initialize the `ravdess_metadata.db` and begin processing and uploading files to S3.*

### 3. Model Training
Once the data is processed and synced to your local `dataset/spectrograms` directory:

1. Open `DL_miniHackathon.ipynb`.
2. Ensure you have an active GPU environment (CUDA/MPS supported).
3. Log in to Weights & Biases when prompted.
4. Run all cells to initialize the CNN and begin training for 50 epochs.

## 📊 Model Performance
The custom CNN features 4 convolutional layers (32 -> 64 -> 128 -> 256 channels) with a global average pooling layer and dropout regularization (0.25) to prevent overfitting. Training progression, validation accuracy, and loss metrics are automatically logged and visualized in your WandB dashboard.

## 🔮 Future Enhancements
* **Data Augmentation:** Implement dynamic time-masking or frequency-masking during the Librosa transformation to improve model generalizability.
* **CI/CD Integration:** Add GitHub Actions to automatically lint code, build the Docker image, and run unit tests on pipeline tasks.
* **Cloud Compute:** Migrate the PyTorch training loop to AWS SageMaker or EC2 instances to fully utilize the S3 data lake.

## Dataset Credits
"The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)" by Livingstone & Russo is licensed under CC BY-NA-SC 4.0.