# DeepLearning_MiniHackathon

![Project Banner](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C) ![WandB](https://img.shields.io/badge/Weights_&_Biases-Logging-FFBE00) ![HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-yellow)

## What is this?
Team KEM (Kritan, Elijah, and Mayur) built this project for a Deep Learning Mini-Hackathon.

The main idea was to figure out if a computer can tell how someone is feeling just by listening to them. Usually, people use complex audio models (like RNNs) for this, but we decided to try a Computer Vision approach instead. We turned the sound clips into pictures called **Spectrograms** and then trained a standard image classifier (CNN) to look at the picture and guess the emotion.

## The Data
We used the **RAVDESS** dataset.

* **Original Audio:** We got the raw audio files from Kaggle: [RAVDESS Emotional Speech Audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio?resource=download)
* **Our Processed Data:** We converted that audio into 128x128 grayscale spectrograms and hosted them on Hugging Face here: [elijkon/DL_Spectrograms](https://huggingface.co/datasets/elijkon/DL_Spectrograms)

### Decoding the Filenames
The files have long names like `03-01-06-01-02-01-12.wav`. Here is how to read them:

| Position | Attribute | Codes |
| :--- | :--- | :--- |
| 1 | Modality | 01=Full-AV, 02=Video, 03=Audio |
| 2 | Vocal Channel | 01=Speech, 02=Song |
| 3 | Emotion | 01=Neutral, 02=Calm, 03=Happy, 04=Sad, 05=Angry, 06=Fearful, 07=Disgust, 08=Surprised |
| 4 | Intensity | 01=Normal, 02=Strong |
| 5 | Statement | 01="Kids are talking...", 02="Dogs are sitting..." |
| 6 | Repetition | 01=1st, 02=2nd |
| 7 | Actor | 01-24 (Odd=Male, Even=Female) |

## The Model
We built a Convolutional Neural Network (CNN) from scratch using PyTorch.

**How it works:**
* It has 4 convolution layers that get deeper as they go (32 -> 64 -> 128 -> 256 channels).
* We used `Dropout(0.25)` to stop the model from memorizing the data (overfitting).
* We used `Adam` to optimize it because it usually works the best for this kind of thing.

## Tech Stack
* **Code:** PyTorch
* **Data:** Hugging Face
* **Tracking:** Weights & Biases (wandb)
* **Where we ran it:** Google Colab (using the free GPUs)

## How to Try It
1.  **Open the Notebook:**
    Click the badge below to open our code in Google Colab:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elijkon/DeepLearning_MiniHackathon/blob/main/DL_miniHackathon.ipynb)

2.  **Add your Token:**
    You need a Hugging Face token to download the data. Add it to the "Secrets" tab in Colab and name it `HF_TOKEN`.

3.  **Install Stuff:**
    The notebook will automatically install the libraries you need.

4.  **Train it:**
    Just run the training cell. It's set to run for 50 epochs.

## How it Did
We tracked the accuracy and loss in WandB. The notebook calculates the final accuracy on a test set (data the model hasn't seen before) after the last epoch.

## What I'd Change Next Time
If we kept working on this, here is what we would do:
* **Mess with the data:** We would add noise or cut parts of the audio out (augmentation) to make the model smarter.
* **Use a bigger model:** Instead of building our own, we would try fine-tuning a big pre-made model like ResNet.
* **Combine methods:** We would try to use both the audio waves and the images together.

## Credits & Citation
"The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)" by Livingstone & Russo is licensed under CC BY-NA-SC 4.0.
