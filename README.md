# 🖼️ Image Captioning + Text-to-Speech

Generate a caption from an image and convert it into audio.

## 🚀 Features

* Upload any image
* AI generates a caption using **BLIP**
* Caption is converted to speech using **MMS TTS**
* Audio (.wav) is played instantly
* Simple Gradio interface

## 🧠 Models Used

* **Image-to-Text:** `Salesforce/blip-image-captioning-large`
* **Text-to-Speech:** `facebook/mms-tts-eng`


## 🧩 How It Works

1. User uploads an image
2. BLIP generates caption
3. MMS converts caption → speech
4. Audio is returned
