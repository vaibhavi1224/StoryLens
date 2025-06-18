# 📘 README.md

# StoryLens - Multi-modal Photo Story Generator

## 📖 What It Does
Upload any image — vacation, birthday, or a random moment — and this AI-powered app will create a poetic story inspired by it, then narrate it in an AI-generated voice!

## 🚀 Tech Stack
- `microsoft/kosmos-2` for vision-to-text (story generation)
- `coqui/xtts-v2` via `TTS` library for AI voice narration
- Streamlit for the UI

### 📁 Folder Structure Suggestion:

```
storylens/
├── app.py
├── requirements.txt
└── README.md
```

## 🔧 Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
````

### 2. Run App

```bash
streamlit run app.py
```

### ✅ Features Implemented:

* 📸 **Image Upload** via Streamlit
* 🧠 **Story/Poem Generation** using `microsoft/kosmos-2` Vision2Seq model
* 🔊 **AI Audio Narration** using `coqui/xtts-v2` via the `TTS` library
* 🎧 **Audio Playback** embedded in Streamlit UI
