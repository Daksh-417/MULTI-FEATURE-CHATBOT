# 🤖 Multi-Feature AI Chatbot

> Day 14 project — five AI powers in one Streamlit app.
> Speak it 📢 · Hear it 🎙️ · Paint it 🎨 · Predict it ✍️ · Ask it 🧠

Converted from a Colab notebook (`Day-14.ipynb`) into a beautiful, beginner-friendly
single-file web app (`app.py`).

---

## ✨ Features

| # | Feature | Model / Library | What it does |
|---|---------|-----------------|--------------|
| 1 | 📢 Text → Speech | gTTS (Google) | Types text → plays audio in the browser |
| 2 | 🎙️ Speech → Text | SpeechRecognition (Google API) | Upload a WAV → get the transcript |
| 3 | 🎨 Text → Image | Stable Diffusion v1.5 | Describe a scene → generates a PNG |
| 4 | ✍️ Text Prediction | GPT-2 | Give a starting line → continues the story |
| 5 | 🧠 LLM Domain Q&A | RoBERTa (deepset/roberta-base-squad2) | Ask questions → finds answers inside the full LLM treatise |

---

## 🚀 Quick Start

### 1. Install Python 3.9+ and the dependencies

```bash
pip install -r requirements.txt
