# ════════════════════════════════════════════════════════════
#  🤖 MULTI-FEATURE AI CHATBOT — Streamlit UI (Day 14)
#  Beginner style: one function per feature + lots of comments
# ════════════════════════════════════════════════════════════

# ── STEP 1: IMPORTS ──────────────────────────────────────────
import streamlit as st
import torch
from io import BytesIO
from gtts import gTTS
import speech_recognition as sr
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from diffusers import StableDiffusionPipeline

# ── STEP 2: PAGE SETUP ───────────────────────────────────────
st.set_page_config(page_title="Multi-Feature Chatbot", page_icon="🤖", layout="wide")

# ── STEP 3: CUSTOM CSS (this is what makes it look cool) ─────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@500;700;900&family=Space+Grotesk:wght@400;500;700&display=swap');

/* Layered ambient background: two soft glows + dot grid */
.stApp {
  background:
    radial-gradient(900px 500px at 88% -10%, rgba(69,201,192,.10), transparent 60%),
    radial-gradient(750px 450px at -8% 18%, rgba(255,179,71,.09), transparent 60%),
    radial-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
    linear-gradient(180deg, #0c1b21, #0a141a);
  background-size: auto, auto, 26px 26px, auto;
}
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

/* Hide default Streamlit chrome */
#MainMenu, footer { visibility: hidden; }

/* ── Top status bar ── */
.topbar { display:flex; align-items:center; gap:10px; font-size:.72rem;
  letter-spacing:.22em; text-transform:uppercase; color:#8fb3b0; margin-bottom:14px; }
.topbar-right { margin-left:auto; color:#5d7f7d; }
.led { width:9px; height:9px; border-radius:50%; background:#5df0a8;
  box-shadow:0 0 10px #5df0a8; animation:pulse 1.6s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }

/* ── Big title ── */
.title { font-family:'Unbounded', sans-serif; font-weight:900;
  font-size:clamp(2rem, 4.6vw, 3.4rem); line-height:1.05; color:#eef6f6; margin:0; }
.title span { color:#ffb347; }
.subtitle { color:#9db8b6; font-size:1.02rem; margin:12px 0 18px; max-width:640px; }

/* ── Feature chips ── */
.chips { display:flex; flex-wrap:wrap; gap:10px; margin-bottom:34px; }
.chip { border:1px solid rgba(255,255,255,.14); background:rgba(255,255,255,.04);
  padding:6px 15px; border-radius:999px; font-size:.8rem; color:#cfe3e2;
  transition:all .2s ease; cursor:default; }
.chip:hover { transform:translateY(-3px); border-color:#ffb347; color:#ffb347; }

/* ── Feature panel card ── */
.panel { display:flex; gap:18px; align-items:flex-start; padding:22px 24px;
  border:1px solid rgba(255,255,255,.09); border-left:4px solid var(--accent,#ffb347);
  border-radius:14px; margin-bottom:26px;
  background:linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.015));
  animation:rise .5s ease both; }
@keyframes rise { from{opacity:0; transform:translateY(14px)} to{opacity:1; transform:none} }
.panel-num { font-family:'Unbounded'; font-weight:700; font-size:1.5rem;
  color:var(--accent); opacity:.9; min-width:52px; }
.panel-title { font-family:'Unbounded'; font-weight:700; font-size:1.15rem; color:#eef6f6; }
.panel-desc { color:#9db8b6; font-size:.92rem; margin-top:5px; }

/* ── Buttons ── */
.stButton > button {
  background:linear-gradient(135deg, #ffb347, #ff8c42); color:#14232b;
  font-weight:700; font-family:'Space Grotesk'; border:none; border-radius:12px;
  padding:.6rem 1.4rem; transition:all .18s ease; }
.stButton > button:hover { transform:translateY(-2px);
  box-shadow:0 10px 26px rgba(255,150,60,.35); }

/* ── Inputs ── */
.stTextInput input { background:rgba(255,255,255,.05); color:#eef6f6;
  border:1px solid rgba(255,255,255,.12); border-radius:10px; }
.stTextInput input:focus { border-color:#ffb347; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background:linear-gradient(180deg,#0a171d,#0b1318);
  border-right:1px solid rgba(255,255,255,.07); }
.side-kicker { font-size:.68rem; letter-spacing:.24em; text-transform:uppercase;
  color:#ffb347; margin-bottom:6px; }
.side-title { font-family:'Unbounded'; font-weight:700; font-size:1.05rem;
  color:#eef6f6; margin-bottom:20px; }
</style>
""", unsafe_allow_html=True)

# ── STEP 4: THE "BOOK" (knowledge base for Feature 5) ────────
# The QA bot can ONLY answer from this text.
# 👉 Want the full Day-14 treatise? Paste your long notebook text here.
THE_BOOK = """
Large Language Models (LLMs) are AI systems trained on massive amounts of text to understand and generate human language. They are built on the Transformer architecture, introduced in the 2017 paper "Attention Is All You Need" by Vaswani et al.

The core mechanism of a Transformer is self-attention. It uses Queries, Keys and Values to let every word look at every other word and decide what is important. Modern LLMs like GPT, LLaMA and Qwen are decoder-only Transformers that generate text one token at a time. This training objective is called causal language modeling or next-token prediction.

Before text enters the model it is split into tokens using algorithms like Byte-Pair Encoding (BPE). Models are pre-trained on trillions of tokens from datasets such as Common Crawl and The Pile.

Scaling laws, especially the Chinchilla paper from DeepMind, showed that model performance improves predictably with more parameters, more data and more compute.

After pre-training, models go through alignment. Supervised Fine-Tuning (SFT) teaches them to follow instructions, and Reinforcement Learning from Human Feedback (RLHF) aligns them with human preferences. Direct Preference Optimization (DPO) is a simpler and more stable alternative to RLHF.

At inference time, techniques like KV caching, FlashAttention and quantization (INT8, INT4, GPTQ, AWQ) make models faster and smaller. Retrieval-Augmented Generation (RAG) connects LLMs to external documents to reduce hallucinations, which are confident but factually wrong answers.

Common benchmarks include MMLU for knowledge, HumanEval for code, and GSM8K for math. The LMSYS Chatbot Arena ranks models by human preference.

The frontier includes Mixture of Experts (MoE) models that activate only a few expert networks per token, and State Space Models like Mamba that scale to very long contexts.
"""

# ── STEP 5: MODEL LOADERS (cached = loaded once, reused) ─────
@st.cache_resource(show_spinner="⏳ Warming up GPT-2...")
def load_gpt2():
    return pipeline("text-generation", model="gpt2")

@st.cache_resource(show_spinner="⏳ Loading QA brain (RoBERTa)...")
def load_qa():
    name = "deepset/roberta-base-squad2"
    return AutoTokenizer.from_pretrained(name), AutoModelForQuestionAnswering.from_pretrained(name)

@st.cache_resource(show_spinner="⏳ Loading Stable Diffusion (first time is slow)...")
def load_sd():
    use_gpu = torch.cuda.is_available()
    dtype = torch.float16 if use_gpu else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=dtype)
    return pipe.to("cuda" if use_gpu else "cpu")

# ── STEP 6: SMALL HELPER — draws the feature card header ─────
def panel_header(num, title, desc, color):
    st.markdown(f"""
    <div class="panel" style="--accent:{color}">
      <div class="panel-num">{num}</div>
      <div>
        <div class="panel-title">{title}</div>
        <div class="panel-desc">{desc}</div>
      </div>
    </div>""", unsafe_allow_html=True)

# ── STEP 7: ONE FUNCTION PER FEATURE ─────────────────────────

# FEATURE 1 — Text → Speech
def feature_tts():
    panel_header("01", "Text → Speech", "Type anything — the bot reads it out loud using Google TTS.", "#ffb347")
    text = st.text_input("✍️ Type something to speak:", placeholder="Hello! I am your multi-feature chatbot.")
    lang = st.text_input("🌍 Language code:", value="en", help="en, hi, es, fr, de, ja ...")
    if st.button("🔊 Speak it!", use_container_width=True):
        if not text.strip():
            st.warning("Please type some text first 🙂")
        else:
            with st.spinner("Synthesizing voice..."):
                buf = BytesIO()                 # save audio in memory (no messy files)
                gTTS(text=text, lang=lang).write_to_fp(buf)
                buf.seek(0)
            st.audio(buf, format="audio/mp3")
            st.success("Done! Press play above ▶️")

# FEATURE 2 — Speech → Text
def feature_stt():
    panel_header("02", "Speech → Text",
                 "Browsers can't open your mic directly, so upload a recording instead (WAV works best).",
                 "#45c9c0")
    uploaded = st.file_uploader("🎙️ Upload an audio file:", type=["wav", "flac"])
    lang = st.text_input("🌍 Language:", value="en-US")
    if uploaded is not None and st.button("🧠 Transcribe!", use_container_width=True):
        with st.spinner("Listening carefully..."):
            with open("temp_audio.wav", "wb") as f:
                f.write(uploaded.getbuffer())
            r = sr.Recognizer()
            try:
                with sr.AudioFile("temp_audio.wav") as source:
                    audio = r.record(source)
                text = r.recognize_google(audio, language=lang)
                st.success("Heard it!")
                st.markdown(f'### 🗣️ You said:\n> "{text}"')
            except sr.UnknownValueError:
                st.error("Sorry, I could not understand the audio 😕")
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# FEATURE 3 — Text → Image
def feature_image():
    panel_header("03", "Text → Image",
                 "Stable Diffusion paints your words. GPU ≈ 10 seconds, CPU ≈ several minutes.",
                 "#ff7a6b")
    prompt = st.text_input("🎨 Describe your image:", placeholder="A robot painting on a rooftop at sunset")
    if st.button("🖌️ Generate!", use_container_width=True):
        if not prompt.strip():
            st.warning("Describe something first 🙂")
        else:
            with st.spinner("Dreaming up your image..."):
                pipe = load_sd()
                image = pipe(prompt).images[0]
            st.image(image, caption=prompt, use_container_width=True)
            image.save("generated_image.png")
            st.success("Saved as generated_image.png")

# FEATURE 4 — Text Prediction (GPT-2)
def feature_predict():
    panel_header("04", "Text Prediction", "Give GPT-2 a starting line and it continues the story.", "#7de2a8")
    prompt = st.text_input("✍️ Starting line:", value="Human is")
    length = st.slider("Max length (tokens):", 20, 150, 50)
    if st.button("🚀 Continue the story!", use_container_width=True):
        with st.spinner("GPT-2 is writing..."):
            gen = load_gpt2()
            out = gen(prompt, max_length=length, do_sample=True, temperature=0.8)[0]["generated_text"]
        st.markdown("### 📖 GPT-2 wrote:")
        st.markdown(f"> {out}")

# FEATURE 5 — LLM Domain Q&A
def feature_qa():
    panel_header("05", "LLM Domain Q&A", "RoBERTa finds the exact answer inside the knowledge base below.", "#6fb7ff")
    with st.expander("📚 Read the knowledge base (the 'book')"):
        st.markdown(THE_BOOK)
    question = st.text_input("❓ Your question:", placeholder="What is an LLM?")
    if st.button("🧠 Answer me!", use_container_width=True):
        if not question.strip():
            st.warning("Ask something first 🙂")
        else:
            with st.spinner("Searching the book..."):
                tokenizer, model = load_qa()
                inputs = tokenizer(question, THE_BOOK, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                start = torch.argmax(outputs.start_logits)
                end = torch.argmax(outputs.end_logits) + 1
                answer = tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end]))
                conf = (torch.softmax(outputs.start_logits, dim=-1)[0, start].item() +
                        torch.softmax(outputs.end_logits, dim=-1)[0, end - 1].item()) / 2
            st.markdown(f"### 💡 Answer\n**{answer.strip()}**")
            st.progress(conf, text=f"Confidence: {conf:.0%}")

# ── STEP 8: HEADER (always visible) ──────────────────────────
st.markdown("""
<div class="topbar">
  <span class="led"></span> Systems online
  <span class="topbar-right">Day 14 · Colab → Streamlit</span>
</div>
<h1 class="title">MULTI-FEATURE <span>CHATBOT</span></h1>
<p class="subtitle">Five AI powers in one deck — speak it, hear it, paint it, predict it, ask it. Pick a power from the control panel.</p>
<div class="chips">
  <span class="chip">📢 gTTS</span>
  <span class="chip">🎙️ SpeechRecognition</span>
  <span class="chip">🎨 Stable Diffusion</span>
  <span class="chip">✍️ GPT-2</span>
  <span class="chip">🧠 RoBERTa QA</span>
</div>
""", unsafe_allow_html=True)

# ── STEP 9: SIDEBAR MENU ─────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="side-kicker">Day 14 · AI Lab</div>'
                '<div class="side-title">🎛️ CONTROL PANEL</div>', unsafe_allow_html=True)

    menu = {
        "📢 Text → Speech":    feature_tts,
        "🎙️ Speech → Text":    feature_stt,
        "🎨 Text → Image":     feature_image,
        "✍️ Text Prediction":  feature_predict,
        "🧠 LLM Q&A":          feature_qa,
    }
    choice = st.radio("Pick a power:", list(menu.keys()))

    st.divider()
    st.caption("⚡ Models load on first use, then stay cached in memory.")
    st.caption("🌐 TTS, STT and model downloads need internet.")

# ── STEP 10: RUN THE CHOSEN FEATURE ──────────────────────────
menu[choice]()

st.markdown("<br><div style='text-align:center;color:#5d7f7d;font-size:.75rem;letter-spacing:.15em'>"
            "DAY 14 · BUILT WITH STREAMLIT 🤖</div>", unsafe_allow_html=True)
