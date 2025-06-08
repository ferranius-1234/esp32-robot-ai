import os
import torch
import requests
import tempfile
from flask import Flask, request, jsonify, send_file
from transformers import pipeline
from scipy.io.wavfile import write

# Leer clave API desde variable de entorno
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)

# Cargar modelo Whisper tiny
whisper_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

# Cargar modelo Silero TTS en español
language = 'es'
model, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language=language,
    speaker='v3_es'
)

# Convertir texto a voz y devolver ruta temporal del archivo de audio
def text_to_speech(text):
    sample_rate = 48000
    audio = model.apply_tts(text=text, speaker='es_0', sample_rate=sample_rate)
    tmp_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    write(tmp_path.name, sample_rate, audio)
    return tmp_path.name

# Llamar a Gemini con el texto transcrito
def ask_gemini(text):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": text}]}]}
    response = requests.post(f"{url}?key={GEMINI_API_KEY}", headers=headers, json=payload)
    res_json = response.json()
    return res_json["candidates"][0]["content"]["parts"][0]["text"]

# Ruta de prueba
@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "API funcionando correctamente"})

# Ruta que recibe un audio, lo transcribe, pregunta a Gemini y responde con voz
@app.route("/process-audio", methods=["POST"])
def process_audio():
    if "file" not in request.files:
        return jsonify({"error": "Falta el archivo de audio"}), 400

    audio_file = request.files["file"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_file.save(tmp.name)
        text = whisper_pipe(tmp.name)["text"]

    response_text = ask_gemini(text)
    audio_path = text_to_speech(response_text)

    return send_file(audio_path, mimetype="audio/wav")

if __name__ == "__main__":
    app.run(debug=True)
