# Groq Inference Scripts

This directory contains scripts for text generation and speech recognition using Groq's API.

## Prerequisites

1. Install the Groq Python library:
   ```bash
   pip install groq
   ```

2. Set your Groq API key as an environment variable:
   ```bash
   export GROQ_API_KEY='your-api-key'
   ```

3. Make the scripts executable:
   ```bash
   chmod +x text-inference-with-llama3-70b.py
   chmod +x text-inference-with-mixtral-8x7b.py
   chmod +x speech_recognition/whisper-large-v3.py
   chmod +x speech_recognition/whisper-medium.py
   ```

## Text Generation Scripts

### Llama 3 70B

```bash
# Basic usage
./text-inference-with-llama3-70b.py --prompt "Explain quantum computing in simple terms"

# With custom parameters
./text-inference-with-llama3-70b.py --prompt "Write a short story" --max-tokens 2048 --temperature 0.9 --top-p 0.95

# Using stdin
echo "Summarize this meeting transcript:" | cat meeting.txt - | ./text-inference-with-llama3-70b.py
```

### Mixtral 8x7B

```bash
# Basic usage
./text-inference-with-mixtral-8x7b.py --prompt "Explain the difference between AI and ML"

# With system prompt
./text-inference-with-mixtral-8x7b.py --prompt "Create a recipe for chocolate cake" --system-prompt "You are a professional chef"

# With custom parameters
./text-inference-with-mixtral-8x7b.py --prompt "Write a poem about nature" --max-tokens 1500 --temperature 0.8
```

## Speech Recognition Scripts

### Whisper Large v3

```bash
# Basic usage
./speech_recognition/whisper-large-v3.py --audio-file path/to/audio.mp3

# Specify language
./speech_recognition/whisper-large-v3.py --audio-file path/to/audio.mp3 --language en

# Get subtitles format
./speech_recognition/whisper-large-v3.py --audio-file path/to/audio.mp3 --response-format srt
```

### Whisper Medium

```bash
# Basic usage
./speech_recognition/whisper-medium.py --audio-file path/to/audio.mp3

# With prompt to guide transcription
./speech_recognition/whisper-medium.py --audio-file path/to/audio.mp3 --prompt "Meeting about quarterly sales"

# Get verbose JSON output
./speech_recognition/whisper-medium.py --audio-file path/to/audio.mp3 --response-format verbose_json
```

## Notes

- All scripts support error handling and will provide meaningful error messages.
- Make sure your audio files are in formats supported by Groq (e.g., mp3, wav, m4a, etc.).
- For the best speech recognition results, use clear audio with minimal background noise.
- Response formats for speech recognition include: text (default), json, verbose_json, srt, vtt.
- The speech recognition scripts pass file objects directly to the Groq API rather than base64-encoded strings, as required by the latest Groq Python SDK. 