#!/usr/bin/env python3
"""
Speech recognition script using Whisper Large v3 via Groq
"""

import os
import argparse
import sys
from groq import Groq

def parse_arguments():
    parser = argparse.ArgumentParser(description='Speech recognition with Whisper Large v3 via Groq')
    parser.add_argument('--audio-file', type=str, required=True, help='Path to audio file')
    parser.add_argument('--language', type=str, help='Optional language code (e.g., "en", "fr")')
    parser.add_argument('--response-format', type=str, default='text', choices=['text', 'json', 'verbose_json', 'srt', 'vtt'], 
                        help='Response format')
    parser.add_argument('--temperature', type=float, default=0, help='Sampling temperature')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Check if GROQ_API_KEY is set
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not set.")
        print("Set it with: export GROQ_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found.")
        sys.exit(1)
    
    # Initialize Groq client
    client = Groq(api_key=api_key)
    
    try:
        # Prepare request parameters
        request_params = {
            "model": "whisper-large-v3",
            "file": open(args.audio_file, "rb"),  # Pass the file object directly
            "response_format": args.response_format,
            "temperature": args.temperature
        }
        
        # Add language if specified
        if args.language:
            request_params["language"] = args.language
        
        # Call the Groq API for audio transcription
        response = client.audio.transcriptions.create(**request_params)
        
        # Output the response - access the correct attribute based on response format
        if hasattr(response, 'text'):
            print(response.text)
        else:
            # If response is a dictionary or string
            print(response)
        
    except Exception as e:
        print(f"Error during API call: {e}")
        sys.exit(1)
    finally:
        # Close the file if it was opened
        if 'file' in request_params and hasattr(request_params['file'], 'close'):
            request_params['file'].close()

if __name__ == "__main__":
    main() 