#!/usr/bin/env python3
"""
Text inference script for Groq's Llama 3 70B model
"""

import os
import argparse
import sys
from groq import Groq

def parse_arguments():
    parser = argparse.ArgumentParser(description='Text inference with Groq Llama 3 70B')
    parser.add_argument('--prompt', type=str, help='Input prompt for text generation')
    parser.add_argument('--max-tokens', type=int, default=1024, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-p sampling parameter')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Check if GROQ_API_KEY is set
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not set.")
        print("Set it with: export GROQ_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Get prompt from arguments or stdin
    prompt = args.prompt
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    
    if not prompt:
        print("Error: No prompt provided. Use --prompt or pipe text to stdin.")
        sys.exit(1)
    
    # Initialize Groq client
    client = Groq(api_key=api_key)
    
    try:
        # Call the Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama3-70b-8192",
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        # Output the response
        print(chat_completion.choices[0].message.content)
        
    except Exception as e:
        print(f"Error during API call: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 