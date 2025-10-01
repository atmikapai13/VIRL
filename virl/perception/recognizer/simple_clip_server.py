#!/usr/bin/env python3
"""
Simple CLIP server without gradio dependencies
"""
import argparse
import torch
try:
    import clip
except ImportError:
    print("Using open_clip as fallback")
    import open_clip as clip
from flask import Flask, request, jsonify
import base64
from PIL import Image
import io

app = Flask(__name__)

class SimpleCLIPWrapper:
    def __init__(self, model_name='ViT-B/32'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model {model_name} on {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        print("CLIP model loaded successfully")

    def predict(self, image, text_list, temperature=100.0):
        # Preprocess image
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Tokenize text
        text_tokens = clip.tokenize(text_list).to(self.device)
        
        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(image_tensor, text_tokens)
            logits_per_image = logits_per_image / 100.0 * temperature
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            
        return logits_per_image.cpu().numpy().tolist(), probs.tolist()

# Global model instance
clip_model = None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # Get text list
        text_list = data['text'].split(',,')
        temperature = data.get('temperature', 100.0)
        
        # Make prediction
        logits, probs = clip_model.predict(image, text_list, temperature)
        
        return jsonify({
            'logits': logits[0],
            'probabilities': probs[0]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'device': clip_model.device})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=22411)
    parser.add_argument("--model_name", type=str, default='ViT-B/32')
    args = parser.parse_args()
    
    print("Initializing CLIP model...")
    clip_model = SimpleCLIPWrapper(args.model_name)
    
    print(f"Starting CLIP server on port {args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=False)