import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from collections import Counter

class ImageTagger:
    def __init__(self, device="cpu"):
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

    def generate_caption(self, image_path, max_length=32):
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # 方式2：带前缀约束（更精准）
            prefix = "a photography of"
            inputs = self.processor(image, text=prefix, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_length=max_length)
            
            return self.processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

def batch_process(folder_path, top_n=25):
    tagger = ImageTagger(device="cuda" if torch.cuda.is_available() else "cpu")
    captions = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                path = os.path.join(root, file)
                if caption := tagger.generate_caption(path):
                    captions.append(caption)
    
    counter = Counter(captions)
    for i, (caption, count) in enumerate(counter.most_common(top_n), 1):
        print(f"{i}. [{count}] {caption}")

if __name__ == "__main__":
    import sys
    batch_process(sys.argv[1])