import gradio as gr
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests


def predict(url):
    image = Image.open(requests.get(url, stream=True).raw)

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    result= f"Predicted class: {model.config.id2label[predicted_class_idx]}"
    return result

iface = gr.Interface(fn=predict, inputs="text", outputs="text")
iface.launch()