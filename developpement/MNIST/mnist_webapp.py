import gradio as gr
import argparse
from PIL import Image
import requests
import io
import numpy as np


def recognize_digit(image):
    # Convert to PIL Image necessary if using the API method
    #print(image)

    image = image['background']

    # Convert to PIL Image
    image = Image.fromarray(image.astype('uint8'))

    #Convert to 1 channel grey image
    image = image.convert('L')

    #Inverse the white and the dark
    image = Image.eval(image, lambda x: 255-x)

    # Convert to numpy array
    image = np.array(image)

    image = Image.fromarray(image.astype('uint8'))
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")
    
    # Send request to the API
    response = requests.post("http://127.0.0.1:5000/predict", data=img_binary.getvalue())
    return response.json()["prediction"]

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    gr.Interface(fn=recognize_digit, 
                inputs="sketchpad", 
                outputs='label',
                live=True,
                description="Draw a number on the sketchpad to see the model's prediction.",
                ).launch(debug=True, share=True);