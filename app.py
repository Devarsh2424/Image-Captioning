# to create nueral network
import torch

# for interface
import gradio as gr

# to open images
from PIL  import Image

# used for audio
import scipy.io.wavfile as wavfile

# Use a pipeline as a high-level helper
from transformers import pipeline


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

narrator = pipeline("text-to-speech", model="kakao-enterprise/vits-ljs")

# Load the pretrained weights
caption_image = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=device)


# Define the function to generate audio from text 
def generate_audio(text):
    # Generate the narrated text
    narrated_text = narrator(text)

    # Save the audio to WAV file
    wavfile.write("output.wav", rate=narrated_text["sampling_rate"],
                  data=narrated_text["audio"][0])

    # Return the path to the saved output WAV file
    return "output.wav" # return audio

def caption_my_image(pil_image):

    semantics = caption_image(images=pil_image)[0]['generated_text']
    audio = generate_audio(semantics)
    return semantics,audio  # returns both text and audio output


gr.close_all()

demo = gr.Interface(fn=caption_my_image,
                    inputs=[gr.Image(label="Select Image",type="pil")],
                    outputs=[ 
                        gr.Textbox(label="Image Caption"),
                        gr.Audio(label="Image Caption Audio")],
                    title="IMAGE CAPTIONING WITH AUDIO OUTPUT",
                    description="THIS APPLICATION WILL BE USED TO CAPTION IMAGES WITH THE HELP OF AI")
demo.launch()