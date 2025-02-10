
from PIL import Image
#importing models
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import gradio as gr
from diffusers import DiffusionPipeline,StableDiffusion3Pipeline

load_dotenv()

Hugging_face_token=os.getenv('huggingface_token')

# ! huggingface-cli login --token $Hugging_face_token


# loading image captionning model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Set the model name for our LLMs.
GEMINI_MODEL = "gemini-2.0-flash"

# Store the API key in a variable.
GEMINI_API_KEY = os.getenv("google_api_key")

class stable_dif:
  def __init__(self,sizes):
    self.sizes=sizes

  def model(self):
    if self.sizes == 'medium':
      pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium")
    elif self.sizes == 'large':
      pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large-turbo")
    elif self.sizes == 'small':
      pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
    return pipe
  
stable=stable_dif('small')
pipe=stable.model()



def image_story_generator(image,requirement,style):

  raw_image = Image.open(image)

  # get caption from image
  inputs = processor(raw_image, return_tensors="pt")
  out = model.generate(**inputs, min_length=20)
  model_prompt=processor.decode(out[0], skip_special_tokens=True)

  #load gemnini for creating story
  llm = ChatGoogleGenerativeAI(google_api_key=GEMINI_API_KEY, model=GEMINI_MODEL, temperature=0.3)

  query =f' Write a 4 chapters story based on {model_prompt} and\
  that fits the following requirements: {requirement}. Give a detailed\
  description of the charaters appearences.'

  result = llm.invoke(query)
  story= result.content.replace('\n',' ')

  # create promts for image gen from story
  image_prompt_llm=ChatGoogleGenerativeAI(google_api_key=GEMINI_API_KEY, model=GEMINI_MODEL, temperature=0.3)

  # create shemas to format output
  schemas=[
      ResponseSchema(name='prompt 1', description='the prompt'),
      ResponseSchema(name='prompt 2', description='the prompt'),
      ResponseSchema(name='prompt 3', description='the prompt'),
      ResponseSchema(name='prompt 4', description='the prompt')
  ]

  # initialize parser for output
  parser=StructuredOutputParser.from_response_schemas(schemas)
  instructions=parser.get_format_instructions()

  query = f' Based on this story: {story}. Create 4 prompts for stable diffusion that tells of a maximum of 77 tokens\
  what happens in each chapters. Describe the characters everytime their name is mentioned. Each image should be created in the same exact style {style}.\
  '+ '\n\n'+instructions

  result=image_prompt_llm.invoke(query)
  image_prompts = parser.parse(result.content)

  # iterate through the prompts and generate new images
  images=[]
  for i in image_prompts.keys():

    image = pipe(image_prompts[i]).images[0]
    images.append(image)


  return images, story

# gradio
interface = gr.Interface(
    fn=image_story_generator,
    inputs=[gr.Image(type='filepath'),gr.Textbox('enter story requirements'), gr.Textbox('pick a style for the images')],

    outputs=[gr.Gallery(),
        gr.Textbox('story')
    ],
    description='Upload an image to start the story generation process.'
)

interface.launch()