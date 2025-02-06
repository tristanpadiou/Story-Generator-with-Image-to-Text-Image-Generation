
# Story Generator with Image-to-Text & Image Generation

This project generates a story based on a drawing image input, along with corresponding images for each chapter of the story. It utilizes advanced machine learning models to generate captions from the image, create a story, and generate chapter-based images in the same artistic style.

## Overview

This tool takes a drawing image as input and generates a story that matches the themes, characters, and atmosphere of the input image. The story is divided into 4 chapters, and each chapter has an associated image generated using Stable Diffusion. The story is also tailored to specific user requirements and can generate character descriptions to fit the visual elements.

## Features

- **Image-to-Text Conversion**: Extracts captions and descriptions from the input image using an image captioning model.
- **Story Generation**: Creates a 4-chapter story based on the extracted image caption and user-defined requirements.
- **Character Description**: Includes detailed character descriptions whenever names are mentioned in the story.
- **Image Generation**: Generates 4 unique images (one for each chapter) using Stable Diffusion, maintaining the same artistic style throughout.

## Requirements

1. **Google Colab**: This project is intended to run on Google Colab for easy access to GPU resources.
2. **Libraries**:
    - `torch` for model handling.
    - `transformers` for pre-trained models.
    - `Pillow` for image processing.
    - `openai` (or Gemini) for generating story and image prompts.
    - `diffusers` for Stable Diffusion image generation.
3. **Google API Key** for Gemini (or another language model provider).

## Setup Instructions

1. Open Google Colab.
2. Clone this repository (if using GitHub).
3. Install necessary dependencies by running the following command in a Colab cell:
    ```python
    !pip install torch transformers diffusers Pillow openai
    ```

4. Set up the environment variables, particularly the Google API key for the Gemini model:
    ```python
    GEMINI_API_KEY = 'your-google-api-key'
    GEMINI_MODEL = 'google_model_name'
    ```

5. Import required libraries and initialize the models in your Colab notebook:
    ```python
    from PIL import Image
    import torch
    from transformers import AutoProcessor, AutoModelForImageCaptioning
    from diffusers import StableDiffusionPipeline
    ```

## How to Use

1. **Prepare Your Input**: Upload a drawing image that you want to base the story on.
2. **Run the Story Generator**: 
    - Call the `image_story_generator(image, requirement, style)` function where:
      - `image`: The path to the image (either local or from Google Drive).
      - `requirement`: Specific requirements for the story (e.g., character traits, story tone).
      - `style`: Desired artistic style for the generated images.
      
    Example usage:
    ```python
    images, story = image_story_generator("path_to_your_image.jpg", "A fantasy adventure with a dragon and knight", "medieval art")
    ```

3. **View the Results**:
    - The function will return the generated images and the story.
    - Display the images in Colab using `display()`.
    ```python
    from IPython.display import display

    # Display generated images
    for img in images:
        display(img)
    
    # Print the generated story
    print(story)
    ```

## Code Explanation

### 1. Image Captioning:
   - The `image_story_generator` function first uses an image captioning model (like BLIP or similar) to generate a description or caption for the input image.
   
### 2. Story Generation:
   - Using the caption as a prompt, the function queries a generative language model (Google Gemini in this case) to generate a 4-chapter story. The story is personalized based on the user-defined requirements and includes detailed character descriptions.

### 3. Image Generation:
   - The function constructs prompts for Stable Diffusion to create 4 images corresponding to the chapters of the generated story. Each prompt maintains the artistic style set by the user.
   
### 4. Image Output:
   - The images are generated and returned as output, ready to be displayed or saved.

## Example Output

- **Story**: A 4-chapter story that evolves with characters, events, and settings based on the drawing image.
- **Images**: 4 images, each corresponding to a chapter of the story, created in the same artistic style.

## Potential Customizations

- **Change the number of chapters**: Modify the code to generate more or fewer chapters in the story.
- **Custom artistic styles**: Experiment with different artistic styles for the generated images.
- **Story Requirements**: Provide more specific or creative requirements to guide the story generation.
