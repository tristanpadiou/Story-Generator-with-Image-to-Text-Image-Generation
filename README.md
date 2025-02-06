# Story Generator with Image-to-Text and Image Generation

This project is a story generation tool that takes a drawing image as input, extracts a caption from the image, and then generates a 4-chapter story with matching images. The project uses machine learning models to generate story content and corresponding visuals, while maintaining a consistent artistic style throughout. The generated story and images are displayed in a user-friendly interface powered by Gradio.

## Features

- **Image-to-Text**: Extracts a caption from the input image.
- **Story Generation**: Uses the image caption to generate a 4-chapter story.
- **Character Descriptions**: Includes detailed character descriptions in the story.
- **Image Generation**: Generates 4 images for the 4 chapters of the story, maintaining the same artistic style.
- **Gradio Interface**: Provides an easy-to-use web interface for uploading images and generating stories with images.

## Requirements

1. **Google Colab**: This project runs in Google Colab for easy access to powerful hardware.
2. **Dependencies**:
    - `torch` for model handling.
    - `transformers` for image captioning and language models.
    - `Pillow` for image processing.
    - `openai` or `ChatGoogleGenerativeAI` for generating text-based stories.
    - `gradio` for building the interface.
    - `diffusers` for Stable Diffusion image generation.
3. **Google API Key** for accessing the Gemini API (or a similar model provider).

## Setup Instructions

1. **Open Google Colab**: Open a new notebook in Google Colab.
2. **Clone or Import the Project**: If hosted on GitHub, clone the repository to your Colab environment, or copy the code into a new Colab notebook.
3. **Install Dependencies**: Install the necessary Python libraries by running the following command:
    ```python
    !pip install torch transformers diffusers Pillow gradio openai
    ```

4. **Set Up API Key**: Set up your Google Gemini API key (or equivalent) and model name in the Colab environment:
    ```python
    GEMINI_API_KEY = 'your-google-api-key'
    GEMINI_MODEL = 'google_model_name'
    ```

5. **Import Required Libraries**: Import the necessary libraries to initialize models, process images, and launch the Gradio interface:
    ```python
    from PIL import Image
    import torch
    import gradio as gr
    from transformers import AutoProcessor, AutoModelForImageCaptioning
    from diffusers import StableDiffusionPipeline
    ```

## How to Use

1. **Upload an Image**: 
   - Open the Colab notebook and run the Gradio interface.
   - Upload a drawing image by clicking on the “Upload Image” section of the Gradio interface.

2. **Enter Story Requirements**:
   - Provide specific requirements for the story, such as character traits, themes, or desired tones (e.g., "A fantasy adventure with a dragon and knight").

3. **Pick a Style**:
   - Choose the style for the generated images (e.g., “medieval art,” “watercolor,” etc.).

4. **Run the Interface**:
   - Once the image and inputs are provided, the program will generate the story and images for you.

5. **View the Results**:
   - The generated story will appear in a textbox.
   - A gallery of images generated for each chapter of the story will be displayed below the story.

### Example:

```python
interface.launch()
```

Once the interface launches, follow these steps:
1. Upload a drawing image.
2. Provide story requirements (e.g., "A heroic quest with magical creatures").
3. Choose a style (e.g., "fantasy art").
4. The interface will generate and display the story along with the corresponding chapter images.

## Code Breakdown

### 1. Image Captioning:
   - The code uses a pre-trained image captioning model to generate a description of the uploaded image. The caption serves as the prompt for generating the story.

### 2. Story Generation:
   - The caption is then passed to a language model (using the Gemini API or similar) to generate a 4-chapter story. The story is customized based on the user's input (e.g., character descriptions, story themes).

### 3. Image Generation:
   - Prompts are generated for each chapter of the story, and Stable Diffusion is used to generate the corresponding images. The images are created in the artistic style chosen by the user.

### 4. Gradio Interface:
   - The Gradio interface allows users to interact with the tool easily. It lets them upload images, provide inputs for the story, and view the generated results.

## Example Output

- **Generated Story**: A 4-chapter narrative that matches the themes and visual elements of the uploaded image.
- **Generated Images**: Four images corresponding to the chapters of the story, generated in the specified artistic style.

## Customizations

- **Change the Number of Chapters**: Modify the code to generate more or fewer chapters in the story.
- **Artistic Styles**: Customize the styles of the generated images (e.g., “impressionist,” “cyberpunk,” “abstract”).
- **Specific Story Themes**: Tailor the story generation by providing specific requirements for characters, themes, or plot.

