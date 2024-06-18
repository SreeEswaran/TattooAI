# TattooAI

## Project Description

The TattooAI allows users to generate high-quality, personalized tattoo designs based on textual prompts and visualize them on body images. This project leverages state-of-the-art AI models to produce realistic and aesthetically pleasing tattoo designs.


## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SreeEswaran/TattooAI.git
   cd TattooAI
2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
3. **Train the CycleGAN model**:
   ```bash
   python models/train_cyclegan.py
4. **FineTune Stable Diffusion Model**:
   ```bash
   python models/train_diffusion.py
5. **To run flask server**:
   ```bash
   python app.py


### Notes

1. **Dataset Preparation**:
   - Place tattoo design images in `data/tattoo_designs/`.
   - Place body images in `data/body_images/`.

2. **Model Training**:
   - Adjust the number of epochs and other hyperparameters as needed.

3. **Deployment**:
   - For deployment, consider using a cloud platform such as AWS, GCP, or Heroku.

This should provide you with a comprehensive setup to train and deploy the AI Tattoo App (TattooAI). Adjust paths, hyperparameters, and any specific configurations based on your environment and needs.
