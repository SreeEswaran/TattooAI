from diffusers import StableDiffusionPipeline

def fine_tune_stable_diffusion(data_dir):
    model_id = "CompVis/stable-diffusion-v1-4"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to("cuda")

    # Fine-tuning logic here - this is simplified
    # You would typically need a custom dataset and training loop

    print('Stable Diffusion fine-tuned and ready for generating tattoos!')

def generate_tattoo_design(prompt):
    model_id = "CompVis/stable-diffusion-v1-4"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to("cuda")

    image = pipeline(prompt).images[0]
    image_path = "static/generated_tattoo.png"
    image.save(image_path)
    return image_path

if __name__ == '__main__':
    fine_tune_stable_diffusion('data/tattoo_designs')
