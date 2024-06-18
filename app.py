from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from models.train_diffusion import generate_tattoo_design
from models.cycle_gan_model import CycleGAN
from PIL import Image
import torch
from torchvision import transforms

app = Flask(__name__)
CORS(app)

model = CycleGAN()
model.load_state_dict(torch.load('models/cyclegan_model.pth', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def simulate_tattoo_on_body(tattoo_design_path, body_image_path):
    tattoo_image = Image.open(tattoo_design_path)
    body_image = Image.open(body_image_path)

    tattoo_tensor = transform(tattoo_image).unsqueeze(0)
    body_tensor = transform(body_image).unsqueeze(0)

    with torch.no_grad():
        simulated_tensor = model.generate(tattoo_tensor, body_tensor)
    
    simulated_image = transforms.ToPILImage()(simulated_tensor.squeeze())
    simulated_image_path = "static/tattoo_on_body.png"
    simulated_image.save(simulated_image_path)
    return simulated_image_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_tattoo', methods=['POST'])
def generate_tattoo():
    data = request.get_json()
    prompt = data['prompt']

    tattoo_design_path = generate_tattoo_design(prompt)
    body_image_path = "data/body_images/sample_body_image.jpg"
    simulated_image_path = simulate_tattoo_on_body(tattoo_design_path, body_image_path)

    response = {
        "image_url": tattoo_design_path,
        "body_image_url": simulated_image_path
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
