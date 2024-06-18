import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from cycle_gan_model import CycleGAN

def train_cyclegan(data_dir, epochs=100):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    model = CycleGAN().to("cuda")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    for epoch in range(epochs):
        for i, (images, _) in enumerate(dataloader):
            images = images.to("cuda")

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
    
    torch.save(model.state_dict(), 'models/cyclegan_model.pth')
    print('CycleGAN model saved!')

if __name__ == '__main__':
    train_cyclegan('data/tattoo_designs')
