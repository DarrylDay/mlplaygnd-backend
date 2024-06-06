import io
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import mnist.model

device = mnist.model.device
classes = mnist.model.classes

model = mnist.model.loadModel("models/mnist.pth")
model.eval()

def get_prediction(img_bytes):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda image: image.view(-1, 1, 28, 28))
    ])
    image = Image.open(io.BytesIO(img_bytes))
    img_tensor = transform(image)
    #save_image(img_tensor, "input.png")

    with torch.no_grad():
        x = img_tensor.to(device)
        pred = model(x)
        #print(pred)
        return classes[pred[0].argmax(0)]