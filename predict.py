import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Định nghĩa lại kiến trúc mạng (giống lúc training)
from torchvision import models

# Danh sách các lớp (thay bằng class bạn đã huấn luyện)
class_names = ['bumper_dent', 'bumper_scratch', 'door_dent', 'door_scratch',
               'glass_shatter', 'head_lamp', 'tail_lamp', 'unknown']

# Thiết bị tính toán
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load lại model ResNet18 (giống lúc train)
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("model/car_damage_model.pth", map_location=device))
model.eval().to(device)

# Hàm xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    return predicted_class
