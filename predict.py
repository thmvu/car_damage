import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Danh sách class (phải khớp đúng lúc train)
class_names = ['bumper_dent', 'bumper_scratch', 'door_dent', 'door_scratch',
               'glass_shatter', 'head_lamp', 'tail_lamp', 'unknown']

# Thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load mô hình đã lưu
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("car_damage_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Tiền xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# Thư mục chứa ảnh cần test
test_folder = "images"

# Dự đoán từng ảnh
for img_name in os.listdir(test_folder):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img_path = os.path.join(test_folder, img_name)
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = class_names[predicted.item()]
    
    print(f"{img_name}: → {label}")
