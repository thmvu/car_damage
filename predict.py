import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# ✅ Cho phép load class FasterRCNN (fix lỗi pickle)
torch.serialization.add_safe_globals([fasterrcnn_resnet50_fpn])

# ✅ Thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load nguyên model đã lưu bằng torch.save(model)
model = torch.load("model/model.pth", map_location=device, weights_only=False)
model.to(device)
model.eval()

# ✅ Tên các class — thứ tự phải đúng với khi mày train
CLASS_NAMES = [
    "__background__",        # ID 0
    "door_scratch",          # ID 1
    "bumper_dent",           # ID 2
    "bumper_scratch",        # ID 3
    "door_dent",             # ID 4
    "glass_shatter",         # ID 5
    "head_lamp",             # ID 6
    "tail_lamp"              # ID 7
]

# ✅ Transform ảnh vào
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

# ✅ Hàm dự đoán ảnh
def predict_image(image_path, output_path="static/result.jpg", conf_thresh=0.5):
    # Load ảnh và chuyển tensor
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(image_tensor)[0]

    boxes = preds['boxes']
    labels = preds['labels']
    scores = preds['scores']

    # Dùng OpenCV để vẽ kết quả
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    detected_classes = set()

    for box, label, score in zip(boxes, labels, scores):
        if score < conf_thresh:
            continue
        x1, y1, x2, y2 = map(int, box.tolist())
        label_name = CLASS_NAMES[label]
        detected_classes.add(label_name)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_cv, f"{label_name} ({score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

    cv2.imwrite(output_path, img_cv)

    if not detected_classes:
        return output_path, "Không phát hiện hư hỏng"
    else:
        return output_path, ", ".join(sorted(detected_classes))
