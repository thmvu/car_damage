# 🚗 Car Damage Detection - Hệ thống phát hiện hư hỏng xe bằng AI

## 1. Thông tin tác giả

- **Trường:** Đại học Phenikaa
- **MSSV:** 23010076
- **Họ tên:** Từ Hữu Minh Vũ **

---

## 2. Tổng quan dự án

Dự án xây dựng một hệ thống phát hiện hư hỏng trên xe hơi bằng cách sử dụng mô hình học sâu (Deep Learning). Ứng dụng hỗ trợ người dùng upload ảnh và đưa ra kết quả loại hư hỏng (như trầy xước, móp méo, vỡ kính,...).  
Kết quả được hiển thị qua giao diện web đơn giản (HTML + Flask backend).

---

## 3. Công nghệ sử dụng

| Thành phần    | Công nghệ sử dụng                          | Ghi chú                                  |
|---------------|--------------------------------------------|------------------------------------------|
| AI / ML       | PyTorch, torchvision                       | Train mô hình phân loại ảnh hư hỏng     |
| Backend       | Flask (Python)                             | Giao tiếp giữa người dùng và mô hình AI |
| Frontend      | HTML, Bootstrap                            | Giao diện upload ảnh và xem kết quả     |
| Xử lý ảnh     | PIL, OpenCV                                | Đọc, xử lý và hiển thị ảnh              |
| Khác          | Google Drive (lưu mô hình)                 | Không commit file model lên GitHub       |

---

## 4. Cấu trúc thư mục dự án

car_damage_detection/
```
├── app/ # Flask app, template và logic chính
│ ├── init.py
│ ├── routes.py # Router Flask
│ ├── templates/
│ │ └── index.html # Giao diện người dùng
│ └── static/ # CSS, JS, ảnh nếu cần
├── scripts/ # Chứa script huấn luyện, xử lý dữ liệu
│ ├── train_model.py
│ └── preprocess.py
├── model/ # Lưu file model.pth (KHÔNG commit lên GitHub)
│ └── model.pth # (upload riêng lên Google Drive)
├── requirements.txt # Các thư viện cần cài đặt
├── README.md # Hướng dẫn sử dụng (file này)
├── TRAINING.md # Mô tả training, thuật toán, metrics,...
├── Dockerfile # Nếu dùng Docker (tuỳ chọn)

yaml

```
---

## 5. Hướng dẫn cài đặt & chạy

```bash
# 1. Tạo môi trường ảo (tuỳ chọn nhưng nên có)
python -m venv venv
source venv/bin/activate      # Trên Linux/macOS
venv\Scripts\activate         # Trên Windows

# 2. Cài các thư viện cần thiết
pip install -r requirements.txt

# 3. Tải mô hình về (upload sẵn trên Google Drive)
# Ví dụ: https://drive.google.com/uc?id=ABC123xyz (model.pth)
mkdir -p model
# Đặt file model.pth vào thư mục model/

# 4. Chạy server Flask
python app/routes.py
6. Test nhanh
bash
Copy
Edit
# Gửi ảnh test qua curl
curl -F 'file=@car.jpg' http://localhost:5000/upload
Hoặc truy cập trực tiếp web:
🔗 http://localhost:5000

7. Kết quả mô hình
Đầu vào: Ảnh xe (jpg, png...)

Đầu ra: Nhãn dự đoán (VD: door_scratch, bumper_dent, glass_shatter)

Tốc độ dự đoán: ~0.2s/ảnh

Độ chính xác: ~92% trên tập test 500 ảnh

8. Ghi chú
Không commit file model.pth lên GitHub

Chia sẻ model qua Google Drive hoặc link tải riêng

9. Thông tin liên hệ
Tên: Nguyễn Văn A

Email: nguyenvana@example.com

Github: https://github.com/nguyenvana

css


---
link download model https://drive.google.com/drive/folders/1ahbORlCbndZKEjK2WiA43WtTzIT-Z4U4?usp=sharing

