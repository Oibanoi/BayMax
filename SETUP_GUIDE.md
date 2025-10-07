# Hướng dẫn cài đặt chi tiết - MedGuide AI

## 📋 Checklist cài đặt

- [ ] Python 3.9+ đã cài đặt
- [ ] Git đã cài đặt
- [ ] FFmpeg đã cài đặt
- [ ] CUDA Toolkit 11.8 (tùy chọn cho GPU)
- [ ] Tài khoản Azure OpenAI
- [ ] Tài khoản Pinecone
- [ ] Firebase project

## 🔧 Cài đặt từng bước

### Bước 1: Chuẩn bị môi trường

```bash
# Kiểm tra Python version
python --version  # Cần >= 3.9

# Clone project
git clone <repository-url>
cd HuuPB2_Assignment_04

# Tạo virtual environment
python -m venv medguide_env
```

### Bước 2: Kích hoạt môi trường ảo

**Windows:**
```cmd
medguide_env\Scripts\activate
```

**Linux/Mac:**
```bash
source medguide_env/bin/activate
```

### Bước 3: Cài đặt dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Cài đặt PyTorch với CUDA (nếu có GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Hoặc CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Cài đặt các thư viện khác
pip install -r requirements_clean.txt
```

### Bước 4: Cài đặt FFmpeg

**Windows:**
1. Tải FFmpeg từ https://ffmpeg.org/download.html
2. Giải nén vào `C:\ffmpeg`
3. Thêm `C:\ffmpeg\bin` vào PATH
4. Kiểm tra: `ffmpeg -version`

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

### Bước 5: Cấu hình Azure OpenAI

1. Đăng ký tài khoản Azure
2. Tạo Azure OpenAI resource
3. Deploy models:
   - GPT-4o-mini
   - text-embedding-3-small
4. Lấy endpoint và API keys

### Bước 6: Cấu hình Pinecone

1. Đăng ký tại https://pinecone.io
2. Tạo index mới:
   - Name: `medical-rag-index`
   - Dimensions: 1536
   - Metric: cosine
3. Lấy API key

### Bước 7: Cấu hình Firebase

1. Tạo Firebase project
2. Enable Firestore Database
3. Tạo service account
4. Tải file JSON credentials
5. Đặt file vào thư mục gốc project

### Bước 8: Tạo file .env

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY_EMBEDDING=your-embedding-key
AZURE_DEPLOYMENT_NAME=text-embedding-3-small
OPENAI_API_KEY=your-openai-key
OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
OPENAI_MODEL=GPT-4o-mini

# Pinecone Configuration
PINECONE_API_KEY=your-pinecone-api-key

# Optional: Custom settings
DEBUG=True
LOG_LEVEL=INFO
```

### Bước 9: Kiểm tra cài đặt

```bash
# Test imports
python -c "import streamlit, openai, pinecone, firebase_admin; print('All imports successful!')"

# Test FFmpeg
ffmpeg -version

# Test CUDA (nếu có GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Bước 10: Chạy ứng dụng

```bash
# Chạy ứng dụng đơn giản
streamlit run simple_app.py

# Hoặc ứng dụng đầy đủ
streamlit run main.py
```

## 🐛 Xử lý lỗi thường gặp

### Lỗi import torch
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Lỗi FFmpeg not found
- Windows: Kiểm tra PATH environment variable
- Linux: `sudo apt install ffmpeg`
- Mac: `brew install ffmpeg`

### Lỗi Firebase credentials
```python
# Kiểm tra đường dẫn file JSON trong image_analysis/core.py
cred = credentials.Certificate("path/to/your/firebase-key.json")
```

### Lỗi Pinecone connection
- Kiểm tra API key trong file .env
- Kiểm tra region và index name
- Đảm bảo index đã được tạo với đúng dimensions (1536)

### Lỗi Azure OpenAI
- Kiểm tra endpoint URL
- Kiểm tra API key
- Đảm bảo models đã được deploy

## 📊 Kiểm tra hiệu năng

```bash
# Kiểm tra GPU memory
python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')" 

# Kiểm tra RAM usage
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().total / 1e9:.1f} GB')"
```

## 🔄 Cập nhật project

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements_clean.txt --upgrade

# Restart application
streamlit run simple_app.py
```

## 📞 Hỗ trợ

Nếu gặp vấn đề:
1. Kiểm tra logs trong terminal
2. Xem file .env có đúng format không
3. Kiểm tra network connection
4. Tạo issue trên GitHub với thông tin lỗi chi tiết

## 🎯 Tối ưu hiệu năng

### Cho máy có GPU:
- Cài CUDA Toolkit 11.8
- Sử dụng PyTorch với CUDA support
- Set `gpu=True` trong EasyOCR

### Cho máy CPU only:
- Sử dụng PyTorch CPU version
- Giảm batch size
- Set `gpu=False` trong EasyOCR

### Tối ưu memory:
- Giảm embedding dimensions nếu cần
- Sử dụng streaming cho audio
- Clear cache định kỳ