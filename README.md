# MedGuide AI - Trợ lý Y tế Thông minh 🏥

Ứng dụng AI hỗ trợ tư vấn y tế với khả năng phân tích hình ảnh, xử lý giọng nói và tìm kiếm thông tin y tế thông minh.

## ✨ Tính năng chính

- 🗣️ **Hỗ trợ giọng nói**: Nhận diện giọng nói và chuyển đổi văn bản thành giọng nói
- 📷 **Phân tích hình ảnh y tế**: OCR và phân tích đơn thuốc, kết quả xét nghiệm
- 💊 **Tư vấn thuốc**: Thông tin về tác dụng, liều lượng, tương tác thuốc
- 🧪 **Giải thích xét nghiệm**: Phân tích và giải thích kết quả xét nghiệm
- 🩺 **Tư vấn triệu chứng**: Hỗ trợ nhận diện và tư vấn về các triệu chứng
- 📅 **Đặt lịch khám**: Hỗ trợ đặt lịch hẹn khám bệnh
- 📊 **So sánh kết quả**: So sánh kết quả xét nghiệm và đơn thuốc theo thời gian
- 📁 **Quản lý tài liệu**: Upload và phân loại tài liệu y tế tự động

## 🛠️ Công nghệ sử dụng

- **Frontend**: Streamlit
- **AI/ML**: Azure OpenAI GPT-4o-mini, LangChain
- **Vector Database**: Pinecone, ChromaDB
- **OCR**: EasyOCR
- **Database**: Firebase Firestore
- **Speech**: SpeechRecognition, Edge-TTS
- **Image Processing**: OpenCV, Pillow

## 📋 Yêu cầu hệ thống

- Python 3.9+
- CUDA Toolkit 11.8 (cho GPU RTX3060 - tùy chọn)
- FFmpeg

## 🚀 Cài đặt

### 1. Clone repository
```bash
git clone <repository-url>
cd HuuPB2_Assignment_04
```

### 2. Tạo môi trường ảo
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

### 3. Cài đặt dependencies
```bash
# Cài đặt PyTorch với CUDA support (tùy chọn)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cài đặt các thư viện khác
pip install -r requirements_clean.txt
```

### 4. Cài đặt FFmpeg
- **Windows**: Tải từ https://ffmpeg.org/download.html và thêm vào PATH
- **Linux**: `sudo apt install ffmpeg`
- **Mac**: `brew install ffmpeg`

### 5. Cấu hình môi trường
Tạo file `.env` với các thông tin sau:
```env
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY_EMBEDDING=your_embedding_key
AZURE_DEPLOYMENT_NAME=text-embedding-3-small
OPENAI_API_KEY=your_openai_key
OPENAI_ENDPOINT=your_openai_endpoint
OPENAI_MODEL=GPT-4o-mini

# Pinecone
PINECONE_API_KEY=your_pinecone_key
```

### 6. Cấu hình Firebase
- Đặt file `baymax-a7a0d-firebase-adminsdk-fbsvc-f00628f505.json` vào thư mục gốc
- Hoặc cập nhật đường dẫn trong `image_analysis/core.py`

## 🎯 Cách sử dụng

### Chạy ứng dụng chính
```bash
streamlit run simple_app.py
```

### Chạy ứng dụng đầy đủ tính năng
```bash
streamlit run main.py
```

## 📁 Cấu trúc project

```
HuuPB2_Assignment_04/
├── simple_app.py              # Ứng dụng Streamlit chính
├── main.py                    # Core AI logic
├── requirements_clean.txt     # Dependencies
├── .env                       # Environment variables
├── chroma_integration.py      # ChromaDB integration
├── pinecone_integration.py    # Pinecone vector database
├── speed_to_text.py          # Speech-to-text
├── text_to_speech.py         # Text-to-speech
├── sched_appointment.py      # Appointment scheduling
├── image_analysis/           # Image processing module
│   ├── core.py              # OCR and analysis
│   ├── render.py            # Result rendering
│   └── schemas.py           # Data schemas
├── result_analysis/         # Result comparison
│   ├── core.py             # Analysis logic
│   └── render.py           # Result rendering
├── speech_module/          # Speech processing
│   ├── ffmpeg_decoding.py  # Audio processing
│   ├── tts_mp3_stream.py   # TTS streaming
│   └── test_streamlit_stt.py # STT testing
└── chroma_db/              # Local ChromaDB storage
```

## 🔧 Tính năng chi tiết

### 1. Phân tích hình ảnh y tế
- Upload ảnh đơn thuốc hoặc kết quả xét nghiệm
- OCR tự động với EasyOCR
- Phân loại tài liệu bằng AI
- Trích xuất thông tin có cấu trúc
- Lưu trữ vào Firebase

### 2. Tư vấn thông minh
- Phân loại câu hỏi tự động
- Tìm kiếm thông tin từ vector database
- Tạo phản hồi cá nhân hóa
- Hỗ trợ ngữ cảnh hội thoại

### 3. Xử lý giọng nói
- Ghi âm và nhận diện giọng nói
- Chuyển đổi văn bản thành giọng nói
- Phát âm thanh trực tiếp

### 4. Quản lý dữ liệu
- ChromaDB cho tìm kiếm cục bộ
- Pinecone cho tìm kiếm vector quy mô lớn
- Firebase cho lưu trữ kết quả
- Tự động phân loại và indexing

## ⚠️ Lưu ý quan trọng

- Thông tin chỉ mang tính tham khảo
- Luôn tham khảo bác sĩ chuyên khoa
- Không tự ý thay đổi liều lượng thuốc
- Kiểm tra kỹ thông tin trước khi sử dụng

## 🤝 Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng tạo issue hoặc pull request.

## 📄 License

Project này được phát triển cho mục đích giáo dục và nghiên cứu.