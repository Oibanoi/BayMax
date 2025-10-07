# Kiến trúc hệ thống MedGuide AI

## 🏗️ Tổng quan kiến trúc

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   AI Engine     │    │   Data Layer    │
│   (Streamlit)   │◄──►│   (OpenAI)      │◄──►│   (Vector DB)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Speech I/O    │    │   Image OCR     │    │   Firebase      │
│   (STT/TTS)     │    │   (EasyOCR)     │    │   (Storage)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📱 Frontend Layer - Streamlit

### Thành phần chính:
- **simple_app.py**: Giao diện người dùng chính
- **Chat Interface**: Hỗ trợ text, voice, image input
- **File Upload**: Upload tài liệu y tế (.txt, .pdf, .docx)
- **Audio Controls**: Record, playback, TTS

### Tính năng UI:
- Real-time chat với AI
- Image preview và analysis
- Audio recording và playback
- File management sidebar
- Responsive design

## 🧠 AI Engine Layer

### 1. Core AI (main.py)
```python
class MedGuideAI:
    - classify_user_query()     # Phân loại câu hỏi
    - process_user_query()      # Xử lý pipeline chính
    - analyze_medical_image()   # Phân tích hình ảnh
    - generate_response()       # Tạo phản hồi
```

### 2. Query Classification
- **Input**: User text/voice
- **Categories**: 
  - symptoms (triệu chứng)
  - drug_groups (thuốc)
  - lab_results (xét nghiệm)
  - sched_appointment (đặt lịch)
  - compare_* (so sánh)
- **Output**: Classified topic

### 3. Response Generation Pipeline
```
User Input → Classification → Vector Search → Context Retrieval → AI Response
```

## 🖼️ Image Analysis Module

### image_analysis/core.py
```python
def process_image_pipeline():
    1. OCR với EasyOCR
    2. Phân loại tài liệu (đơn thuốc/xét nghiệm)
    3. Structured data extraction
    4. Lưu vào Firebase
```

### Supported Documents:
- **Đơn thuốc**: Medicine name, dosage, effects, interactions
- **Xét nghiệm**: Test values, ranges, evaluations, explanations

### OCR Pipeline:
```
Image → EasyOCR → Text → LLM Classification → Structured Data
```

## 🗄️ Data Layer

### 1. Vector Databases

#### ChromaDB (Local)
- **Collections**: symptoms, drug_groups, lab_results
- **Usage**: Development và local search
- **Storage**: ./chroma_db/

#### Pinecone (Cloud)
- **Index**: medical-rag-index
- **Dimensions**: 1536 (text-embedding-3-small)
- **Metric**: Cosine similarity
- **Usage**: Production vector search

### 2. Firebase Firestore
```
Collections:
├── lab_results_grouped/
│   └── {user_id}_{timestamp}
│       ├── user_id
│       ├── document_date
│       └── results[]
└── medicine_lists_grouped/
    └── {user_id}_{date}
        ├── user_id
        ├── document_date
        └── medicines[]
```

### 3. Data Flow
```
User Upload → OCR → Classification → Vector Embedding → Storage
                                                      ↓
                                              Pinecone + Firebase
```

## 🎤 Speech Processing

### Speech-to-Text (STT)
- **Library**: SpeechRecognition
- **Input**: Audio bytes từ Streamlit
- **Output**: Vietnamese text
- **File**: speed_to_text.py

### Text-to-Speech (TTS)
- **Library**: Edge-TTS
- **Input**: AI response text
- **Output**: MP3 audio stream
- **Features**: Real-time playback, multiple voices

### Audio Pipeline:
```
Microphone → Audio Bytes → STT → Text → AI → TTS → Audio Output
```

## 🔄 Processing Workflows

### 1. Text Query Workflow
```
1. User inputs text/voice
2. Classification (symptoms/drugs/labs/etc.)
3. Vector search in relevant collection
4. Context retrieval
5. LLM generation với context
6. Response + TTS audio
```

### 2. Image Analysis Workflow
```
1. User uploads image
2. OCR extraction
3. Document type classification
4. Structured data extraction
5. Firebase storage
6. Analysis response
```

### 3. Comparison Workflow
```
1. User requests comparison
2. Retrieve historical data from Firebase
3. Compare latest vs previous results
4. Generate comparison analysis
5. Render comparison table/chart
```

## 🔧 Configuration Management

### Environment Variables (.env)
```env
# AI Services
AZURE_OPENAI_ENDPOINT
OPENAI_API_KEY
PINECONE_API_KEY

# Models
OPENAI_MODEL=GPT-4o-mini
AZURE_DEPLOYMENT_NAME=text-embedding-3-small
```

### Service Initialization
```python
# AI Services
client = AzureOpenAI(...)
pinecone_db = MedicalPineconeDB()
chroma_db = MedicalChromaDB()

# Firebase
firebase_admin.initialize_app(cred)
```

## 📊 Performance Considerations

### Caching Strategy
- **Streamlit**: @st.cache_resource cho AI models
- **Session State**: Conversation history, context
- **Vector DB**: Embedded search results

### Scalability
- **Vector Search**: Pinecone auto-scaling
- **AI Calls**: Rate limiting và retry logic
- **Storage**: Firebase auto-scaling
- **Compute**: Stateless design

### Memory Management
- **Images**: Process và dispose immediately
- **Audio**: Streaming processing
- **Embeddings**: Batch processing
- **Context**: Limited conversation history

## 🔒 Security & Privacy

### Data Protection
- **API Keys**: Environment variables only
- **User Data**: Anonymized user IDs
- **Medical Data**: Encrypted in transit
- **Local Storage**: Temporary processing only

### Access Control
- **Firebase**: Service account authentication
- **Pinecone**: API key authentication
- **Azure**: Managed identity support

## 🚀 Deployment Architecture

### Development
```
Local Machine → Streamlit Dev Server → Local ChromaDB + Cloud Services
```

### Production (Recommended)
```
Docker Container → Streamlit App → Load Balancer → Cloud Services
                                      ↓
                              Pinecone + Firebase + Azure OpenAI
```

### Monitoring
- **Logs**: Structured logging với levels
- **Metrics**: Response times, error rates
- **Health Checks**: Service availability
- **Alerts**: Critical error notifications

## 🔄 Future Enhancements

### Planned Features
- [ ] Multi-user authentication
- [ ] Advanced analytics dashboard
- [ ] Mobile app support
- [ ] Offline mode capabilities
- [ ] Integration với EHR systems
- [ ] Multi-language support
- [ ] Advanced ML models

### Technical Improvements
- [ ] Microservices architecture
- [ ] GraphQL API
- [ ] Real-time notifications
- [ ] Advanced caching layer
- [ ] CI/CD pipeline
- [ ] Automated testing suite