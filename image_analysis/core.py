import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from image_analysis.schemas import MedicineList, LabList
import easyocr
from PIL import Image
import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore

def init_firebase():
    if not firebase_admin._apps:  # Nếu chưa có app nào được khởi tạo
        cred = credentials.Certificate("baymax-a7a0d-firebase-adminsdk-fbsvc-f00628f505.json")
        firebase_admin.initialize_app(cred)
        print("Firebase initialized.")
    else:
        print("Firebase already initialized.")


# Load biến môi trường
load_dotenv()

# Gọi hàm khởi tạo
init_firebase()
db = firestore.client()

llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL"),
)

# OCR tiếng Việt
def image_to_text(uploaded_file):
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    if image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    reader = easyocr.Reader(['vi'], gpu=False)
    results = reader.readtext(image_np)
    texts = [{"text": text, "confidence": conf} for _, text, conf in results]
    text_for_prompt = "\n".join(
        f"{item['text']} (độ tin cậy: {item['confidence']:.2f})"
        for item in texts
    )
    return text_for_prompt

# Prompt phân loại
classification_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Phân loại tài liệu sau thành 1 trong các loại: đơn thuốc, kết quả xét nghiệm, unknown. 
Nội dung dưới đây là kết quả OCR từ ảnh, mỗi dòng gồm nội dung và độ tin cậy:
{text}
Chỉ trả về duy nhất tên loại.
"""
)

def classify_doc_type(text: str) -> str:
    try:
        chain = classification_prompt | llm
        result = chain.invoke({"text": text})
        print("Phân loại LLM trả về:", result)
        return result.content.strip().lower()
    except Exception as e:
        print("Lỗi khi gọi OpenAI:", e)
        return "unknown"

def analyze_medicine_with_knowledge(ocr_text: str) -> list:
    """
    Dùng OpenAI để tìm hiểu thông tin thuốc từ nội dung OCR của đơn thuốc.
    """
    knowledge_prompt = PromptTemplate(
        input_variables=["ocr_text"],
        template="""
    Bạn là bác sĩ y khoa.
    
    Dưới đây là nội dung OCR từ ảnh đơn thuốc (có thể chứa nhiều tên thuốc, liều lượng, cách dùng):
    
    {ocr_text}
    
    Yêu cầu:
    1. Xác định tất cả các thuốc có trong nội dung trên.
    2. Xác định ngày tháng xuất hiện trong nội dung OCR. ĐẶC BIỆT chú ý lấy đúng ngày kê đơn hoặc ngày xuất hiện cuối cùng trên tài liệu (không lấy các ngày khác như ngày sinh, ngày hẹn tái khám). Đảm bảo chỉ lấy đúng một ngày duy nhất là ngày kê đơn hoặc ngày xuất hiện cuối đơn thuốc.
        - Nếu tài liệu có ngày kê đơn hoặc ngày xuất hiện cuối cùng trên tài liệu → trích xuất đúng ngày đó.
        - Format ngày đúng chuẩn ISO datetime, ví dụ: "2025-08-12T00:00:00".
        - Nếu nhiều ngày xuất hiện: Ưu tiên ngày gần nhất với các từ khoá như “ngày kê đơn”, “ngày”, “ngày cấp”, "date", "prescription date", "issued date", hoặc ngày ở cuối tài liệu.
        - Nếu không có ngày nào được ghi rõ ràng → dùng ngày, giờ hiện tại ở Việt Nam, format ISO datetime, ví dụ: "2025-08-12T00:00:00".
        - Tuyệt đối không lấy ngày sinh, ngày tái khám, hoặc các ngày không liên quan.
    
    3. Với mỗi thuốc, cung cấp thông tin theo schema JSON sau:
       - medicine_name: giữ nguyên tên thuốc (bao gồm cả hàm lượng nếu có)
       - effect: tác dụng chính của thuốc
       - side_effects: tác dụng phụ hoặc lưu ý khi dùng
       - interaction_with_history: tương tác với tiền sử bệnh của bệnh nhân
    
    4. Tiền sử bệnh nhân: béo phì
    
    5. Trả về JSON duy nhất theo schema:
    {{
        "document_date": "YYYY-MM-DDTHH:MM:SS",  // ngày tài liệu, nếu không tìm thấy thì dùng ngày hiện tại ở Việt Nam
        "medicines": [
            {{
                "medicine_name": "...",
                "effect": "...",
                "side_effects": "...",
                "interaction_with_history": "..."
            }}
        ]
    }}
    
    6. Không kèm bất kỳ giải thích hoặc văn bản ngoài JSON. Chỉ trả về JSON thuần.
    """
    )


    chain = knowledge_prompt | llm.with_structured_output(MedicineList)
    result = chain.invoke({"ocr_text": ocr_text})
    return result  # list[MedicineItem]


def analyze_lab_with_knowledge(lab_text: str) -> list:
    """
    Dùng OpenAI để giải thích ý nghĩa cho danh sách kết quả xét nghiệm.
    lab_text: nội dung OCR của bảng kết quả xét nghiệm
    """
    knowledge_prompt = PromptTemplate(
        input_variables=["lab_text"],
        template="""
    Bạn là chuyên gia xét nghiệm y khoa.
    
    Dưới đây là nội dung OCR từ một bảng kết quả xét nghiệm y tế:
    
    {lab_text}
    
    Tiền sử bệnh nhân: béo phì
    
    Yêu cầu:
    1. Xác định thời gian (ngày tháng) xuất hiện trong tài liệu document_date (ví dụ ngày thực hiện xét nghiệm)
    * Chuẩn hóa sang định dạng YYYY-MM-DD nếu có.
    * Nếu không tìm thấy thì trả về null.
    2. Xác định từng xét nghiệm riêng biệt.
    3. Với mỗi chỉ số, kiểm tra tính hợp lý của các thông tin sau: 
    * Tên chỉ số
    * Giá trị đo được
    * Khoảng tham chiếu
    * Nếu MỘT TRONG các thông tin trên bị thiếu, không hợp lý hoặc nghi ngờ sai (ví dụ ký tự lạ, không phải số khi cần số, đơn vị không phù hợp, không có đơn vị, khoảng tham chiếu không logic), thì:
      - Ghi "Chưa rõ" cho các trường sai hoặc thiếu
      - evaluation = "Chưa rõ"
      - explanation = "Không đủ thông tin để phân tích"
    * Nếu tất cả thông tin hợp lý, mới tiến hành so sánh Giá trị đo được với Khoảng tham chiếu:
      - evaluation = "Ổn" nếu nằm trong khoảng tham chiếu
      - evaluation = "Không ổn" nếu nằm ngoài khoảng tham chiếu
      - explanation = Mô tả tác động của chỉ số đó đến sức khỏe bệnh nhân
    4. Trả về danh sách JSON (list), mỗi phần tử có các trường:
      - test_name (Tên xét nghiệm)
      - value (Giá trị đo được)
      - unit (Đơn vị đo)
      - range (Khoảng tham chiếu)
      - evaluation (Đánh giá kết hợp tiền sử bệnh nhân nếu liên quan)
      - explanation (Mô tả tác động của chỉ số đó đến sức khỏe bệnh nhân)
    5. Trả về JSON theo schema:
        {{
            "document_date": "YYYY-MM-DD",  // ngày tài liệu, nếu không tìm thấy thì dùng ngày hiện tại UTC
            "lab": [
                {{
                    "test_name": "...",
                    "value": "...",
                    "unit": "...",
                    "range": "...",
                    "evaluation": "...",
                    "explanation": "..."
                }}
            ]
        }}
    - Bắt buộc mọi phần tử trong list đều có đầy đủ các trường trên, nế trường nào không có thông tin thì trả về "Chưa rõ".
    - Chỉ trả về JSON thuần dạng mảng (list), không kèm giải thích bên ngoài.
    
    6. Quy tắc cho trường "document_date":
    - Nếu tài liệu có ngày xét nghiệm → trích xuất đúng ngày đó, format ISO datetime (ví dụ: "2025-08-12T00:00:00").
    - Nếu không có ngày → dùng ngày hiện tại (theo giờ server khi xử lý yêu cầu) và format ISO datetime như trên.
    """
    )

    chain = knowledge_prompt | llm.with_structured_output(LabList)
    return chain.invoke({"lab_text": lab_text})

def normalize_test_name(name: str) -> str:
    """Chuẩn hoá tên chỉ số để query dễ hơn."""
    return name.strip().lower().replace(" ", "_")

def save_lab_results_grouped(lab_list, user_id: str, document_date):
    """
    Lưu bộ xét nghiệm nguyên vẹn (grouped) vào collection lab_results_grouped.
    - user_id: mã người dùng
    - document_date: datetime object, ngày xét nghiệm
    - lab_list: list of LabItem (hoặc dict tương tự)
    """
    # Chuyển lab_list (có thể là Pydantic model) thành list dict
    results = [item.model_dump() if hasattr(item, "model_dump") else item for item in lab_list]

    doc_ref = db.collection("lab_results_grouped").document(f"{user_id}_{document_date.strftime('%Y%m%d')}")
    data = {
        "user_id": user_id,
        "document_date": document_date,
        "results": results,
    }
    doc_ref.set(data)

def save_medicine_list_grouped(medicine_list, user_id: str, document_date):
    """
    Lưu đơn thuốc nguyên vẹn (grouped) vào collection medicine_lists_grouped.
    - user_id: mã người dùng
    - document_date: datetime object, ngày của đơn thuốc
    - medicine_list: list of MedicineItem (hoặc dict tương tự)
    """
    # Chuyển medicine_list (có thể là Pydantic model) thành list dict
    print("---go in medicine")
    print("document_date",document_date)
    print("medicine_list", medicine_list)
    medicines = [item.model_dump() if hasattr(item, "model_dump") else item for item in medicine_list]

    doc_ref = db.collection("medicine_lists_grouped").document(f"{user_id}_{document_date.strftime('%Y%m%d')}")
    data = {
        "user_id": user_id,
        "document_date": document_date,
        "medicines": medicines,
    }
    doc_ref.set(data)

def process_image_pipeline(image_path: str):
    print("🔍 Bắt đầu OCR...")
    text = image_to_text(image_path)
    print("📄 Kết quả OCR:\n", text)

    print("📂 Đang phân loại tài liệu...")
    doc_type = classify_doc_type(text)
    print("📌 Loại tài liệu:", doc_type)

    item = None
    if doc_type == "đơn thuốc":
        item = analyze_medicine_with_knowledge(text)
        print("item", item)
        save_medicine_list_grouped(item.medicines, 'A12345', item.document_date)
    elif doc_type == "kết quả xét nghiệm":
        item = analyze_lab_with_knowledge(text)
        print('item', item)
        save_lab_results_grouped(item.lab, 'A12345', item.document_date)

    return {
        "doc_type": doc_type,
        "structured_data": item,
        "text": text
    }
