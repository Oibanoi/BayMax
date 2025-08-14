import streamlit as st
import json
import base64
from openai import AzureOpenAI
from PIL import Image
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
from dotenv import load_dotenv
from chroma_integration import MedicalChromaDB
import text_to_speech
import speed_to_text as sp

from langchain.chains import LLMChain
from image_analysis.core import process_image_pipeline
from image_analysis.render import render_prescription, render_lab
from image_analysis.schemas import LabList
from result_analysis.core import handle_get_result, handle_compare_list_result, handle_compare_list_medicines
from result_analysis.render import render_latest_result, render_lab_comparison, render_latest_prescription

# # Load environment variables
load_dotenv()
# Setup session_state for audio caching
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None
# Page configuration
st.set_page_config(
    page_title="MedGuide AI - Trợ lý Y tế Thông minh",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
class MedGuideAI:
    def __init__(self):
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_version="2024-07-01-preview",
            azure_endpoint="https://aiportalapi.stu-platform.live/jpe",
            api_key="sk-dEyinSJuZ8V_u8gKuPksuA",
        )
        
        # Initialize ChromaDB
        self.chroma_db = MedicalChromaDB()
       
        # Initialize session state for context management
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'patient_context' not in st.session_state:
            st.session_state.patient_context = {
                "medical_history": [],
                "medications": [],
                "allergies": [],
                "symptoms_timeline": []
            }
       
        # System prompt cho MedGuide AI
        self.system_prompt = """
        Bạn là MedGuide AI - Trợ lý y tế thông minh và hữu ích.
       
        NHIỆM VỤ CHÍNH:
        1. Phân tích và giải thích kết quả xét nghiệm một cách chi tiết, dễ hiểu
        2. Phân tích đơn thuốc với thông tin về công dụng, cách dùng, lưu ý
        3. Đưa ra tư vấn và khuyến nghị dựa trên triệu chứng và dữ liệu
        4. Cung cấp lời khuyên về dinh dưỡng, lối sống phù hợp
       
        CÁCH TIẾP CẬN:
        - Phân tích chi tiết và đưa ra nhận xét cụ thể về từng chỉ số
        - Giải thích ý nghĩa của các kết quả bất thường
        - Đưa ra khuyến nghị dinh dưỡng và lối sống cụ thể
        - Gợi ý khi nào cần đi khám bác sĩ
        - Sử dụng ngôn ngữ thân thiện, dễ hiểu
       
        NGUYÊN TẮC AN TOÀN:
        - Luôn kết thúc với: "Đây là thông tin tham khảo, bạn nên tham khảo bác sĩ để có hướng điều trị chính xác"
        - Không tự ý chẩn đoán bệnh cụ thể
        - Khuyến khích thăm khám chuyên khoa khi cần thiết
       
        Hãy trả lời một cách chi tiết, hữu ích và thực tế để người dùng hiểu rõ tình trạng sức khỏe của mình.
        """

    # Context Management Methods
    def add_to_context(self, category: str, data: Any):
        """Thêm thông tin vào context của bệnh nhân"""
        if category in st.session_state.patient_context:
            st.session_state.patient_context[category].append({
                "timestamp": datetime.now().isoformat(),
                "data": data
            })
   
    def add_conversation(self, role: str, content: str):
        """Thêm hội thoại vào lịch sử"""
        st.session_state.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
   
    def get_context_summary(self) -> str:
        """Tạo tóm tắt context để đưa vào prompt"""
        context_parts = []
       
        if st.session_state.patient_context["medical_history"]:
            history = [item["data"] for item in st.session_state.patient_context["medical_history"][-3:]]
            context_parts.append(f"Tiền sử bệnh: {'; '.join(history)}")
       
        if st.session_state.patient_context["medications"]:
            meds = [item["data"] for item in st.session_state.patient_context["medications"][-5:]]
            context_parts.append(f"Thuốc đang dùng: {'; '.join(meds)}")
       
        if st.session_state.patient_context["allergies"]:
            allergies = [item["data"] for item in st.session_state.patient_context["allergies"]]
            context_parts.append(f"Dị ứng: {'; '.join(allergies)}")
       
        return "\n".join(context_parts) if context_parts else "Chưa có thông tin bệnh nhân"
    
    def classify_user_query(self, user_input: str) -> str:
        """Classify user query into topic categories"""
        try:
            classification_prompt = f"""
                Classify the following medical query into one of these categories:
                - symptoms: Questions about symptoms, signs, or medical conditions
                - drug_groups: Questions about medications, drugs, or prescriptions
                - get_prescription: Question about getting prescriptions
                - get_lab_results: Question about getting lab results
                - compare_prescription: Question about comparing prescriptions
                - compare_lab_results: Question about comparing lab results

                Query: "{user_input}"

                Return only the category name (symptoms/drug_groups/get_prescription/get_lab_results/compare_prescription/compare_lab_results).
                """
            
            response = self.client.chat.completions.create(
                model="GPT-4o-mini",
                messages=[{"role": "user", "content": classification_prompt}],
                max_tokens=50,
                temperature=0
            )
            
            category = response.choices[0].message.content.strip().lower()
            return category if category in ['symptoms', 'drug_groups', 'get_prescription', 'get_lab_results', 'compare_prescription', 'compare_lab_results'] else 'symptoms'
            
        except Exception as e:
            return 'symptoms'  # Default fallback

    def process_user_query(self, user_input: str):
        """Main processing pipeline: classify -> query -> generate"""
        try:
            # Step 1: Text classification
            topic = self.classify_user_query(user_input)
            print("topic:", topic)
            search_results=""
            ai_response = "❌ Không tìm thấy kết quả phù hợp."
            # Step 2: Query rel evant collection
            if topic == "get_lab_results":
                latest_lab_results = handle_get_result("lab")
                if latest_lab_results is not None:
                    ai_response = render_latest_result(latest_lab_results)
            elif topic == "compare_lab_results":
                latest_lab_results = handle_get_result("lab", 2)
                prompt_result =  handle_compare_list_result(latest_lab_results)
                if prompt_result is not None:
                    ai_response = render_lab_comparison(prompt_result)
            elif topic == "get_prescription":
                latest_prescription_result = handle_get_result("prescription")
                print("latest_prescription_result", latest_prescription_result)
                ai_response = render_latest_prescription(latest_prescription_result)
            elif topic == "compare_prescription":
                latest_prescription_result = handle_get_result("prescription", 2)
                ai_response = handle_compare_list_medicines(latest_prescription_result)
            else:
                if topic == 'symptoms':
                    search_results = self.chroma_db.search_symptoms(user_input, n_results=3)
                elif topic == 'drug_groups':
                    search_results = self.chroma_db.search_drug_groups(user_input, n_results=3)

                print("search_results:", search_results)
                # Step 3: Text generation with context
                context_info = "\n".join(search_results['documents'][0]) if search_results['documents'] else "No relevant information found"

                generation_prompt = f"""
                    Based on the following medical information, provide a helpful response to the user's question.
                    
                    User Question: {user_input}
                    Topic Category: {topic}
                    
                    Relevant Information:
                    {context_info}
                    
                    Provide a detailed, helpful response in Vietnamese. Always end with: "Đây là thông tin tham khảo, bạn nên tham khảo bác sĩ để có hướng điều trị chính xác"
                    """

                response = self.client.chat.completions.create(
                    model="GPT-4o-mini",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": generation_prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.3
                )
            
                ai_response = response.choices[0].message.content

            # # thong's code start
            print("ai_response: " + ai_response)
            st.session_state.audio_bytes = text_to_speech.run_audio(ai_response)
            # # thong's code end
            
            # Add to conversation history
            self.add_conversation("user", user_input)
            self.add_conversation("assistant", ai_response)
            
            return {
                "topic_classified": topic,
                "search_results": search_results,
                "ai_response": ai_response,
                "conversation_id": len(st.session_state.conversation_history)
            }
            
        except Exception as e:
            error_msg = f"Lỗi xử lý: {str(e)}"
            return {"error": error_msg}
    
    def encode_image(self, image_file):
        """Encode image to base64 for OpenAI Vision API"""
        try:
            return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            return None
 
    def analyze_medical_image(self, image_file, query_type="general"):
        """Phân tích hình ảnh y tế với Vision API"""
        try:
            result = process_image_pipeline(image_file)
            ai_response = ""

            if result:
                if result["doc_type"] == "đơn thuốc":
                    meds = result["structured_data"]  # chắc chắn là list
                    ai_response = render_prescription(meds.medicines)
                elif result["doc_type"] == "kết quả xét nghiệm":
                    labs_structured: LabList = result["structured_data"]  # đây là LabList object
                    labs = labs_structured.lab  # lấy list LabItem bên trong
                    ai_response = render_lab(labs)
                else:
                    ai_response = "❓ Loại tài liệu chưa được hỗ trợ."
            else:
                ai_response = "Không nhận diện được nội dung từ ảnh."

            return ai_response

        except Exception as e:
            return f"Lỗi khi phân tích hình ảnh: {str(e)}"

    def analyze_lab_results(self, test_results: List[Dict], patient_age: int = None, patient_gender: str = None):
        """Phân tích chi tiết kết quả xét nghiệm và đưa ra tư vấn"""
        analysis = []
        abnormal_findings = []
        recommendations = []
       
        for test in test_results:
            test_name = test.get("test_name", "")
            value = test.get("value", 0)
            unit = test.get("unit", "")
           
            # Phân tích chi tiết từng chỉ số
            if "glucose" in test_name.lower() or "đường huyết" in test_name.lower():
                if value > 126:
                    status = "Cao hơn bình thường - có thể chỉ ra nguy cơ tiểu đường"
                    abnormal_findings.append(f"Đường huyết cao ({value} {unit})")
                    recommendations.extend([
                        "Giảm tiêu thụ đường và carbohydrate tinh chế",
                        "Tăng cường hoạt động thể chất (đi bộ 30 phút/ngày)",
                        "Chia nhỏ bữa ăn trong ngày",
                        "Theo dõi cân nặng"
                    ])
                elif value < 70:
                    status = "Thấp hơn bình thường - có thể do nhịn ăn hoặc vấn đề sức khỏe khác"
                    abnormal_findings.append(f"Đường huyết thấp ({value} {unit})")
                    recommendations.extend([
                        "Ăn đủ bữa, không bỏ bữa",
                        "Có sẵn kẹo hoặc nước ngọt khi cần",
                        "Theo dõi triệu chứng hạ đường huyết"
                    ])
                else:
                    status = "Trong giới hạn bình thường - tốt"
           
            elif "cholesterol" in test_name.lower() or "mỡ máu" in test_name.lower():
                if value > 240:
                    status = "Cao - tăng nguy cơ bệnh tim mạch"
                    abnormal_findings.append(f"Cholesterol cao ({value} {unit})")
                    recommendations.extend([
                        "Giảm thực phẩm nhiều chất béo bão hòa",
                        "Tăng omega-3 (cá, hạt óc chó)",
                        "Ăn nhiều rau xanh và trái cây",
                        "Tập thể dục đều đặn"
                    ])
                elif value > 200:
                    status = "Hơi cao - cần chú ý chế độ ăn"
                    recommendations.extend([
                        "Kiểm soát chế độ ăn",
                        "Tăng hoạt động thể chất"
                    ])
                else:
                    status = "Bình thường - tốt"
           
            elif "hemoglobin" in test_name.lower() or "hồng cầu" in test_name.lower():
                if value < 12 and patient_gender == "female":
                    status = "Thấp - có thể thiếu máu"
                    abnormal_findings.append(f"Hemoglobin thấp ({value} {unit})")
                    recommendations.extend([
                        "Ăn thực phẩm giàu sắt (thịt đỏ, gan, rau bina)",
                        "Kết hợp với vitamin C để tăng hấp thu sắt",
                        "Tránh uống trà/cà phê ngay sau bữa ăn"
                    ])
                elif value < 13 and patient_gender == "male":
                    status = "Thấp - có thể thiếu máu"
                    abnormal_findings.append(f"Hemoglobin thấp ({value} {unit})")
                else:
                    status = "Bình thường"
           
            else:
                # Phân tích chung cho các xét nghiệm khác
                status = "Cần tham khảo ý kiến bác sĩ để hiểu rõ ý nghĩa"
           
            analysis.append(f"• **{test_name}**: {value} {unit} - {status}")
       
        # Lưu vào context
        self.add_to_context("symptoms_timeline", f"Xét nghiệm: {', '.join([t['test_name'] for t in test_results])}")
       
        result = {
            "detailed_analysis": analysis,
            "abnormal_findings": abnormal_findings,
            "lifestyle_recommendations": list(set(recommendations)) if recommendations else ["Duy trì lối sống lành mạnh"],
            "follow_up_advice": "Theo dõi định kỳ và tham khảo bác sĩ để có kế hoạch điều chỉnh phù hợp"
        }
       
        return json.dumps(result, ensure_ascii=False, indent=2)
   
    def analyze_prescription(self, medications: List[Dict], current_medications: List[str] = None, allergies: List[str] = None):
        """Phân tích đơn thuốc chi tiết với thông tin hữu ích"""
        drug_analysis = []
        usage_tips = []
       
        # Lưu thuốc vào context
        for med in medications:
            self.add_to_context("medications", f"{med['drug_name']} {med['dosage']}")
       
        for med in medications:
            drug_name = med.get("drug_name", "")
            dosage = med.get("dosage", "")
            frequency = med.get("frequency", "")
            duration = med.get("duration", "")
           
            # Phân tích cơ bản theo tên thuốc (có thể mở rộng)
            usage_info = ""
            if any(keyword in drug_name.lower() for keyword in ["paracetamol", "acetaminophen"]):
                usage_info = " - Thuốc giảm đau, hạ sốt. Uống sau ăn, không quá 4g/ngày"
            elif any(keyword in drug_name.lower() for keyword in ["ibuprofen"]):
                usage_info = " - Thuốc chống viêm, giảm đau. Uống sau ăn để tránh đau dạ dày"
            elif any(keyword in drug_name.lower() for keyword in ["amoxicillin"]):
                usage_info = " - Kháng sinh. Uống đủ liều theo đơn, không tự ý ngừng"
            elif any(keyword in drug_name.lower() for keyword in ["omeprazole"]):
                usage_info = " - Thuốc dạ dày. Uống trước ăn sáng 30-60 phút"
           
            analysis = f"• **{drug_name}** ({dosage}, {frequency}){usage_info}"
            if duration:
                analysis += f" - Thời gian: {duration}"
               
            drug_analysis.append(analysis)
       
        # Lời khuyên chung
        general_tips = [
            "Uống thuốc đúng giờ theo chỉ định của bác sĩ",
            "Không tự ý tăng/giảm liều lượng",
            "Uống thuốc với nước lọc, tránh nước ngọt hoặc rượu bia",
            "Bảo quản thuốc nơi khô ráo, thoáng mát",
            "Thông báo với bác sĩ nếu có tác dụng phụ bất thường"
        ]
       
        result = {
            "medications_analysis": drug_analysis,
            "drug_interactions": [],
            "allergy_warnings": [],
            "usage_guidelines": general_tips,
            "important_notes": "Hoàn thành đủ liệu trình kháng sinh nếu có. Không chia sẻ thuốc với người khác."
        }
       
        return json.dumps(result, ensure_ascii=False, indent=2)
   
    def assess_symptoms(self, symptoms: List[Dict], patient_age: int = None, medical_history: List[str] = None):
        """Cung cấp thông tin giáo dục về các biểu hiện sức khỏe"""
        symptom_summary = []
        urgency_level = "routine"
        general_guidance = []
       
        # Lưu thông tin vào context
        for symptom in symptoms:
            self.add_to_context("symptoms_timeline", f"{symptom['symptom']} - {symptom.get('severity', 'unknown')}")
       
        for symptom in symptoms:
            symptom_name = symptom.get("symptom", "")
            severity = symptom.get("severity", "mild")
            duration = symptom.get("duration", "")
           
            if severity == "severe":
                urgency_level = "needs_attention"
           
            # Hướng dẫn chung
            if any(word in symptom_name.lower() for word in ["chest", "ngực"]):
                general_guidance.append("Tham khảo chuyên khoa tim mạch")
                if severity == "severe":
                    urgency_level = "immediate_care"
            elif any(word in symptom_name.lower() for word in ["head", "đầu"]):
                general_guidance.append("Tham khảo chuyên khoa thần kinh")
            elif any(word in symptom_name.lower() for word in ["cough", "ho"]):
                general_guidance.append("Tham khảo chuyên khoa hô hấp")
           
            symptom_summary.append(f"• {symptom_name} (mức độ: {severity}) - thời gian: {duration}")
       
        result = {
            "symptom_information": symptom_summary,
            "attention_level": urgency_level,
            "general_guidance": list(set(general_guidance)),
            "educational_info": ["Theo dõi các biểu hiện", "Ghi chép lại thời gian và mức độ"],
            "when_to_consult": ["Khi có biểu hiện bất thường", "Khi cần tư vấn chuyên môn"]
        }
       
        return json.dumps(result, ensure_ascii=False, indent=2)
   
    def create_health_plan(self, health_goals: List[str], current_conditions: List[str] = None, lifestyle_factors: Dict = None):
        """Tạo kế hoạch chăm sóc sức khỏe cá nhân hóa"""
        plan = {
            "health_goals": health_goals,
            "nutrition_plan": ["Ăn nhiều rau xanh", "Giảm đường và muối"],
            "exercise_plan": ["Đi bộ 30 phút/ngày", "Yoga 2-3 lần/tuần"],
            "monitoring_schedule": ["Kiểm tra sức khỏe định kỳ"]
        }
       
        return json.dumps(plan, ensure_ascii=False, indent=2)
