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
       
        # Function definitions cho OpenAI function calling
        self.functions = [
            {
                "name": "analyze_lab_results",
                "description": "Phân tích chi tiết kết quả xét nghiệm và đưa ra khuyến nghị",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "test_results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "test_name": {"type": "string"},
                                    "value": {"type": "number"},
                                    "unit": {"type": "string"},
                                    "reference_range": {"type": "string"}
                                },
                                "required": ["test_name", "value"]
                            }
                        },
                        "patient_age": {"type": "number"},
                        "patient_gender": {"type": "string"}
                    },
                    "required": ["test_results"]
                }
            },
            {
                "name": "analyze_prescription",
                "description": "Phân tích đơn thuốc và kiểm tra tương tác thuốc",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "medications": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "drug_name": {"type": "string"},
                                    "dosage": {"type": "string"},
                                    "frequency": {"type": "string"},
                                    "duration": {"type": "string"}
                                },
                                "required": ["drug_name", "dosage", "frequency"]
                            }
                        },
                        "current_medications": {"type": "array", "items": {"type": "string"}},
                        "allergies": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["medications"]
                }
            },
            {
                "name": "assess_symptoms",
                "description": "Đánh giá triệu chứng và đưa ra khuyến nghị chuyên khoa",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symptoms": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "symptom": {"type": "string"},
                                    "severity": {"type": "string", "enum": ["mild", "moderate", "severe"]},
                                    "duration": {"type": "string"},
                                    "triggers": {"type": "string"}
                                },
                                "required": ["symptom", "severity"]
                            }
                        },
                        "patient_age": {"type": "number"},
                        "medical_history": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["symptoms"]
                }
            },
            {
                "name": "create_health_plan",
                "description": "Tạo kế hoạch chăm sóc sức khỏe cá nhân hóa",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "health_goals": {"type": "array", "items": {"type": "string"}},
                        "current_conditions": {"type": "array", "items": {"type": "string"}},
                        "lifestyle_factors": {
                            "type": "object",
                            "properties": {
                                "exercise_level": {"type": "string"},
                                "diet_type": {"type": "string"},
                                "sleep_hours": {"type": "number"},
                                "stress_level": {"type": "string"}
                            }
                        }
                    },
                    "required": ["health_goals"]
                }
            }
        ]
 
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
- lab_results: Questions about test results, lab values, or medical examinations

Query: "{user_input}"

Return only the category name (symptoms/drug_groups/lab_results).
"""
            
            response = self.client.chat.completions.create(
                model="GPT-4o-mini",
                messages=[{"role": "user", "content": classification_prompt}],
                max_tokens=50,
                temperature=0
            )
            
            category = response.choices[0].message.content.strip().lower()
            return category if category in ['symptoms', 'drug_groups', 'lab_results'] else 'symptoms'
            
        except Exception as e:
            return 'symptoms'  # Default fallback
    
    def process_user_query(self, user_input: str):
        """Main processing pipeline: classify -> query -> generate"""
        try:
            # Step 1: Text classification
            topic = self.classify_user_query(user_input)
            print("topic:", topic)
            # Step 2: Query rel evant collection
            if topic == 'symptoms':
                search_results = self.chroma_db.search_symptoms(user_input, n_results=3)
            elif topic == 'drug_groups':
                search_results = self.chroma_db.search_drug_groups(user_input, n_results=3)
            else:  # lab_results
                search_results = self.chroma_db.search_lab_results(user_input, n_results=3)
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
 
    def process_with_function_calling(self, user_input: str, context_type: str = "general"):
        """Xử lý input với function calling"""
        try:
            # Tạo context-aware prompt
            context_summary = self.get_context_summary()
            enhanced_prompt = f"""
            CONTEXT BỆNH NHÂN:
            {context_summary}
           
            YÊU CẦU MỚI: {user_input}
           
            Hãy phân tích và sử dụng function phù hợp để xử lý yêu cầu này.
            """
           
            # Thêm vào lịch sử hội thoại
            self.add_conversation("user", user_input)
           
            # Tạo messages với context
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": enhanced_prompt}
            ]
           
            # Call OpenAI with function calling (using new tools format)
            response = self.client.chat.completions.create(
                model="GPT-4o-mini",
                messages=messages,
                tools=[{"type": "function", "function": func} for func in self.functions],
                tool_choice="auto",
                max_tokens=1500,
                temperature=0.3
            )
           
            response_message = response.choices[0].message
           
            if response_message.tool_calls:
                # Xử lý tool calls (new format)
                tool_call = response_message.tool_calls[0]
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
               
                # Gọi function tương ứng
                if func_name == "analyze_lab_results":
                    result = self.analyze_lab_results(**func_args)
                elif func_name == "analyze_prescription":
                    result = self.analyze_prescription(**func_args)
                elif func_name == "assess_symptoms":
                    result = self.assess_symptoms(**func_args)
                elif func_name == "create_health_plan":
                    result = self.create_health_plan(**func_args)
                else:
                    result = "Function không được hỗ trợ"
               
                # Tạo response cuối với kết quả function (using tool role)
                messages_with_tool = messages + [
                    response_message,
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    }
                ]
               
                final_response = self.client.chat.completions.create(
                    model="GPT-4o-mini",
                    messages=messages_with_tool,
                    max_tokens=1000,
                    temperature=0.3
                )
               
                final_answer = final_response.choices[0].message.content
                self.add_conversation("assistant", final_answer)
               
                return {
                    "function_used": func_name,
                    "function_result": json.loads(result),
                    "ai_interpretation": final_answer,
                    "conversation_id": len(st.session_state.conversation_history)
                }
            else:
                # Không có function call, trả về response thông thường
                answer = response_message.content
                self.add_conversation("assistant", answer)
               
                return {
                    "function_used": None,
                    "ai_response": answer,
                    "conversation_id": len(st.session_state.conversation_history)
                }
               
        except Exception as e:
            error_msg = f"Lỗi xử lý: {str(e)}"
            self.add_conversation("system", error_msg)
            return {"error": error_msg}
 
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
 
 
def main():
    """Main Streamlit application"""
   
    # Initialize MedGuide AI
    medguide = MedGuideAI()
   
    # Header
    st.title("🏥 MedGuide AI - Trợ lý Y tế Thông minh")
    st.markdown("### Tư vấn y tế thông minh với AI")
   
    # Sidebar for navigation
    st.sidebar.title("🔧 Chức năng")
    page = st.sidebar.selectbox(
        "Chọn chức năng:",
        [
            "🏥 Phân tích Y tế Tổng hợp",
            "🩺 Tư vấn triệu chứng",
            "📦 Xử lý batch",
            "👤 Quản lý thông tin bệnh nhân",
            "📊 Lịch sử & Context"
        ]
    )
   
    # Context summary in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Thông tin bệnh nhân")
    context_summary = medguide.get_context_summary()
    if context_summary != "Chưa có thông tin bệnh nhân":
        st.sidebar.text_area("Context hiện tại:", context_summary, height=100, disabled=True)
    else:
        st.sidebar.info("Chưa có thông tin bệnh nhân")
   
    # Main content based on selected page
    if page == "🏥 Phân tích Y tế Tổng hợp":
       
        # Initialize chat history in session state
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
       
       
        # Chat container with scrollable area
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_messages):
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(message["content"])
                        # Display image if exists
                        if "image" in message:
                            st.image(message["image"], caption="Hình ảnh đã gửi", use_container_width=True)
               
                elif message["role"] == "assistant":
                    with st.chat_message("assistant"):
                        st.markdown(message["content"])
                       
                        # Display function details if available
                        if message.get("function_used"):
                            with st.expander(f"🎯 Đã sử dụng Function: {message['function_used']}"):
                                st.json(message.get("function_result", {}))
                        
                        # Display classification and search results
                        if message.get("topic_classified"):
                            topic_map = {
                                'symptoms': '🩺 Triệu chứng',
                                'drug_groups': '💊 Thuốc',
                                'lab_results': '🧪 Xét nghiệm'
                            }
                            with st.expander(f"📊 Phân loại: {topic_map.get(message['topic_classified'], message['topic_classified'])}"):
                                if message.get('search_results') and message['search_results'].get('documents'):
                                    st.write("**Thông tin tham khảo từ cơ sở dữ liệu:**")
                                    for i, doc in enumerate(message['search_results']['documents'][0][:3], 1):
                                        st.write(f"{i}. {doc}")
       
        # Chat input section (modern style)
        st.markdown("---")
       
        # Create form for better UX (closest to Enter behavior)
        with st.form(key="chat_input_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 2])
           
            with col1:
                # Use text_input for single line (more chat-like)
                user_input = st.text_input(
                    label="Tin nhắn",
                    placeholder="💬 Nhập câu hỏi y tế, mô tả triệu chứng... (nhấn Ctrl+Enter để gửi)",
                    label_visibility="collapsed",
                    key="main_chat_input"
                )
           
            with col2:
                # Combined controls in smaller space
                col_send, col_options, col_output_settings = st.columns([1, 1, 1])
                with col_send:
                    send_message = st.form_submit_button("📤", help="Gửi tin nhắn", use_container_width=True)
                with col_options:
                    # Use expander for advanced options
                    with st.expander("⚙️"):
                        query_type = st.selectbox(
                            "Loại:",
                            ["general", "symptom", "lab", "prescription"],
                            help="Tối ưu AI theo loại câu hỏi",
                            key="chat_query_type"
                        )
                with col_output_settings:
                    audio_bytes_res = st.audio_input("Speak something...", key="audio_recorder")

                    if audio_bytes_res:
                        # Display the audio player
                        st.audio(audio_bytes_res, format="audio/wav")
                        print("thong hihihahah")

                        # Create a download button
                        # st.download_button(
                        #     label="Download Audio",
                        #     data=audio_bytes,
                        #     file_name="recorded_audio.wav",
                        #     mime="audio/wav"
                        # )

                        result = sp.speech_to_text(audio_bytes_res)
                        print(result)
       
        # Image upload and additional controls outside form
        col1, col2, col3 = st.columns([2, 1, 1])
       
        with col1:
            # More compact image upload (with dynamic key to clear after send)
            upload_key = f"chat_image_upload_{st.session_state.get('upload_counter', 0)}"
            uploaded_image = st.file_uploader(
                "📷 Kèm hình ảnh y tế",
                type=['jpg', 'jpeg', 'png'],
                help="Đơn thuốc, xét nghiệm...",
                key=upload_key
            )
       
        with col2:
            # Text area for longer messages
            if st.button("📝 Tin nhắn dài", help="Mở khung soạn tin nhắn dài"):
                st.session_state.show_long_message = not st.session_state.get('show_long_message', False)
       
        with col3:
            clear_chat = st.button("🗑️ Xóa chat", type="secondary", help="Xóa toàn bộ lịch sử")
       
        # Long message input (toggle)
        if st.session_state.get('show_long_message', False):
            with st.form(key="long_message_form", clear_on_submit=True):
                long_input = st.text_area(
                    "Tin nhắn dài:",
                    height=120,
                    placeholder="Mô tả chi tiết triệu chứng, tình trạng sức khỏe...",
                    key="long_message_input"
                )
                col1, col2 = st.columns([1, 4])
                with col1:
                    send_long = st.form_submit_button("📤 Gửi tin dài", type="primary")
                with col2:
                    if st.form_submit_button("❌ Đóng"):
                        st.session_state.show_long_message = False
                        st.rerun()
               
                # Handle long message
                if send_long and long_input.strip():
                    user_input = long_input
                    send_message = True
                    st.session_state.show_long_message = False
       
        # Preview uploaded image (compact)
        if uploaded_image is not None:
            with st.expander("🖼️ Xem trước hình ảnh", expanded=False):
                image = Image.open(uploaded_image)
                st.image(image, caption="Hình ảnh sẽ gửi", use_container_width=True)
       
        # Tips for better UX
        st.caption("💡 **Mẹo:** Nhấn Ctrl+Enter trong khung nhập để gửi nhanh, hoặc dùng nút 📝 cho tin nhắn dài")
       
        # Handle send message (works with both forms)
        if send_message and (user_input.strip() if 'user_input' in locals() else False):
            # Get query_type from session state if available
            if 'chat_query_type' not in st.session_state:
                query_type = "general"
           
            # Add user message to chat
            user_message = {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            }
            # Add image to message if uploaded (make a copy to avoid re-upload issue)
            current_image = None
            if uploaded_image is not None:
                # Store image content to avoid re-upload
                image_content = uploaded_image.read()
                uploaded_image.seek(0)  # Reset for processing
                user_message["image"] = uploaded_image
                user_message["has_image"] = True
                current_image = uploaded_image
           
            st.session_state.chat_messages.append(user_message)
           
            # Process with AI
            with st.spinner("🤖 MedGuide AI đang suy nghĩ..."):
                try:
                    if current_image is not None:
                        # Process with image analysis
                        current_image.seek(0)  # Reset file pointer
                       
                        # Determine analysis type from query_type
                        if query_type == "lab":
                            analysis_type = "lab_result"
                        elif query_type == "prescription":
                            analysis_type = "prescription"
                        else:
                            analysis_type = "general"
                       
                        ai_response = medguide.analyze_medical_image(current_image, analysis_type)
                       
                        # Add AI response to chat
                        assistant_message = {
                            "role": "assistant",
                            "content": ai_response,
                            "timestamp": datetime.now().isoformat(),
                            "has_image_analysis": True,
                            "image_type": analysis_type
                        }
                       
                    else:
                        # Use new pipeline: classify -> query -> generate
                        result = medguide.process_user_query(user_input)
                       
                        if "error" in result:
                            ai_response = f"❌ Xin lỗi, có lỗi xảy ra: {result['error']}"
                            assistant_message = {
                                "role": "assistant",
                                "content": ai_response,
                                "timestamp": datetime.now().isoformat()
                            }
                        else:
                            ai_response = result.get('ai_response', 'Xin lỗi, tôi không thể xử lý yêu cầu này.')
                           
                            assistant_message = {
                                "role": "assistant",
                                "content": ai_response,
                                "timestamp": datetime.now().isoformat(),
                                "topic_classified": result.get('topic_classified'),
                                "search_results": result.get('search_results')
                            }
                   
                    st.session_state.chat_messages.append(assistant_message)
                   
                    # Clear uploaded image after processing to prevent re-upload
                    if current_image is not None:
                        # Force clear the file uploader by incrementing session counter
                        if 'upload_counter' not in st.session_state:
                            st.session_state.upload_counter = 0
                        st.session_state.upload_counter += 1
                   
                    # Auto refresh after sending
                    st.rerun()
                   
                except Exception as e:
                    error_message = {
                        "role": "assistant",
                        "content": f"❌ Xin lỗi, có lỗi xảy ra khi xử lý yêu cầu: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.chat_messages.append(error_message)
                    st.rerun()
       
        # Handle clear chat
        if clear_chat:
            st.session_state.chat_messages = []
            st.rerun()
       
        # Chat statistics and tips
        if st.session_state.chat_messages:
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
           
            with col1:
                total_messages = len(st.session_state.chat_messages)
                user_messages = len([m for m in st.session_state.chat_messages if m["role"] == "user"])
                st.metric("Tổng tin nhắn", total_messages)
           
            with col2:
                image_messages = len([m for m in st.session_state.chat_messages if m.get("has_image")])
                st.metric("Tin nhắn có hình ảnh", image_messages)
           
            with col3:
                function_calls = len([m for m in st.session_state.chat_messages if m.get("function_used")])
                st.metric("Function calls", function_calls)
       
        else:
            # Show welcome message and tips when no chat history
            st.info("""
            👋 **Chào mừng đến với MedGuide AI!**
           
            🗣️ **Bạn có thể:**
            - Hỏi về triệu chứng, thuốc, xét nghiệm
            - Upload hình ảnh đơn thuốc hoặc kết quả xét nghiệm để phân tích
            - Trò chuyện liên tục với AI để được tư vấn chi tiết
           
            💡 **Cách sử dụng nhanh:**
            - Nhập tin nhắn ngắn ở khung trên và nhấn 📤 hoặc Ctrl+Enter
            - Dùng 📝 cho tin nhắn dài hoặc mô tả chi tiết
            - Chọn đúng loại câu hỏi trong ⚙️ để AI phân tích chính xác
            - Kéo thả hình ảnh vào 📷 để phân tích kèm theo
            """)
           
        st.audio(st.session_state.audio_bytes, format="audio/mp3")
    elif page == "🩺 Tư vấn triệu chứng":
        st.header("🩺 Tư vấn triệu chứng")
       
        with st.form("symptoms_form"):
            col1, col2 = st.columns(2)
           
            with col1:
                symptoms_text = st.text_area("Mô tả triệu chứng:", height=150)
                medical_history = st.text_area("Tiền sử bệnh (nếu có):", height=100)
           
            with col2:
                age = st.number_input("Tuổi:", min_value=0, max_value=150, value=30)
                severity = st.selectbox("Mức độ nghiêm trọng:", ["mild", "moderate", "severe"])
                duration = st.text_input("Thời gian xuất hiện:")
           
            submitted = st.form_submit_button("🩺 Tư vấn")
           
            if submitted and symptoms_text:
                query = f"""
                Đánh giá triệu chứng:
                - Triệu chứng: {symptoms_text}
                - Tuổi: {age}
                - Mức độ: {severity}
                - Thời gian: {duration}
                - Tiền sử: {medical_history}
                """
               
                with st.spinner("Đang phân tích triệu chứng..."):
                    result = medguide.process_with_function_calling(query, "symptom_assessment")
                   
                    if result.get('function_used'):
                        st.success(f"🎯 Function: **{result['function_used']}**")
                       
                        # Display structured results
                        if 'function_result' in result:
                            func_result = result['function_result']
                           
                            col1, col2 = st.columns(2)
                           
                            with col1:
                                if 'urgency_level' in func_result:
                                    urgency = func_result['urgency_level']
                                    if urgency == "emergency":
                                        st.error(f"🚨 Mức độ khẩn cấp: **{urgency.upper()}**")
                                    elif urgency == "urgent":
                                        st.warning(f"⚠️ Mức độ khẩn cấp: **{urgency}**")
                                    else:
                                        st.info(f"ℹ️ Mức độ khẩn cấp: **{urgency}**")
                               
                                if 'recommended_specialists' in func_result:
                                    specialists = func_result['recommended_specialists']
                                    if specialists:
                                        st.markdown("**🏥 Chuyên khoa được khuyên:**")
                                        for spec in specialists:
                                            st.markdown(f"- {spec}")
                           
                            with col2:
                                if 'immediate_actions' in func_result:
                                    actions = func_result['immediate_actions']
                                    st.markdown("**⚡ Hành động cần thiết:**")
                                    for action in actions:
                                        st.markdown(f"- {action}")
                   
                    response = result.get('ai_interpretation', result.get('ai_response', ''))
                    st.markdown("### 🩺 Tư vấn chi tiết:")
                    st.markdown(response)
   
    elif page == "📦 Xử lý batch":
        st.header("📦 Xử lý batch nhiều yêu cầu")
       
        st.markdown("### Nhập nhiều yêu cầu để xử lý cùng lúc")
       
        # Initialize batch requests in session state
        if 'batch_requests' not in st.session_state:
            st.session_state.batch_requests = [{"text": "", "type": "general"}]
       
        # Controls outside form for adding/removing requests
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("➕ Thêm yêu cầu"):
                st.session_state.batch_requests.append({"text": "", "type": "general"})
                st.rerun()
       
        with col2:
            if st.button("➖ Xóa yêu cầu cuối") and len(st.session_state.batch_requests) > 1:
                st.session_state.batch_requests.pop()
                st.rerun()
       
        with st.form("batch_form"):
            for i, request in enumerate(st.session_state.batch_requests):
                st.markdown(f"#### Yêu cầu {i+1}")
                col1, col2 = st.columns([3, 1])
               
                with col1:
                    text = st.text_area(f"Nội dung yêu cầu {i+1}:", value=request["text"], key=f"batch_text_{i}")
                with col2:
                    req_type = st.selectbox(
                        f"Loại yêu cầu {i+1}:",
                        ["general", "symptom", "lab", "prescription"],
                        index=["general", "symptom", "lab", "prescription"].index(request["type"]),
                        key=f"batch_type_{i}"
                    )
               
                st.session_state.batch_requests[i] = {"text": text, "type": req_type}
                st.markdown("---")
           
            # Form submit button
            submitted = st.form_submit_button("📦 Xử lý Batch", type="primary")
           
            if submitted:
                # Filter valid requests
                valid_requests = [req for req in st.session_state.batch_requests if req["text"].strip()]
               
                if valid_requests:
                    st.markdown("### 🔄 Đang xử lý batch...")
                   
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                   
                    results = []
                    for i, request in enumerate(valid_requests):
                        status_text.text(f"Xử lý yêu cầu {i+1}/{len(valid_requests)}")
                        progress_bar.progress((i + 1) / len(valid_requests))
                       
                        # Use new pipeline for batch processing
                        if request["type"] in ["symptom", "lab", "prescription"]:
                            result = medguide.process_user_query(request["text"])
                        else:
                            result = medguide.process_with_function_calling(request["text"], request["type"])
                        result["batch_index"] = i + 1
                        result["input_type"] = request["type"]
                        results.append(result)
                   
                    status_text.text("✅ Hoàn thành xử lý batch!")
                   
                    # Display results
                    st.markdown("### 📊 Kết quả Batch")
                    for i, result in enumerate(results):
                        with st.expander(f"📋 Kết quả {i+1} ({result.get('input_type', 'N/A')})"):
                            if result.get('function_used'):
                                st.info(f"🎯 Function: **{result['function_used']}**")
                           
                            response = result.get('ai_interpretation', result.get('ai_response', 'Không có phản hồi'))
                            st.markdown(response)
                else:
                    st.warning("⚠️ Vui lòng nhập ít nhất một yêu cầu hợp lệ.")
   
    elif page == "👤 Quản lý thông tin bệnh nhân":
        st.header("👤 Quản lý thông tin bệnh nhân")
       
        col1, col2 = st.columns(2)
       
        with col1:
            st.markdown("### ➕ Thêm thông tin mới")
            with st.form("patient_info_form"):
                category = st.selectbox(
                    "Loại thông tin:",
                    ["medical_history", "medications", "allergies", "symptoms_timeline"]
                )
               
                data = st.text_area("Nội dung thông tin:", height=100)
               
                submitted = st.form_submit_button("💾 Lưu thông tin")
               
                if submitted and data:
                    medguide.add_to_context(category, data)
                    st.success(f"✅ Đã thêm thông tin vào {category}")
                    st.rerun()
       
        with col2:
            st.markdown("### 📋 Thông tin hiện tại")
           
            for category, items in st.session_state.patient_context.items():
                if items:
                    with st.expander(f"{category.replace('_', ' ').title()} ({len(items)})"):
                        for i, item in enumerate(reversed(items[-5:])):  # Show last 5 items
                            st.markdown(f"**{len(items)-i}.** {item['data']}")
                            st.caption(f"Thời gian: {item['timestamp'][:19]}")
                else:
                    st.info(f"{category.replace('_', ' ').title()}: Chưa có thông tin")
       
        # Clear data option
        st.markdown("---")
        if st.button("🗑️ Xóa tất cả dữ liệu bệnh nhân", type="secondary"):
            if st.button("⚠️ Xác nhận xóa", type="primary"):
                st.session_state.patient_context = {
                    "medical_history": [],
                    "medications": [],
                    "allergies": [],
                    "symptoms_timeline": []
                }
                st.session_state.conversation_history = []
                st.success("🗑️ Đã xóa tất cả dữ liệu bệnh nhân")
                st.rerun()
   
    elif page == "📊 Lịch sử & Context":
        st.header("📊 Lịch sử hội thoại & Context")
       
        # Conversation history
        st.markdown("### 💬 Lịch sử hội thoại")
        if st.session_state.conversation_history:
            # Create DataFrame for better display
            conversations = []
            for conv in st.session_state.conversation_history:
                conversations.append({
                    "Thời gian": conv["timestamp"][:19],
                    "Vai trò": conv["role"],
                    "Nội dung": conv["content"][:100] + "..." if len(conv["content"]) > 100 else conv["content"]
                })
           
            df = pd.DataFrame(conversations)
            st.dataframe(df, use_container_width=True)
           
            # Detailed view
            st.markdown("### 🔍 Chi tiết hội thoại")
            for i, conv in enumerate(reversed(st.session_state.conversation_history[-10:])):
                with st.expander(f"{conv['role'].title()} - {conv['timestamp'][:19]}"):
                    st.markdown(conv['content'])
        else:
            st.info("Chưa có lịch sử hội thoại")
       
        # Context summary
        st.markdown("---")
        st.markdown("### 📋 Tóm tắt Context")
        context = medguide.get_context_summary()
        if context != "Chưa có thông tin bệnh nhân":
            st.text_area("Context hiện tại:", context, height=200, disabled=True)
        else:
            st.info("Chưa có thông tin context")
       
        # Statistics
        st.markdown("---")
        st.markdown("### 📈 Thống kê")
        col1, col2, col3, col4 = st.columns(4)
       
        with col1:
            st.metric("Số cuộc hội thoại", len(st.session_state.conversation_history))
        with col2:
            st.metric("Tiền sử bệnh", len(st.session_state.patient_context["medical_history"]))
        with col3:
            st.metric("Thuốc đang dùng", len(st.session_state.patient_context["medications"]))
        with col4:
            st.metric("Dị ứng", len(st.session_state.patient_context["allergies"]))
 
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>🏥 MedGuide AI - Trợ lý Y tế Thông minh với Function Calling, Batching & Context Management</p>
        <p><em>⚠️ Thông tin này chỉ mang tính tham khảo, cần tham khảo ý kiến bác sĩ chuyên khoa</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )
 
if __name__ == "__main__":
    main()