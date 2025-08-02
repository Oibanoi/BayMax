import chromadb
from chromadb.config import Settings
import uuid

class MedicalChromaDB:
    def __init__(self, persist_directory="./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.setup_collections()
        
    def setup_collections(self):
        """Create 3 collections and insert mock data"""
        # Collection 1: Symptoms
        self.symptoms_collection = self.client.get_or_create_collection(
            name="symptoms",
            metadata={"description": "Medical symptoms"}
        )
        
        # Collection 2: Drug groups
        self.drug_groups_collection = self.client.get_or_create_collection(
            name="drug_groups", 
            metadata={"description": "Drug group information"}
        )
        
        # Collection 3: Lab results
        self.lab_results_collection = self.client.get_or_create_collection(
            name="lab_results",
            metadata={"description": "Laboratory test results"}
        )
        
        self.insert_mock_data()
    
    def insert_mock_data(self):
        """Insert mock data into collections"""
        
        # Mock data for symptoms
        symptoms_data = [
            {"id": str(uuid.uuid4()), "text": "Đau đầu kéo dài, chóng mặt, buồn nôn", "category": "neurological", "severity": "moderate"},
            {"id": str(uuid.uuid4()), "text": "Ho khan, khó thở, đau ngực", "category": "respiratory", "severity": "severe"},
            {"id": str(uuid.uuid4()), "text": "Đau bụng, tiêu chảy, buồn nôn", "category": "gastrointestinal", "severity": "mild"},
            {"id": str(uuid.uuid4()), "text": "Sốt cao, ớn lạnh, đau cơ", "category": "infectious", "severity": "severe"},
            {"id": str(uuid.uuid4()), "text": "Mệt mỏi, chán ăn, sụt cân", "category": "systemic", "severity": "moderate"},
            {"id": str(uuid.uuid4()), "text": "Đau khớp, sưng khớp, cứng khớp buổi sáng", "category": "musculoskeletal", "severity": "moderate"},
            {"id": str(uuid.uuid4()), "text": "Đau ngực trái, hồi hộp, khó thở", "category": "cardiovascular", "severity": "severe"},
            {"id": str(uuid.uuid4()), "text": "Nổi mẩn đỏ, ngứa, sưng", "category": "dermatological", "severity": "mild"}
        ]
        
        self.symptoms_collection.add(
            documents=[item["text"] for item in symptoms_data],
            metadatas=[{"category": item["category"], "severity": item["severity"]} for item in symptoms_data],
            ids=[item["id"] for item in symptoms_data]
        )
        
        # Mock data for drug groups
        drug_groups_data = [
            {"id": str(uuid.uuid4()), "text": "Paracetamol - Thuốc giảm đau, hạ sốt. Liều dùng: 500mg x 3 lần/ngày", "group": "analgesic_antipyretic", "usage": "take_after_meals"},
            {"id": str(uuid.uuid4()), "text": "Amoxicillin - Kháng sinh nhóm penicillin. Liều dùng: 500mg x 3 lần/ngày", "group": "antibiotic", "usage": "take_1h_before_meals"},
            {"id": str(uuid.uuid4()), "text": "Omeprazole - Thuốc ức chế bơm proton. Liều dùng: 20mg x 1 lần/ngày", "group": "gastric", "usage": "take_morning_empty_stomach"},
            {"id": str(uuid.uuid4()), "text": "Metformin - Thuốc điều trị tiểu đường type 2. Liều dùng: 500mg x 2 lần/ngày", "group": "diabetes", "usage": "take_with_meals"},
            {"id": str(uuid.uuid4()), "text": "Amlodipine - Thuốc hạ huyết áp nhóm CCB. Liều dùng: 5mg x 1 lần/ngày", "group": "hypertension", "usage": "take_morning"},
            {"id": str(uuid.uuid4()), "text": "Cetirizine - Thuốc kháng histamin H1. Liều dùng: 10mg x 1 lần/ngày", "group": "allergy", "usage": "take_evening"},
            {"id": str(uuid.uuid4()), "text": "Ibuprofen - Thuốc chống viêm không steroid. Liều dùng: 400mg x 3 lần/ngày", "group": "anti_inflammatory", "usage": "take_after_meals"},
            {"id": str(uuid.uuid4()), "text": "Simvastatin - Thuốc hạ cholesterol nhóm statin. Liều dùng: 20mg x 1 lần/ngày", "group": "blood_lipid", "usage": "take_evening"}
        ]
        
        self.drug_groups_collection.add(
            documents=[item["text"] for item in drug_groups_data],
            metadatas=[{"group": item["group"], "usage": item["usage"]} for item in drug_groups_data],
            ids=[item["id"] for item in drug_groups_data]
        )
        
        # Mock data for lab results
        lab_results_data = [
            {"id": str(uuid.uuid4()), "text": "Glucose máu đói: 126 mg/dL (bình thường: 70-100). Chỉ số cao, nghi ngờ tiểu đường", "test_type": "blood_chemistry", "status": "high"},
            {"id": str(uuid.uuid4()), "text": "Cholesterol toàn phần: 240 mg/dL (bình thường: <200). Nguy cơ tim mạch", "test_type": "blood_lipid", "status": "high"},
            {"id": str(uuid.uuid4()), "text": "Hemoglobin: 9.5 g/dL (bình thường: 12-15). Thiếu máu nhẹ", "test_type": "complete_blood_count", "status": "low"},
            {"id": str(uuid.uuid4()), "text": "ALT: 65 U/L (bình thường: <40). Chức năng gan bất thường", "test_type": "liver_function", "status": "high"},
            {"id": str(uuid.uuid4()), "text": "Creatinine: 1.8 mg/dL (bình thường: 0.6-1.2). Chức năng thận giảm", "test_type": "kidney_function", "status": "high"},
            {"id": str(uuid.uuid4()), "text": "TSH: 8.5 mIU/L (bình thường: 0.4-4.0). Suy giáp", "test_type": "thyroid_hormone", "status": "high"},
            {"id": str(uuid.uuid4()), "text": "HbA1c: 8.2% (bình thường: <5.7%). Kiểm soát đường huyết kém", "test_type": "diabetes", "status": "high"},
            {"id": str(uuid.uuid4()), "text": "CRP: 15 mg/L (bình thường: <3). Viêm nhiễm cấp tính", "test_type": "inflammation", "status": "high"}
        ]
        
        self.lab_results_collection.add(
            documents=[item["text"] for item in lab_results_data],
            metadatas=[{"test_type": item["test_type"], "status": item["status"]} for item in lab_results_data],
            ids=[item["id"] for item in lab_results_data]
        )
    
    def search_symptoms(self, query, n_results=5):
        """Search for similar symptoms"""
        return self.symptoms_collection.query(
            query_texts=[query],
            n_results=n_results
        )
    
    def search_drug_groups(self, query, n_results=5):
        """Search for drug information"""
        return self.drug_groups_collection.query(
            query_texts=[query],
            n_results=n_results
        )
    
    def search_lab_results(self, query, n_results=5):
        """Search for similar lab results"""
        return self.lab_results_collection.query(
            query_texts=[query],
            n_results=n_results
        )

# Test function
if __name__ == "__main__":
    db = MedicalChromaDB()
    
    # Test search
    print("=== Test symptoms search ===")
    results = db.search_symptoms("đau đầu chóng mặt")
    print(results)
    
    print("\n=== Test drug search ===")
    results = db.search_drug_groups("thuốc giảm đau")
    print(results)
    
    print("\n=== Test lab results search ===")
    results = db.search_lab_results("đường huyết cao")
    print(results)