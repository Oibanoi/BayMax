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
            {"id": str(uuid.uuid4()), "text": "Persistent headache, dizziness, nausea", "category": "neurological", "severity": "moderate"},
            {"id": str(uuid.uuid4()), "text": "Dry cough, shortness of breath, chest pain", "category": "respiratory", "severity": "severe"},
            {"id": str(uuid.uuid4()), "text": "Abdominal pain, diarrhea, nausea", "category": "gastrointestinal", "severity": "mild"},
            {"id": str(uuid.uuid4()), "text": "High fever, chills, muscle pain", "category": "infectious", "severity": "severe"},
            {"id": str(uuid.uuid4()), "text": "Fatigue, loss of appetite, weight loss", "category": "systemic", "severity": "moderate"},
            {"id": str(uuid.uuid4()), "text": "Joint pain, joint swelling, morning stiffness", "category": "musculoskeletal", "severity": "moderate"},
            {"id": str(uuid.uuid4()), "text": "Left chest pain, palpitations, shortness of breath", "category": "cardiovascular", "severity": "severe"},
            {"id": str(uuid.uuid4()), "text": "Red rash, itching, swelling", "category": "dermatological", "severity": "mild"}
        ]
        
        self.symptoms_collection.add(
            documents=[item["text"] for item in symptoms_data],
            metadatas=[{"category": item["category"], "severity": item["severity"]} for item in symptoms_data],
            ids=[item["id"] for item in symptoms_data]
        )
        
        # Mock data for drug groups
        drug_groups_data = [
            {"id": str(uuid.uuid4()), "text": "Paracetamol - Pain reliever, fever reducer. Dosage: 500mg x 3 times/day", "group": "analgesic_antipyretic", "usage": "take_after_meals"},
            {"id": str(uuid.uuid4()), "text": "Amoxicillin - Penicillin group antibiotic. Dosage: 500mg x 3 times/day", "group": "antibiotic", "usage": "take_1h_before_meals"},
            {"id": str(uuid.uuid4()), "text": "Omeprazole - Proton pump inhibitor. Dosage: 20mg x 1 time/day", "group": "gastric", "usage": "take_morning_empty_stomach"},
            {"id": str(uuid.uuid4()), "text": "Metformin - Type 2 diabetes treatment. Dosage: 500mg x 2 times/day", "group": "diabetes", "usage": "take_with_meals"},
            {"id": str(uuid.uuid4()), "text": "Amlodipine - CCB group antihypertensive. Dosage: 5mg x 1 time/day", "group": "hypertension", "usage": "take_morning"},
            {"id": str(uuid.uuid4()), "text": "Cetirizine - H1 antihistamine. Dosage: 10mg x 1 time/day", "group": "allergy", "usage": "take_evening"},
            {"id": str(uuid.uuid4()), "text": "Ibuprofen - Non-steroidal anti-inflammatory. Dosage: 400mg x 3 times/day", "group": "anti_inflammatory", "usage": "take_after_meals"},
            {"id": str(uuid.uuid4()), "text": "Simvastatin - Statin group cholesterol lowering. Dosage: 20mg x 1 time/day", "group": "blood_lipid", "usage": "take_evening"}
        ]
        
        self.drug_groups_collection.add(
            documents=[item["text"] for item in drug_groups_data],
            metadatas=[{"group": item["group"], "usage": item["usage"]} for item in drug_groups_data],
            ids=[item["id"] for item in drug_groups_data]
        )
        
        # Mock data for lab results
        lab_results_data = [
            {"id": str(uuid.uuid4()), "text": "Fasting glucose: 126 mg/dL (normal: 70-100). High level, suspected diabetes", "test_type": "blood_chemistry", "status": "high"},
            {"id": str(uuid.uuid4()), "text": "Total cholesterol: 240 mg/dL (normal: <200). Cardiovascular risk", "test_type": "blood_lipid", "status": "high"},
            {"id": str(uuid.uuid4()), "text": "Hemoglobin: 9.5 g/dL (normal: 12-15). Mild anemia", "test_type": "complete_blood_count", "status": "low"},
            {"id": str(uuid.uuid4()), "text": "ALT: 65 U/L (normal: <40). Abnormal liver function", "test_type": "liver_function", "status": "high"},
            {"id": str(uuid.uuid4()), "text": "Creatinine: 1.8 mg/dL (normal: 0.6-1.2). Decreased kidney function", "test_type": "kidney_function", "status": "high"},
            {"id": str(uuid.uuid4()), "text": "TSH: 8.5 mIU/L (normal: 0.4-4.0). Hypothyroidism", "test_type": "thyroid_hormone", "status": "high"},
            {"id": str(uuid.uuid4()), "text": "HbA1c: 8.2% (normal: <5.7%). Poor glycemic control", "test_type": "diabetes", "status": "high"},
            {"id": str(uuid.uuid4()), "text": "CRP: 15 mg/L (normal: <3). Acute inflammation", "test_type": "inflammation", "status": "high"}
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
    results = db.search_symptoms("headache dizziness")
    print(results)
    
    print("\n=== Test drug search ===")
    results = db.search_drug_groups("pain reliever")
    print(results)
    
    print("\n=== Test lab results search ===")
    results = db.search_lab_results("high glucose")
    print(results)