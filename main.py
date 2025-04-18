from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from models import generate_report
from orthanc_client import OrthancClient
from supabase_client import SupabaseClient  # New import
import pydicom
import io
from PIL import Image
import base64
import os
from typing import Optional, List, Dict
import traceback
import requests
from retrieval import get_similar_documents
import re
from sentence_transformers import SentenceTransformer
import torch
from requests.exceptions import JSONDecodeError
from pydantic import BaseModel
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, status, Security
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

# Models for authentication
class Token(BaseModel):
    access_token: str
    token_type: str
class LoginRequest(BaseModel):
    username: str
    password: str

class TokenData(BaseModel):
    doctor_id: Optional[str] = None

class Doctor(BaseModel):
    id: str
    email: str
    name: str
    specialty: Optional[str] = None

class DoctorInDB(Doctor):
    hashed_password: str

# Auth settings
SECRET_KEY = "YOUR_SECRET_KEY_HERE"  # In production, use a proper secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Create OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure connections
ORTHANC_URL = os.getenv("ORTHANC_URL", "http://34.159.48.206:8042")
ORTHANC_USERNAME = os.getenv("ORTHANC_USERNAME", "orthanc")
ORTHANC_PASSWORD = os.getenv("ORTHANC_PASSWORD", "orthanc")
SUPABASE_URL = os.getenv("SUPABASE_URL", "http://34.159.48.206:8000")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyAgCiAgICAicm9sZSI6ICJhbm9uIiwKICAgICJpc3MiOiAic3VwYWJhc2UtZGVtbyIsCiAgICAiaWF0IjogMTY0MTc2OTIwMCwKICAgICJleHAiOiAxNzk5NTM1NjAwCn0.dc_X5iR_VP_qT0zsiyj_I_OZ2T9FtRU2BBNWN8Bu4GE")

# Load model on startup
@app.on_event("startup")
def load_model():
    app.state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    app.state.gibberish_model = AutoModelForSequenceClassification.from_pretrained(
        "madhurjindal/autonlp-Gibberish-Detector-492513457", 
        token=False
    )
    app.state.gibberish_tokenizer = AutoTokenizer.from_pretrained(
        "madhurjindal/autonlp-Gibberish-Detector-492513457",
        token=False
    )

# Dependencies for clients
def get_orthanc_client():
    return OrthancClient(
        orthanc_url=ORTHANC_URL, 
        username=ORTHANC_USERNAME, 
        password=ORTHANC_PASSWORD
    )

def get_supabase_client():
    return SupabaseClient(
        supabase_url=SUPABASE_URL,
        supabase_key=SUPABASE_KEY
    )

# Authentication functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
        
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_doctor(token: str = Security(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        doctor_id: str = payload.get("sub")
        if doctor_id is None:
            raise credentials_exception
        token_data = TokenData(doctor_id=doctor_id)
    except JWTError:
        raise credentials_exception
        
    # Here you would typically fetch the doctor from your database
    # For now, we'll just return the ID
    return Doctor(id=token_data.doctor_id, email="", name="", specialty="")

# Token endpoint for login
@app.post("/token", response_model=Token)
async def login_for_access_token(
    login_data: LoginRequest,
    supabase_client: SupabaseClient = Depends(get_supabase_client)
):
    doctor = supabase_client.get_doctor_by_credentials(login_data.username, 
        login_data.password)
    if not doctor:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": doctor["id"]}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Orthanc Report Generator API with Doctor Authentication"}

# Doctor-specific endpoints
@app.get("/doctor/patients")
async def get_doctor_patients(
    current_doctor : Doctor=Depends(get_current_doctor),
    orthanc_client: OrthancClient = Depends(get_orthanc_client),
    supabase_client: SupabaseClient = Depends(get_supabase_client)
):
    """Get all patients for the logged-in doctor"""
    try:
        doctor_id = current_doctor.id
        
        # Get doctor's patients from mapping table
        doctor_patients = supabase_client.get_doctor_patients(doctor_id)
        
        # Extract patient DICOM IDs
        patient_dicom_ids = [p["patient_dicom_id"] for p in doctor_patients]
        
        # Get all patients from Orthanc
        all_orthanc_patients = orthanc_client.get_patients()
        
        # Filter patients that belong to this doctor
        doctor_orthanc_patients = []
        for patient_id in all_orthanc_patients:
            try:
                patient_data = orthanc_client.get_patient_info(patient_id)
                tags = patient_data.get('MainDicomTags', {})
                patient_dicom_id = tags.get("PatientID", "")
                
                # Check if this patient belongs to the doctor
                if patient_dicom_id in patient_dicom_ids:
                    # Get additional info from Supabase if available
                    supabase_info = supabase_client.get_patient_basic_info(patient_dicom_id)
                    
                    doctor_orthanc_patients.append({
                        "id": patient_id,
                        "dicom_id": patient_dicom_id,
                        "name": tags.get("PatientName", "Unknown"),
                        "gender": tags.get("PatientSex", "U"),
                        "birth_date": tags.get("PatientBirthDate", ""),
                        "study_count": len(patient_data.get('Studies', [])),
                        "last_update": patient_data.get('LastUpdate', ""),
                        "patient_age": tags.get("PatientAge", ""),
                        # Additional fields from Supabase if available
                        "contact_info": supabase_info.get("contact_info") if supabase_info else None,
                        "insurance": supabase_info.get("insurance") if supabase_info else None,
                        "additional_notes": supabase_info.get("additional_notes") if supabase_info else None
                    })
            except Exception as e:
                print(f"Error processing patient {patient_id}: {str(e)}")
        
        return {"patients": doctor_orthanc_patients}
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to map patient to doctor
@app.post("/admin/assign-patient")
async def assign_patient_to_doctor(
    doctor_id: str,
    patient_dicom_id: str,
    supabase_client: SupabaseClient = Depends(get_supabase_client)
):
    """Assign a patient to a doctor (admin endpoint)"""
    if supabase_client.assign_patient_to_doctor(doctor_id, patient_dicom_id):
        return {"status": "success", "message": "Patient assigned to doctor"}
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to assign patient to doctor"
        )

# Add doctor-specific version of existing endpoints
@app.get("/doctor/patient/{patient_id}/studies")
async def get_doctor_patient_studies(
    patient_id: str,
    current_doctor :Doctor= Depends(get_current_doctor),
    orthanc_client: OrthancClient = Depends(get_orthanc_client),
    supabase_client: SupabaseClient = Depends(get_supabase_client)
):
    """Get studies for a specific patient (with doctor verification)"""
    try:
        # First verify this patient belongs to the doctor
        patient_data = orthanc_client.get_patient_info(patient_id)
        tags = patient_data.get('MainDicomTags', {})
        patient_dicom_id = tags.get("PatientID", "")
        
        # Get doctor's patients
        doctor_id = current_doctor.id
        doctor_patients = supabase_client.get_doctor_patients(doctor_id)
        doctor_patient_ids = [p["patient_dicom_id"] for p in doctor_patients]
        
        # Check if this patient belongs to the doctor
        if patient_dicom_id not in doctor_patient_ids:
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to access this patient's information"
            )
        
        # If authorized, get the studies
        studies = orthanc_client.get_patient_studies(patient_id)
        return {"studies": studies}
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orthanc/test-connection")
def test_connection(orthanc_client: OrthancClient = Depends(get_orthanc_client)):
    """Test connection to Orthanc server"""
    try:
        # Just attempt to make a simple request to check connectivity
        response = requests.get(
            f"{ORTHANC_URL}/system", 
            auth=(ORTHANC_USERNAME, ORTHANC_PASSWORD)
        )
        response.raise_for_status()
        return {
            "status": "connected", 
            "orthanc_version": response.json().get("Version", "Unknown")
        }
    except Exception as e:
        traceback.print_exc()  # Print traceback for debugging
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to connect to Orthanc server: {str(e)}"
        )

# Enhanced patient list endpoint
@app.get("/orthanc/patients")
def list_patients(
    orthanc_client: OrthancClient = Depends(get_orthanc_client)
):
    """List all patients with detailed information"""
    try:
        patient_ids = orthanc_client.get_patients()
        
        detailed_patients = []
        for pid in patient_ids:
            try:
                patient_data = orthanc_client.get_patient_info(pid)
                tags = patient_data.get('MainDicomTags', {})
                detailed_patients.append({
                    "id": pid,
                    "name": tags.get("PatientName", "Unknown"),
                    "gender": tags.get("PatientSex", "U"),
                    "birth_date": tags.get("PatientBirthDate", ""),
                    "study_count": len(patient_data.get('Studies', [])),
                    "last_update": patient_data.get('LastUpdate', ""),
                    "patient_age": tags.get("PatientAge", ""),
                    "patient_id_dicom": tags.get("PatientID", "")
                })
            except Exception as e:
                print(f"Error fetching details for patient {pid}: {str(e)}")
        
        return {"patients": detailed_patients}
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orthanc/patients/{patient_id}/studies")
def list_patient_studies(
    patient_id: str, 
    orthanc_client: OrthancClient = Depends(get_orthanc_client)
):
    """List all studies for a patient"""
    try:
        studies = orthanc_client.get_patient_studies(patient_id)
        return {"studies": studies}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/orthanc/series/{series_id}")
def get_series_info(
    series_id: str,
    orthanc_client: OrthancClient = Depends(get_orthanc_client)
):
    """Get detailed information for a series"""
    try:
        series_info = orthanc_client.get_series_info(series_id)
        return series_info
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/orthanc/studies/{study_id}/series")
def list_study_series(
    study_id: str, 
    orthanc_client: OrthancClient = Depends(get_orthanc_client)
):
    """List all series in a study"""
    try:
        series = orthanc_client.get_study_series(study_id)
        return {"series": series}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orthanc/series/{series_id}/instances")
def list_series_instances(
    series_id: str, 
    orthanc_client: OrthancClient = Depends(get_orthanc_client)
):
    """List all instances in a series"""
    try:
        instances = orthanc_client.get_series_instances(series_id)
        return {"instances": instances}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orthanc/generate-report")
def generate_orthanc_report(
    instance_id: str,
    orthanc_client: OrthancClient = Depends(get_orthanc_client)
):
    """Generate report from an instance in Orthanc"""
    try:
        # Get DICOM file from Orthanc
        dicom_content = orthanc_client.get_instance_dicom(instance_id)
        
        # Get instance metadata for additional information
        try:
            instance_tags = orthanc_client.get_instance_tags(instance_id)
        except Exception as tag_error:
            # If we can't get tags, continue without them
            print(f"Error getting instance tags: {tag_error}")
            instance_tags = {}
        
        # Process the DICOM file and generate report
        result = process_dicom(dicom_content)
        
        # Add instance metadata to the response
        result["instance_metadata"] = {
            "id": instance_id,
            "patient_name": instance_tags.get("PatientName", ""),
            "patient_id": instance_tags.get("PatientID", ""),
            "study_description": instance_tags.get("StudyDescription", ""),
            "series_description": instance_tags.get("SeriesDescription", ""),
            "age": instance_tags.get("PatientAge", ""),
            "sex": instance_tags.get("PatientSex", ""),
            "patient_birth_date": instance_tags.get("PatientBirthDate", ""),
            "patientweight": instance_tags.get("PatientWeight", ""),
        }
        
        return result
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orthanc/view-dicom")
def view_dicom_image(
    instance_id: str,
    orthanc_client: OrthancClient = Depends(get_orthanc_client)
):
    """Retrieve and view a DICOM image without generating a report"""
    try:
        # Get DICOM file from Orthanc
        dicom_content = orthanc_client.get_instance_dicom(instance_id)
        
        # Get instance metadata for additional information
        try:
            instance_tags = orthanc_client.get_instance_tags(instance_id)
        except Exception as tag_error:
            print(f"Error getting instance tags: {tag_error}")
            instance_tags = {}
        
        # Process the DICOM file to extract the image
        try:
            # Parse DICOM data
            dicom_dataset = pydicom.dcmread(io.BytesIO(dicom_content))
            
            # Extract pixel data and convert to image
            if hasattr(dicom_dataset, 'pixel_array'):
                image_array = dicom_dataset.pixel_array
                image = Image.fromarray(image_array)
                
                # Convert to RGB and prepare image
                image = image.convert("RGB")
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                
                return {
                    "image_preview": f"data:image/png;base64,{image_base64}",
                    "instance_metadata": {
                        "id": instance_id,
                        "patient_name": instance_tags.get("PatientName", ""),
                        "patient_id": instance_tags.get("PatientID", ""),
                        "study_description": instance_tags.get("StudyDescription", ""),
                        "series_description": instance_tags.get("SeriesDescription", "")
                    }
                }
            else:
                raise HTTPException(status_code=400, detail="DICOM file has no pixel data")
            
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error processing DICOM: {str(e)}")
            
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reports")
def save_generated_report(report_data: dict):
    """Save a report after gibberish validation"""
    try:
        report_content = report_data.get("content")
        if not report_content:
            raise HTTPException(status_code=400, detail="Report content required")

        # --- Gibberish Detection ---
        inputs = app.state.gibberish_tokenizer(
            report_content, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512  
        )
        
        with torch.no_grad():
            outputs = app.state.gibberish_model(**inputs)
        
        probs = F.softmax(outputs.logits, dim=-1)
        predicted_index = torch.argmax(probs, dim=1).item()
        predicted_label = app.state.gibberish_model.config.id2label[predicted_index]
        
        # Check if the text is classified as gibberish (assuming label 0 is "not gibberish")
        if probs[0][0] < 0.5:  # If "not gibberish" probability < 50%
            raise HTTPException(
                status_code=400,
                detail=f"Report contains low-quality text (classified as: {predicted_label}, confidence: {probs[0][predicted_index]:.2f})"
            )
        
        # --- Save to Supabase (original logic without similarity check) ---
        embedding = app.state.embedding_model.encode(report_content)
        embedding_tensor = torch.tensor(embedding).float()
        normalized_embedding = embedding_tensor / torch.norm(embedding_tensor, p=2)

        data = {
            "content": report_content,
            "embedding": normalized_embedding.detach().cpu().tolist()
        }
        
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/db1",
            headers=headers,
            json=data
        )
        response.raise_for_status()

        result_id = None
        if response.content:
            try:
                result = response.json()
                result_id = result.get("id") if isinstance(result, dict) else None
            except JSONDecodeError:
                pass

        return {"status": "success", "message": "Report saved successfully"}

    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
def process_dicom(dicom_content: bytes):
    """Common function to process DICOM data and generate report with RAG"""
    try:
        # Validate DICOM format
        dicom_dataset = pydicom.dcmread(io.BytesIO(dicom_content))
        
        # Extract pixel data and convert to image
        if hasattr(dicom_dataset, 'pixel_array'):
            image_array = dicom_dataset.pixel_array
            image = Image.fromarray(image_array)
            
            # Convert to RGB and prepare image
            image = image.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        else:
            raise HTTPException(status_code=400, detail="DICOM file has no pixel data")

        # Generate the findings and impression
        report = generate_report(image_bytes)
        report_text = f"{report['findings']} {report['impression']}"

        # Retrieve similar documents using RAG
        try:
            retrieved_docs = get_similar_documents(
                query=report_text,
                model=app.state.embedding_model,
                supabase_url=SUPABASE_URL,
                supabase_key=SUPABASE_KEY,
                top_k=1
            )
        except Exception as e:
            retrieved_docs = []
            print(f"Document retrieval failed: {str(e)}")

        # Format retrieved reports
        formatted_retrieved = []
        for doc in retrieved_docs:
            content = doc.get("content", "")
            
            # Split into sections and clean whitespace
            sections = re.split(
                r'\nIMPRESSION:|IMPRESSION:|Impression:', 
                content, 
                flags=re.IGNORECASE
            )
            
            # Clean and format text
            findings = re.sub(r'\s+', ' ', sections[0].strip()) if len(sections) > 0 else ""
            impression = re.sub(r'\s+', ' ', sections[1].strip()) if len(sections) > 1 else ""

            formatted_retrieved.append({
                "id": doc.get("id"),
                "similarity": round(doc.get("similarity", 0), 2),
                "report": {
                    "findings": findings,
                    "impression": impression
                }
            })

        return {
            "generated_report": {
                "findings": report["findings"],
                "impression": report["impression"]
            },
            # "image_preview": f"data:image/png;base64,{image_base64}",
            "retrieved_reports": formatted_retrieved
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
