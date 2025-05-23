import requests
from typing import List, Dict, Optional
import traceback

class SupabaseClient:
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase_url = supabase_url.rstrip('/')
        self.supabase_key = supabase_key
        self.headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
        }
    def get_doctor_by_credentials(self, email: str, password: str) -> Optional[Dict]:
        try:
            response = requests.get(
                f"{self.supabase_url}/rest/v1/doctors",
                headers=self.headers,
                params={"email": f"eq.{email}"},
            )
            response.raise_for_status()
            doctors = response.json()
            
            if doctors and len(doctors) > 0:
                # Check plain text password match
                if doctors[0]["password"] == password:
                    return doctors[0]
            return None
        except Exception as e:
            traceback.print_exc()
            return None
    
    def get_doctor_patients(self, doctor_id: str) -> List[Dict]:
        """Get all patients assigned to a specific doctor"""
        try:
            response = requests.get(
                f"{self.supabase_url}/rest/v1/patient_doctor_mapping",
                headers=self.headers,
                params={"doctor_id": f"eq.{doctor_id}"},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            traceback.print_exc()
            return []
    
    def get_patient_basic_info(self, patient_dicom_id: str) -> Optional[Dict]:
        """Get patient basic information by DICOM ID"""
        try:
            response = requests.get(
                f"{self.supabase_url}/rest/v1/patients",
                headers=self.headers,
                params={"dicom_id": f"eq.{patient_dicom_id}"},
            )
            response.raise_for_status()
            patients = response.json()
            
            if patients and len(patients) > 0:
                return patients[0]
            return None
        except Exception as e:
            traceback.print_exc()
            return None
    
    def assign_patient_to_doctor(self, doctor_id: str, patient_dicom_id: str) -> tuple[bool, str]:
        """Associate a patient with a doctor"""
        try:
            # First, check if patient is already assigned to any doctor
            response = requests.get(
                f"{self.supabase_url}/rest/v1/patient_doctor_mapping",
                headers=self.headers,
                params={"patient_dicom_id": f"eq.{patient_dicom_id}"}
            )
            response.raise_for_status()
            existing_assignments = response.json()
            
            # If patient is already assigned to a doctor, return False with message
            if existing_assignments and len(existing_assignments) > 0:
                return False, "Patient is already assigned to a doctor"
            
            # If not assigned, proceed with assignment
            data = {
                "doctor_id": doctor_id,
                "patient_dicom_id": patient_dicom_id
            }
            
            response = requests.post(
                f"{self.supabase_url}/rest/v1/patient_doctor_mapping",
                headers=self.headers,
                json=data
            )
            response.raise_for_status()
            return True, "Patient successfully assigned to doctor"
        except Exception as e:
            traceback.print_exc()
            return False, f"Error: {str(e)}"

    def get_all_doctors(self) -> List[Dict]:
        """Get list of all doctors"""
        try:
            response = requests.get(
                f"{self.supabase_url}/rest/v1/doctors",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            traceback.print_exc()
            return []
    def get_doctor_by_id(self, doctor_id: str) -> Optional[Dict]:
        """Get a doctor by ID"""
        try:
            response = requests.get(
                f"{self.supabase_url}/rest/v1/doctors",
                headers=self.headers,
                params={"id": f"eq.{doctor_id}"}
            )
            response.raise_for_status()
            doctors = response.json()

            if doctors and len(doctors) > 0:
                return doctors[0]
            return None
        except Exception as e:
            traceback.print_exc()
            return None

    def get_admin_by_credentials(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate an admin by username and password"""
        try:
            response = requests.get(
                f"{self.supabase_url}/rest/v1/admins",
                headers=self.headers,
                params={"username": f"eq.{username}"},
            )
            response.raise_for_status()
            admins = response.json()
            
            if admins and len(admins) > 0:
                # Check plain text password match
                if admins[0]["password"] == password:
                    return admins[0]
            return None
        except Exception as e:
            traceback.print_exc()
            return None

