# orthanc_client.py
import requests
import io

class OrthancClient:
    def __init__(self, orthanc_url="http://localhost:8042", username=None, password=None):
        # Remove trailing slash if present to avoid double slashes in URLs
        self.orthanc_url = orthanc_url.rstrip('/')
        self.auth = None
        if username and password:
            self.auth = (username, password)
    
    def get_patients(self):
        """Get list of all patients"""
        response = requests.get(
            f"{self.orthanc_url}/patients", 
            auth=self.auth
        )
        response.raise_for_status()  # This will raise an exception for 4XX/5XX responses
        return response.json()
    
    def get_patient_studies(self, patient_id):
        """Get list of studies for a patient"""
        response = requests.get(
            f"{self.orthanc_url}/patients/{patient_id}/studies", 
            auth=self.auth
        )
        response.raise_for_status()
        return response.json()
    
    def get_study_series(self, study_id):
        """Get list of series in a study"""
        response = requests.get(
            f"{self.orthanc_url}/studies/{study_id}/series", 
            auth=self.auth
        )
        response.raise_for_status()
        return response.json()
    
    def get_series_instances(self, series_id):
        """Get list of instances (DICOM files) in a series"""
        response = requests.get(
            f"{self.orthanc_url}/series/{series_id}/instances", 
            auth=self.auth
        )
        response.raise_for_status()
        return response.json()
    
    def get_instance_dicom(self, instance_id):
        """Get DICOM file for an instance"""
        response = requests.get(
            f"{self.orthanc_url}/instances/{instance_id}/file", 
            auth=self.auth
        )
        response.raise_for_status()
        return response.content
    
    def get_instance_tags(self, instance_id):
        """Get DICOM tags for an instance"""
        response = requests.get(
            f"{self.orthanc_url}/instances/{instance_id}/simplified-tags", 
            auth=self.auth
        )
        response.raise_for_status()
        return response.json()
    def get_patient_info(self, patient_id: str) -> dict:
        """Get detailed patient information"""
        response = requests.get(
            f"{self.orthanc_url}/patients/{patient_id}", 
            auth=self.auth
        )
        response.raise_for_status()
        return response.json()
    
    def get_series_info(self, series_id: str) -> dict:
        """Get detailed series information"""
        response = requests.get(
            f"{self.orthanc_url}/series/{series_id}", 
            auth=self.auth
        )
        response.raise_for_status()
        return response.json()
    def upload_dicom(self, dicom_data):
        """Upload a DICOM file to Orthanc"""
        response = requests.post(
            f"{self.orthanc_url}/instances",
            # auth=(self.username, self.password),
            data=dicom_data
        )
        response.raise_for_status()
        return response.json()
   
