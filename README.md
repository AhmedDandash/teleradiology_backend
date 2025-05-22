# Ai-Assisted-Teleradiology-Platform

A comprehensive FastAPI-based medical imaging system that integrates with Orthanc DICOM server, generates automated radiology reports using Vision Transformer models, and provides an AI-powered chatbot for medical consultations.

## 🏥 Features

- **DICOM Processing**: Upload, view, and process DICOM medical images
- **Automated Report Generation**: AI-powered chest X-ray analysis using Vision Transformer models
- **RAG-Enhanced Reporting**: Retrieval-Augmented Generation for contextual report insights
- **Doctor Authentication**: JWT-based authentication system for medical professionals
- **AI Medical Chatbot**: Interactive assistant for medical queries based on current cases
- **Vector Search**: Semantic similarity search for historical reports
- **Quality Control**: Gibberish detection to ensure report quality
- **Multi-Role Access**: Support for doctors and administrators

## 🚀 Tech Stack

- **Backend**: FastAPI, Python 3.8+
- **AI Models**: 
  - Vision Transformer (ViT) for medical image analysis
  - BERT for natural language processing
  - Sentence Transformers for embeddings
- **Database**: Supabase (PostgreSQL with vector extensions)
- **DICOM Server**: Orthanc
- **Authentication**: JWT tokens
- **Image Processing**: PyDICOM, PIL
- **Vector Search**: PostgreSQL pgvector

## 📋 Prerequisites

- Python 3.8 or higher
- Orthanc DICOM server
- Supabase account and database
- OpenRouter API key (for chatbot functionality)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd medical-imaging-api
   ```
   
2. **Set up environment variables**
   ```bash
   export ORTHANC_URL="http://your-orthanc-server:8042"
   export ORTHANC_USERNAME="your-username"
   export ORTHANC_PASSWORD="your-password"
   export SUPABASE_URL="your-supabase-url"
   export SUPABASE_ANON_KEY="your-supabase-key"
   export OPENROUTER_API_KEY="your-openrouter-key"
   ```

## 🗃️ Database Setup

### Required Supabase Tables

1. **doctors**
   ```sql
   CREATE TABLE doctors (
     id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
     email TEXT UNIQUE NOT NULL,
     name TEXT NOT NULL,
     password TEXT NOT NULL,
     specialty TEXT
   );
   ```

2. **patients**
   ```sql
   CREATE TABLE patients (
     id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
     dicom_id TEXT UNIQUE NOT NULL,
     contact_info TEXT,
     insurance TEXT,
     additional_notes TEXT
   );
   ```

3. **patient_doctor_mapping**
   ```sql
   CREATE TABLE patient_doctor_mapping (
     id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
     doctor_id UUID REFERENCES doctors(id),
     patient_dicom_id TEXT NOT NULL
   );
   ```

4. **db1** (Reports with vector embeddings)
   ```sql
   CREATE TABLE db1 (
     id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
     content TEXT NOT NULL,
     embedding vector(384)
   );
   ```

5. **admins**
   ```sql
   CREATE TABLE admins (
     id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
     username TEXT UNIQUE NOT NULL,
     password TEXT NOT NULL
   );
   ```

### Vector Search Function
```sql
CREATE OR REPLACE FUNCTION match_db1(
  query_embedding vector(384),
  match_count int DEFAULT 5
)
RETURNS TABLE (
  id UUID,
  content TEXT,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    db1.id,
    db1.content,
    1 - (db1.embedding <=> query_embedding) AS similarity
  FROM db1
  ORDER BY db1.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
```

## 🚀 Running the Application

1. **Start the FastAPI server**
   ```bash
   python main.py
   ```
   or
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Access the API documentation**
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

## 📱 API Endpoints

### Authentication
- `POST /token` - Doctor login
- `POST /admin/login` - Admin login

### Doctor Endpoints
- `GET /doctor` - Get doctor information
- `GET /doctor/patients` - Get assigned patients
- `GET /doctor/patient/{patient_id}/studies` - Get patient studies
- `GET /doctor/info-and-patients` - Get doctor info and patients

### Medical Imaging
- `POST /upload-dicom/` - Upload DICOM files
- `GET /orthanc/patients` - List all patients
- `GET /orthanc/patients/{patient_id}/studies` - Get patient studies
- `GET /orthanc/generate-report` - Generate AI report
- `GET /orthanc/generate-report-and-update` - Generate report and update chatbot
- `GET /orthanc/view-dicom` - View DICOM image

### AI Chatbot
- `POST /ask` - Ask medical questions about current case

### Reports
- `POST /reports` - Save generated reports with quality validation

### Admin Endpoints
- `GET /admin/doctors` - Get all doctors
- `POST /admin/assign-patient` - Assign patient to doctor

## 🤖 AI Models Used

1. **Chest X-ray Analysis**
   - `IAMJB/chexpert-mimic-cxr-findings-baseline`
   - `IAMJB/chexpert-mimic-cxr-impression-baseline`

2. **Text Quality Control**
   - `madhurjindal/autonlp-Gibberish-Detector-492513457`

3. **Embeddings**
   - `all-MiniLM-L6-v2` for semantic search

4. **Chatbot**
   - `deepseek/deepseek-r1:free` via OpenRouter

## 🔒 Security Features

- JWT-based authentication with configurable expiration
- Role-based access control (Doctor/Admin)
- Patient-doctor assignment verification
- Secure password handling
- CORS middleware configuration

## 📊 Report Generation Workflow

1. **DICOM Upload**: Medical images uploaded to Orthanc server
2. **Image Processing**: DICOM files converted to processable format
3. **AI Analysis**: Vision Transformer models generate findings and impressions
4. **RAG Enhancement**: Similar historical reports retrieved for context
5. **Quality Control**: Generated text validated for coherence
6. **Storage**: Reports stored with vector embeddings for future retrieval

## 🤖 Chatbot Integration

The system includes an intelligent medical chatbot that:
- Maintains context of the current case
- Answers questions based on generated reports
- Uses retrieved similar cases for enhanced responses
- Provides medically relevant information

## 🧪 Testing

Test the API endpoints using the interactive documentation at `/docs` or with tools like Postman.

Example authentication:
```bash
curl -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "doctor@example.com", "password": "password"}'
```

## 📄 File Structure

```
├── main.py                 # FastAPI application and routes
├── models.py              # AI model loading and report generation
├── newchatbot.py          # Medical chatbot functionality
├── orthanc_client.py      # Orthanc DICOM server client
├── supabase_client.py     # Supabase database client
├── retrieval.py           # Vector search and RAG functionality

```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for research and development purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## 📞 Support

For support, please open an issue in the GitHub repository or contact the development team.

---

**Made with ❤️ for advancing medical imaging technology**
