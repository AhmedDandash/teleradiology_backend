# chatbot.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import gc

# Store current findings and report globally
current_findings_text = None
current_retrieved_report = None

# Use a smaller but reliable model that works well on CPU
DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Fallback to original if needed
MEDICAL_MODEL = "TheBloke/medalpaca-13B-GGUF"  # Alternative medical model

# Global model variables - only initialize once
pipe = None
tokenizer = None
model = None

def initialize_model():
    """Initialize model with proper settings for CPU."""
    global pipe, tokenizer, model
    
    try:
        # Clean up any existing models
        if model is not None:
            del model
            del tokenizer
            if pipe is not None:
                del pipe
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        # Simple pipeline approach - no device_map
        pipe = pipeline(
            "text-generation",
            model=DEFAULT_MODEL,
            # No device_map parameter
        )
        
        print(f"Model {DEFAULT_MODEL} loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # Try even simpler fallback if initial load fails
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            print("Attempting fallback model loading...")
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
            model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL)
            
            pipe = pipeline(
                "text-generation", 
                model=model, 
                tokenizer=tokenizer
            )
            print("Fallback model loaded successfully")
            return True
        except Exception as fallback_error:
            print(f"Fallback model loading failed: {str(fallback_error)}")
            return False

def update_case(findings_text: str, retrieved_report: str):
    """Update the current case context."""
    global current_findings_text, current_retrieved_report
    current_findings_text = findings_text
    current_retrieved_report = retrieved_report
    print("Case updated successfully")

def build_initial_prompt():
    """Build the prompt dynamically based on the latest uploaded case."""
    if current_findings_text is None or current_retrieved_report is None:
        return "No case has been uploaded yet."

    # Format the retrieved report text
    retrieved_text = ""
    if isinstance(current_retrieved_report, list):
        for i, report in enumerate(current_retrieved_report):
            if isinstance(report, dict) and "report" in report:
                findings = report["report"].get("findings", "")
                impression = report["report"].get("impression", "")
                retrieved_text += f"Similar Report {i+1}:\n- Findings: {findings}\n- Impression: {impression}\n\n"
    else:
        retrieved_text = str(current_retrieved_report)

    return f"""You are an AI radiologist assistant.

The following information is available about the chest X-ray:

[Vision Transformer Findings]:
{current_findings_text}

[Retrieved Similar Report]:
{retrieved_text}

Based on this, you can now answer any questions the doctor might have.
Always base your answers only on the provided findings and retrieved report.
If you don't know, say 'I cannot answer this based on the provided information.'
"""

def get_doctor_answer(question: str) -> str:
    """Answer doctor's question based on current case context."""
    global pipe
    
    # Lazy initialization of the model
    if pipe is None:
        success = initialize_model()
        if not success:
            return "Error: Could not initialize the language model. Please check system logs."
    
    try:
        # Build prompt with context and question
        prompt = build_initial_prompt()
        full_prompt = prompt + f"\n\nDoctor Question: {question}\nAnswer:"
        
        # Generate response
        result = pipe(
            full_prompt, 
            max_length=500,
            temperature=0.7,
            num_return_sequences=1,
            do_sample=True
        )
        
        # Extract answer from full generated text
        generated_text = result[0]['generated_text']
        
        # Try to isolate just the answer part
        if "Answer:" in generated_text:
            parts = generated_text.split("Answer:")
            if len(parts) > 1:
                answer = parts[-1].strip()
            else:
                answer = generated_text
        else:
            # If no "Answer:" marker, just return the whole thing
            answer = generated_text
        
        return answer
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error generating response: {str(e)}"