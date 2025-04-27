# chatbot.py

from transformers import pipeline

# Load the model ONCE
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

# Store current findings and report globally
current_findings_text = None
current_retrieved_report = None

def update_case(findings_text: str, retrieved_report: str):
    """Update the current case context."""
    global current_findings_text, current_retrieved_report
    current_findings_text = findings_text
    current_retrieved_report = retrieved_report

def build_initial_prompt():
    """Build the prompt dynamically based on the latest uploaded case."""
    if current_findings_text is None or current_retrieved_report is None:
        return "No case has been uploaded yet."

    return f"""
You are an AI radiologist assistant.

The following information is available about the chest X-ray:

[Vision Transformer Findings]:
{current_findings_text}

[Retrieved Similar Report]:
{current_retrieved_report}

Based on this, you can now answer any questions the doctor might have.
Always base your answers only on the provided findings and retrieved report.
If you don't know, say 'I cannot answer this based on the provided information.'
"""

def get_doctor_answer(question: str) -> str:
    """Answer doctor's question based on current case context."""
    prompt = build_initial_prompt()
    full_prompt = prompt + f"\n\nDoctor Question: {question}\nAnswer:"
    result = pipe(full_prompt, max_length=500)[0]['generated_text']
    return result
