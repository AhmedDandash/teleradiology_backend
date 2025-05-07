from openai import OpenAI
import os

# Store current findings and report globally
current_findings_text = None
current_retrieved_report = None

# Initialize OpenRouter client
client = None

def initialize_model():
    """Initialize the OpenRouter client."""
    global client
    
    try:
        # Use the provided API key
        api_key = os.getenv("OPENROUTER_API_KEY", "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")
        
        # Initialize OpenAI client with OpenRouter base URL
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        print("OpenRouter client initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing OpenRouter client: {str(e)}")
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
    global client
    
    # Lazy initialization of the client
    if client is None:
        success = initialize_model()
        if not success:
            return "Error: Could not initialize the language model. Please check system logs."
    
    try:
        # Build prompt with context and question
        system_prompt = build_initial_prompt()
        user_prompt = question
        
        # Call the OpenRouter API
        response = client.chat.completions.create(
            model="deepseek/deepseek-r1:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        # Extract the answer
        answer = response.choices[0].message.content
        return answer
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error generating response: {str(e)}"