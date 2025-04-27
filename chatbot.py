# chatbot.py

from transformers import pipeline, StoppingCriteria, StoppingCriteriaList


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids, tokenizer):
        super().__init__()
        self.stop_token_ids = stop_token_ids
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        generated_ids = input_ids[0]
        new_token_id = generated_ids[-1]  # Only check the last generated token
        new_text = self.tokenizer.decode([new_token_id], skip_special_tokens=True)

        if "Doctor Question:" in new_text:
            return True
        return False


pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

# Store current case context
current_findings_text = None
current_retrieved_report = None

# Store conversation history
conversation_history = ""

def update_case(findings_text: str, retrieved_report: str):
    """Update the current case context and reset conversation history."""
    global current_findings_text, current_retrieved_report, conversation_history
    current_findings_text = findings_text
    current_retrieved_report = retrieved_report
    conversation_history = ""  # Reset history when new case uploaded

def build_initial_prompt():
    """Build initial case information."""
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
If the doctor asks you to change the findings or retrieved report, Do it for him based on his needs.

Always provide detailed, full-sentence answers.
If applicable, explain the reasoning based on the findings and retrieved report.
Do not limit yourself to only saying 'yes' or 'no'.
"""

def get_doctor_answer(question: str) -> str:
    """Answer doctor's question based on current case context."""
    global conversation_history

    prompt = build_initial_prompt()
    full_prompt = prompt + "\n\n" + conversation_history + f"\nDoctor Question: {question}\nAnswer:"

    tokenizer = pipe.tokenizer  # Get tokenizer
    stop_criteria = StoppingCriteriaList([StopOnTokens([], tokenizer)])

    result = pipe(
        full_prompt,
        max_new_tokens=300,
        temperature=0.7,  # Allow some creativity but still focused
        top_p=0.9,        # Nucleus sampling (take top 90% probability mass)
        stopping_criteria=stop_criteria
    )[0]['generated_text']
    # Clean up
    if "Answer:" in result:
        answer = result.split("Answer:")[-1].strip()
    else:
        answer = result.strip()

    if "Doctor Question:" in answer:
        answer = answer.split("Doctor Question:")[0].strip()

    # Update conversation history
    conversation_history += f"\nDoctor Question: {question}\nAnswer: {answer}\n"

    return answer
