from transformers import VisionEncoderDecoderModel, ViTImageProcessor, BertTokenizer
from PIL import Image
import io
import torch

# Load both models
findings_model_name = "IAMJB/chexpert-mimic-cxr-findings-baseline"
impression_model_name = "IAMJB/chexpert-mimic-cxr-impression-baseline"

findings_model = VisionEncoderDecoderModel.from_pretrained(findings_model_name).eval()
impression_model = VisionEncoderDecoderModel.from_pretrained(impression_model_name).eval()

findings_processor = ViTImageProcessor.from_pretrained(findings_model_name)
impression_processor = ViTImageProcessor.from_pretrained(impression_model_name)

findings_tokenizer = BertTokenizer.from_pretrained(findings_model_name)
impression_tokenizer = BertTokenizer.from_pretrained(impression_model_name)

generation_args = {
    "num_return_sequences": 1,
    "max_length": 128,
    "use_cache": True,
    "decoder_start_token_id": findings_tokenizer.cls_token_id,  # same for both
}

def generate_report(image_data: bytes) -> dict:
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Generate Findings
    findings_inputs = findings_processor(images=image, return_tensors="pt").pixel_values
    with torch.no_grad():
        findings_ids = findings_model.generate(findings_inputs, **generation_args)
    findings_text = findings_tokenizer.decode(findings_ids[0], skip_special_tokens=True)

    # Generate Impression
    impression_inputs = impression_processor(images=image, return_tensors="pt").pixel_values
    with torch.no_grad():
        impression_ids = impression_model.generate(impression_inputs, **generation_args)
    impression_text = impression_tokenizer.decode(impression_ids[0], skip_special_tokens=True)

    return {
        "findings": findings_text,
        "impression": impression_text
    }
