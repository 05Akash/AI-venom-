from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load the fine-tuned model and tokenizer
model_path = "outputs"  # Path to your fine-tuned model directory
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Function for generating code summary using the trained model
def generate_summary(input_code, tokenizer, model, max_length=512):
    input_text = f"display code summarization\n{input_code}"
    input_ids = tokenizer(input_text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
    with torch.no_grad():
        output = model.generate(input_ids)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output

# Test the model with your input code
input_code = """
IDENTIFICATION DIVISION.
PROGRAM-ID. HelloWorld.
PROCEDURE DIVISION.
    DISPLAY 'Hello, world!'.
    STOP RUN.
"""

generated_summary = generate_summary(input_code, tokenizer, model)
print("Generated Documentation:")
print(generated_summary)
