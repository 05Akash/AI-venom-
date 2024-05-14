from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

model_path = "outputs"  
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained(model_path)

def generate_text(input_text, tokenizer, model, max_length=512):
    input_ids = tokenizer(input_text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
    with torch.no_grad():
        output = model.generate(input_ids)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output

cobol_code = """
display code summarization
IDENTIFICATION DIVISION.
PROGRAM-ID. HelloWorld.
PROCEDURE DIVISION.
    DISPLAY 'Hello, world!'.
    STOP RUN.
"""

input_text = f"{cobol_code} <sep> Generate documentation for this code."

generated_text = generate_text(input_text, tokenizer, model)
print("Generated Documentation:")
print(generated_text)
