# Load the pre-trained T5-base model
from transformers import T5Tokenizer, T5ForConditionalGeneration

summarizer_tokenizer = T5Tokenizer.from_pretrained('t5-base', legacy=False)
summarizer_model = T5ForConditionalGeneration.from_pretrained('t5-base')


# ----------------------------
# Text Summarization
# ----------------------------

def generate_summary(text, max_length=50, min_length=10):
    """
    Generate a summary for the given text using T5-small with CPU optimization.
    """
    input_text = "summarize: " + text
    inputs = summarizer_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    summary_ids = summarizer_model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length,
        length_penalty=1.0,  # Reduce complexity slightly
        num_beams=2,  # Lower beams for faster execution
        early_stopping=True
    )

    summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
