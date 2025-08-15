import ollama
import html

def generate_summary(text):

    prompt = f"""You are a professional review summarizer. Extract the key points from this product review.

            GUIDELINES:
            - Identify 5-8 main points mentioned in the review
            - Use the reviewer's own words when possible
            - Focus on product features, user experience, and outcomes
            - Keep each phrase concise (2-6 words)
            - Maintain factual accuracy

            REVIEW: "{text}"

            Extract key phrases (comma-separated, single line):"""



    try:
        response = ollama.chat(
            model='llama3.2',
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        summary = response["message"]["content"].strip()
    except Exception as e:
        summary = f"Could not generate summary: {e}"

    return summary



