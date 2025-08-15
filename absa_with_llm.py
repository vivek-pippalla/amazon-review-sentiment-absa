import ollama
import re

def aspect_based_sentiment_llm(review_text):
 
    prompt = f"""You are an expert product review analyst specializing in aspect-based sentiment analysis.

            TASK: Analyze the product review below and extract specific product aspects with their corresponding sentiments.

            REVIEW: "{review_text}"

            INSTRUCTIONS:
            1. Identify ONLY concrete product features, components, or characteristics mentioned in the review
            2. Ignore generic terms like "product", "item", "it", "thing", "overall", "general"
            3. For each aspect, determine if the sentiment is Positive, Negative, or Neutral based on the context
            4. Consider negations, comparisons, and subtle language carefully
            5. If an aspect is mentioned but no clear sentiment is expressed, mark as Neutral

            OUTPUT FORMAT:
            Return only the aspect-sentiment pairs in this exact format:
            aspect1: sentiment
            aspect2: sentiment
            ...

            EXAMPLES:
            - "battery life: Positive" (if battery life is praised,. You don't have to mention life or any related wrords as aspects)
            - "screen quality: Negative" (if screen quality is criticized)
            - "price: Neutral" (if price is mentioned without clear sentiment)

            IMPORTANT:
            - Do NOT include explanations, introductions, or formatting
            - Do NOT extract aspects that aren't explicitly mentioned
            - Focus on specific product features, not general opinions
            - Each aspect should be a noun or noun phrase (2-4 words max)

            BEGIN ANALYSIS:"""

    
    try:
        response = ollama.chat(
            model='llama3.2',
            messages=[{"role": "user", "content": prompt}]
        )
        lines = response['message']['content'].strip().splitlines()
        aspect_sentiment = {}

        for line in lines:
            match = re.match(r"(.+?):\s*(Positive|Negative|Neutral)", line, re.IGNORECASE)
            if match:
                aspect = match.group(1).strip().lower()
                sentiment = match.group(2).capitalize()
                aspect_sentiment[aspect] = sentiment

        return aspect_sentiment

    except Exception as e:
        print("ABSA LLM error:", e)
        return {}
