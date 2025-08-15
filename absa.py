import numpy as np
import spacy
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en.stop_words import STOP_WORDS
# Load spaCy model for NLP tasks
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # Fallback if model not installed
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize CPU-compatible transformer model for sentiment analysis
# Using a smaller, CPU-friendly model
try:
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    sentiment_model.eval()  # Set to evaluation mode
    device = torch.device("cpu")
    sentiment_model.to(device)
    has_transformer = True
except:
    print("Transformer model not available, falling back to rule-based approach")
    has_transformer = False

# ----------------------------
### Domain Knowledge and Lexicons
# ----------------------------
# Generic filter set for aspects that shouldn't be considered
generic_filter_set = {
    "this phone", "phone", "day", "this product", "product", "item", "thing", 
    "unit", "order", "package", "pack", "amazon", "review", "device", "gadget", 
    "object", "overall", "model", "version", "specification", "specs", "quality", 
    "service", "support", "general", "common", "standard", "conditions", "condition"
}

# Generic aspects that are too broad to be meaningful
generic_aspects = {
    "product", "replacement", "order", "phone", "it", "this", "that", "one", 
    "item", "unit", "thing", "object", "stuff", "package", "pack", "box", 
    "container", "shipment", "service", "delivery", "device", "gadget", "model", 
    "version", "specification", "specs", "general", "standard", "common", "quality", 
    "support", "lot", "kind", "type", "sorts", "part", "bits", "piece", "condition"
}

# Non-informative or non-aspect words
excluded_words = {
    "day", "days", "week", "weeks", "month", "months", "year", "years", "time", 
    "sometimes", "often", "always", "never", "today", "tomorrow", "yesterday",
    "then", "now", "later", "soon", "eventually", "finally", "initially", "aspect",
    "review", "opinion", "thought", "idea", "view", "fact", "thing", "case", "few", "few hours" , "hours", "hour", "minutes" , "minute", "seconds", "second",
    "moment", "period", "duration", "timeframe", "span", "interval", "phase", "stage",
    "instance", "situation", "scenario", "context", "environment", "setting", "background",
    "circumstance", "condition", "state", "place", "location", "area", "region", "zone", "space",
    "site", "venue", "locale", "point", "spot", "position", "situation", "positioning", "placement"
}

# Negative sentiment indicators
negative_indicators = {
    "not", "no", "never", "hard", "difficult", "poor", "flaw", "problem", "issue", "bad", "wrong", 
    "terrible", "awful", "disappointing", "disappointed", "subpar", "mediocre", "horrible", 
    "unsatisfactory", "defective", "broken", "ugly", "overpriced", "waste", "wasteful", "lousy", 
    "unreliable", "unresponsive", "inferior", "dislike", "hate", "annoying", "dismal", "crappy", "sucks", 
    "abysmal", "pitiful", "drain", "drains", "lacking", "fails", "fail", "failing", "useless"
}

# Positive sentiment indicators
positive_indicators = {
    "great", "good", "excellent", "best", "amazing", "wonderful", "fantastic", "awesome", 
    "superb", "incredible", "impressive", "outstanding", "perfect", "brilliant", "exceptional", 
    "phenomenal", "love", "loved", "delightful", "enjoyable", "marvelous", "spectacular", "fabulous", 
    "stunning", "terrific", "splendid", "enjoy", "enjoys", "enjoying", "enjoyed"
}

# Neutral sentiment indicators
neutral_indicators = {
    "okay", "ok", "decent", "average", "fair", "acceptable", "moderate", "mediocre", 
    "regular", "standard", "common", "usual", "typical", "normal", "ordinary", "so-so"
}

# Negation words that can flip sentiment
negation_words = {
    "not", "no", "never", "neither", "nor", "none", "isn't", "aren't", "wasn't", 
    "weren't", "don't", "doesn't", "didn't", "cannot", "can't", "couldn't", 
    "shouldn't", "wouldn't", "won't", "without", "lack", "lacks", "lacking"
}

# Aspect categories and their related terms (for hierarchical categorization)
aspect_categories = {
    "display": ["screen", "display", "resolution", "brightness", "clarity", "color", "lcd", "oled", "panel"],
    "battery": ["battery", "charge", "charging", "power", "life", "drain", "drains", "draining", "endurance"],
    "camera": ["camera", "picture", "photo", "video", "megapixel", "lens", "zoom", "focus", "aperture", "photography"],
    "performance": ["performance", "speed", "fast", "slow", "processor", "cpu", "gpu", "lag", "responsive", "snappy"],
    "design": ["design", "look", "build", "quality", "construction", "material", "premium", "aesthetic", "style"],
    "audio": ["sound", "audio", "speaker", "volume", "bass", "treble", "noise", "headphone", "earphone"],
    "storage": ["storage", "memory", "space", "gb", "capacity"],
    "software": ["software", "os", "operating", "system", "android", "ios", "app", "application", "update", "firmware"],
    "connectivity": ["wifi", "bluetooth", "signal", "reception", "network", "connection", "nfc", "usb", "port"]
}

# Qualifiers and their parent aspects (for handling cases like "low light conditions" for camera)
aspect_qualifiers = {
    "camera": ["low light", "night", "dark", "bright", "indoor", "outdoor", "portrait", "landscape", "video", "selfie"],
    "display": ["outdoor", "sunlight", "viewing angle", "brightness", "color"],
    "battery": ["heavy use", "standby", "idle", "screen on", "gaming", "battery life"],
    "performance": ["gaming", "multitask", "multitasking", "app", "background", "loading"],
    "audio": ["loud", "quiet", "bass", "treble", "vocal", "voice", "call"]
}

# Intensity modifiers that strengthen or weaken sentiment
intensity_modifiers = {
    "strengthen": ["very", "extremely", "incredibly", "remarkably", "exceptionally", "absolutely", "really", "truly"],
    "weaken": ["somewhat", "slightly", "a bit", "a little", "kind of", "sort of", "rather", "fairly"]
}

# ----------------------------
### Helper Functions for Aspect Extraction and Preprocessing
# ----------------------------
def normalize_text(text):
    """Normalize text by removing extra whitespace, punctuation and converting to lowercase."""
    # Convert to lowercase
    text = text.lower().strip()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text

def normalize_aspect(aspect):
    """Normalize aspect: lowercase and remove leading articles."""
    aspect = aspect.lower().strip()
    for article in ["a ", "an ", "the "]:
        if aspect.startswith(article):
            aspect = aspect[len(article):]
    return aspect

def is_generic(aspect):
    """Return True if the aspect is generic, contains digits, or is in the excluded list."""
    aspect_lower = aspect.lower()
    return (aspect_lower in generic_aspects or 
            any(char.isdigit() for char in aspect) or 
            aspect_lower in excluded_words or
            len(aspect) <= 2 or  # Very short aspects are likely not meaningful
            all(word in STOP_WORDS for word in aspect_lower.split()))

def is_qualifier_not_aspect(text, aspect):
    """Check if the extracted text is a qualifier rather than a standalone aspect."""
    for category, qualifiers in aspect_qualifiers.items():
        if any(qualifier in aspect for qualifier in qualifiers):
            # This might be a qualifier - check if it's associated with a real aspect in the text
            for term in aspect_categories[category]:
                if term in text and term not in aspect:
                    return True, category
    return False, None

def get_aspect_category(aspect):
    """Map an aspect to its category based on domain knowledge."""
    aspect_lower = aspect.lower()
    
    # Check direct category matches
    for category, terms in aspect_categories.items():
        for term in terms:
            if term in aspect_lower:
                return category
    
    # Check qualifier matches
    for category, qualifiers in aspect_qualifiers.items():
        for qualifier in qualifiers:
            if qualifier in aspect_lower:
                return category
    
    return "general"

def get_related_aspects(aspects, threshold=0.7):
    """Group related aspects based on semantic similarity."""
    if not aspects:
        return {}
    
    # Create document-term matrix
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3))
    aspect_matrix = vectorizer.fit_transform(aspects)
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(aspect_matrix)
    
    # Group related aspects
    related_groups = {}
    used_aspects = set()
    
    for i, aspect in enumerate(aspects):
        if aspect in used_aspects:
            continue
            
        related = [aspect]
        used_aspects.add(aspect)
        
        for j, other_aspect in enumerate(aspects):
            if i != j and other_aspect not in used_aspects and similarity_matrix[i, j] > threshold:
                related.append(other_aspect)
                used_aspects.add(other_aspect)
        
        # Use the longest aspect as the representative
        main_aspect = max(related, key=len)
        related_groups[main_aspect] = related
    
    return related_groups

# ----------------------------
### Advanced Aspect Extraction
# ----------------------------
def extract_aspects_improved(text):
    """
    Improved aspect extraction using:
    1. Noun chunks from spaCy
    2. Dependency parsing for aspect-opinion/verb-based patterns
    3. Compound and adjective-noun phrases
    4. Regex-based pattern matching
    5. Lemmatization and qualifier-aware filtering
    """
    doc = nlp(normalize_text(text))
    candidate_aspects = set()
    
    # Strategy 1: Extract from noun chunks (multi-word aspects)
    for chunk in doc.noun_chunks:
        if 1 <= len(chunk.text.split()) <= 4:
            candidate = normalize_aspect(chunk.text)
            if candidate and not is_generic(candidate):
                candidate_aspects.add(candidate)
    
    # Strategy 2: Dependency parsing + opinion verbs
    opinion_verbs = positive_indicators.union(negative_indicators)
    
    for token in doc:
        # VERB + NOUN (e.g., "hate camera", "love battery")
        if token.pos_ == "VERB" and token.lemma_.lower() in opinion_verbs:
            for child in token.children:
                if child.pos_ == "NOUN" and not is_generic(child.text):
                    candidate_aspects.add(child.lemma_.lower())
        
        # NOUN with adjectival modifiers
        if token.pos_ == "NOUN" and token.dep_ in {"nsubj", "dobj", "pobj"}:
            if not is_generic(token.text):
                candidate_aspects.add(token.lemma_.lower())
            
            for mod in token.children:
                if mod.pos_ == "ADJ":
                    phrase = f"{mod.text.lower()} {token.text.lower()}"
                    if not is_generic(phrase):
                        candidate_aspects.add(phrase)
    
    # Strategy 3: Compound nouns (e.g., "battery life")
    for token in doc:
        if token.pos_ == "NOUN" and not is_generic(token.text):
            compounds = [child.text for child in token.children if child.dep_ == "compound"]
            if compounds:
                phrase = " ".join([c.lower() for c in compounds] + [token.text.lower()])
                candidate_aspects.add(phrase)
    
    # Strategy 4: Regex-based pattern matching
    patterns = [
        r"(\w+\s+quality)",      
        r"(\w+\s+life)",         
        r"(\w+\s+performance)",  
        r"(low\s+light\s+\w+)",  
        r"(\w+\s+resolution)"    
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            if not is_generic(match):
                candidate_aspects.add(match)
    
    # Strategy 5: Qualifier-aware filtering
    final_aspects = set()
    for aspect in candidate_aspects:
        is_qualifier, category = is_qualifier_not_aspect(text, aspect)
        if not is_qualifier:
            final_aspects.add(aspect)
        else:
            # If the qualifier gives useful context (e.g., "low light"), keep it
            if any(q in aspect for q in ["low light", "fast charging", "high quality"]):
                final_aspects.add(aspect)

    # Normalize with lemmatization to reduce redundant forms
    lemmatized_aspects = set()
    for aspect in final_aspects:
        doc_aspect = nlp(aspect)
        lemma = " ".join([token.lemma_ for token in doc_aspect])
        lemmatized_aspects.add(lemma.lower())
    
    return list(lemmatized_aspects)


def filter_and_deduplicate_aspects(aspects, review_text, fuzzy_threshold=0.75):
    """Filter and deduplicate aspects using semantic similarity and fuzzy matching."""
    if not aspects:
        return {}
    
    # Step 1: Filter out generic aspects
    filtered_aspects = [aspect for aspect in aspects if not is_generic(aspect)]
    
    # Step 2: Group related aspects (e.g., "screen" and "screen resolution")
    related_groups = get_related_aspects(filtered_aspects, threshold=fuzzy_threshold)
    
    # Step 3: Resolve hierarchical aspects (handle qualifiers properly)
    aspect_hierarchy = {}
    for main_aspect, related in related_groups.items():
        # Check if any aspect contains another
        for aspect1 in related:
            for aspect2 in related:
                if aspect1 != aspect2 and aspect1 in aspect2:
                    # aspect2 is more specific, so we'll keep that
                    aspect_hierarchy[aspect1] = aspect2
    
    # Step 4: Finalize the unique aspect list
    unique_aspects = set()
    for main_aspect, related in related_groups.items():
        # Pick the most specific aspect from each group
        specific_aspect = main_aspect
        for aspect in related:
            # Use longer aspects as they tend to be more specific
            if len(aspect) > len(specific_aspect):
                specific_aspect = aspect
            # If an aspect is in the hierarchy, use its more specific version
            if aspect in aspect_hierarchy:
                specific_aspect = aspect_hierarchy[aspect]
        
        unique_aspects.add(specific_aspect)
    
    # Step 5: Filter out aspects that are actually qualifiers
    final_aspects = set()
    for aspect in unique_aspects:
        is_qualifier, _ = is_qualifier_not_aspect(review_text, aspect)
        if not is_qualifier and len(aspect.split()) <= 4:  # Avoid overly long aspects
            final_aspects.add(aspect)
    
    return list(final_aspects)

# ----------------------------
### Advanced Sentiment Analysis
# ----------------------------
def check_negation_context(text, target_word, window_size=5):
    """Check if a target word is negated within a given context window."""
    words = text.lower().split()
    if target_word not in words:
        return False
    
    target_indices = [i for i, word in enumerate(words) if word == target_word]
    
    for idx in target_indices:
        # Check for negation words within the window
        start = max(0, idx - window_size)
        end = min(len(words), idx + 1)  # +1 because we want to include the target word
        context_before = words[start:end]
        
        if any(neg in context_before for neg in negation_words):
            return True
    
    return False

def analyze_clause_sentiment(clause, aspect):
    """Analyze sentiment for a specific clause containing an aspect."""
    clause_lower = clause.lower()
    
    # Check for direct sentiment indicators
    pos_indicators = sum(1 for word in positive_indicators if word in clause_lower)
    neg_indicators = sum(1 for word in negative_indicators if word in clause_lower)
    neutral_indicators = sum(1 for word in neutral_indicators if word in clause_lower)
    
    # Check for negation patterns
    negated_positive = sum(1 for word in positive_indicators 
                          if word in clause_lower and check_negation_context(clause_lower, word))
    negated_negative = sum(1 for word in negative_indicators 
                          if word in clause_lower and check_negation_context(clause_lower, word))
    
    # Adjust counts based on negation
    pos_indicators = pos_indicators - negated_positive + negated_negative
    neg_indicators = neg_indicators - negated_negative + negated_positive
    
    # Check for intensity modifiers
    intensity_factor = 1.0
    for word in intensity_modifiers["strengthen"]:
        if word in clause_lower:
            intensity_factor = 1.5
            break
    for word in intensity_modifiers["weaken"]:
        if word in clause_lower:
            intensity_factor = 0.5
            break
    
    # Apply intensity to indicator counts
    pos_indicators = pos_indicators * intensity_factor
    neg_indicators = neg_indicators * intensity_factor
    
    # Calculate sentiment score
    if pos_indicators > neg_indicators:
        return min(0.5 + 0.1 * pos_indicators, 1.0)  # Scale between 0.5 and 1.0
    elif neg_indicators > pos_indicators:
        return max(0.5 - 0.1 * neg_indicators, 0.0)  # Scale between 0.0 and 0.5
    elif neutral_indicators > 0:
        return 0.5  # Neutral sentiment
    else:
        return 0.5  # Default to neutral if no indicators found
        
def analyze_sentiment_with_transformer(text, aspect):
    """Use transformer model to predict sentiment."""
    if not has_transformer:
        return None
    
    try:
        # Create input with aspect highlighted using [aspect] tags
        highlighted_text = text.replace(aspect, f"[{aspect}]")
        inputs = tokenizer(highlighted_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            
            # DistilBERT SST-2 model: 0=negative, 1=positive
            sentiment_score = predictions[0][1].item()  # Positive class probability
            
            return sentiment_score
    except Exception as e:
        print(f"Transformer model error: {e}")
        return None

def get_aspect_context(review, aspect):
    """Extract relevant context for an aspect considering contrastive structures."""
    doc = nlp(review)
    
    # Get all sentences containing the aspect
    aspect_sentences = []
    for sent in doc.sents:
        if aspect.lower() in sent.text.lower():
            aspect_sentences.append(sent.text)
    
    if not aspect_sentences:
        return [review]  # Fallback to full review if aspect not found
    
    # Process sentences with contrastive structures
    relevant_clauses = []
    for sentence in aspect_sentences:
        # Check for contrastive conjunctions
        contrastive_markers = ["but", "however", "although", "though", "yet", "nevertheless", "nonetheless"]
        
        for marker in contrastive_markers:
            if f" {marker} " in sentence.lower():
                clauses = re.split(f" {marker} ", sentence, flags=re.IGNORECASE)
                # Find which clause contains our aspect
                for clause in clauses:
                    if aspect.lower() in clause.lower():
                        relevant_clauses.append(clause.strip())
                        break
                break
        else:
            # No contrastive structure found, use the whole sentence
            relevant_clauses.append(sentence)
    
    return relevant_clauses

def get_aspect_sentiment_improved(review, aspect):
    """
    Enhanced sentiment analysis for a specific aspect, handling contrastive structures
    and using a combination of transformer and rule-based approaches.
    """
    # Get relevant context for the aspect
    relevant_contexts = get_aspect_context(review, aspect)
    
    # Try transformer model first for each context
    if has_transformer:
        transformer_scores = []
        for context in relevant_contexts:
            score = analyze_sentiment_with_transformer(context, aspect)
            if score is not None:
                transformer_scores.append(score)
        
        if transformer_scores:
            avg_transformer_score = np.mean(transformer_scores)
            # Convert to sentiment label
            if avg_transformer_score > 0.6:
                return "Positive", avg_transformer_score
            elif avg_transformer_score < 0.4:
                return "Negative", avg_transformer_score
            else:
                return "Neutral", avg_transformer_score
    
    # Fallback to rule-based sentiment analysis
    rule_based_scores = []
    for context in relevant_contexts:
        score = analyze_clause_sentiment(context, aspect)
        rule_based_scores.append(score)
    
    avg_rule_score = np.mean(rule_based_scores)
    
    # Convert to sentiment label
    if avg_rule_score > 0.6:
        return "Positive", avg_rule_score
    elif avg_rule_score < 0.4:
        return "Negative", avg_rule_score
    else:
        return "Neutral", avg_rule_score

# ----------------------------
### Main ABSA Function
# ----------------------------
def aspect_based_sentiment_improved(review):
    """
    Perform improved aspect-based sentiment analysis that handles hierarchical aspects,
    contrastive structures, and provides more accurate sentiment classification.
    """
    # Normalize review text
    normalized_review = normalize_text(review)
    
    # Extract candidate aspects
    candidate_aspects = extract_aspects_improved(normalized_review)
    
    # Filter and deduplicate aspects
    unique_aspects = filter_and_deduplicate_aspects(candidate_aspects, normalized_review)
    
    # Get sentiment for each aspect
    aspect_sentiments = {}
    for aspect in unique_aspects:
        sentiment, score = get_aspect_sentiment_improved(normalized_review, aspect)
        aspect_sentiments[aspect] = sentiment
    
    return aspect_sentiments


