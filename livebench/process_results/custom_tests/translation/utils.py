"""
Translation Test - Continuous Scoring

This test evaluates models on their ability to translate sentences between languages.
Uses text similarity metrics for continuous scoring (0-1) compared to reference translations.
"""
import re
from difflib import SequenceMatcher


def translation_process_results(ground_truth: str, llm_answer: str, debug=False) -> float:
    """
    Grade translation quality using text similarity.

    This function provides continuous scoring from 0 to 1 based on:
    1. Token overlap (word-level matching)
    2. Character-level similarity
    3. Length ratio penalty
    4. N-gram overlap

    Args:
        ground_truth: The reference translation
        llm_answer: The model's translation
        debug: If True, print debug information

    Returns:
        A score between 0 and 1 representing translation quality
    """
    # Extract the translation from common formatting
    parsed_answer = extract_translation(llm_answer)

    if not parsed_answer or not ground_truth:
        if debug:
            print('EMPTY TRANSLATION')
            print('GROUND TRUTH:', ground_truth)
            print('RAW ANSWER:', llm_answer)
        return 0.0

    # Normalize both texts
    ref_normalized = normalize_text(ground_truth)
    ans_normalized = normalize_text(parsed_answer)

    # Calculate multiple similarity metrics
    token_sim = token_similarity(ref_normalized, ans_normalized)
    char_sim = character_similarity(ref_normalized, ans_normalized)
    length_penalty = length_ratio_penalty(ref_normalized, ans_normalized)
    ngram_sim = ngram_similarity(ref_normalized, ans_normalized)

    # Weighted combination for translation
    # Token similarity is crucial (40%)
    # Character similarity helps with spelling/morphology (30%)
    # N-gram similarity captures word order/phrasing (25%)
    # Length penalty (5%)
    score = (0.40 * token_sim + 0.30 * char_sim + 0.25 * ngram_sim + 0.05 * (1 - length_penalty))

    # Bonus for high character similarity (good translations are precise)
    if char_sim > 0.7:
        score = min(1.0, score * 1.1)

    # Ensure score is in [0, 1]
    score = max(0.0, min(1.0, score))

    if debug:
        print(f'SCORE: {score:.3f}')
        print(f'  Token similarity: {token_sim:.3f}')
        print(f'  Char similarity: {char_sim:.3f}')
        print(f'  N-gram similarity: {ngram_sim:.3f}')
        print(f'  Length penalty: {length_penalty:.3f}')
        print('GROUND TRUTH:', ground_truth)
        print('PARSED ANSWER:', parsed_answer)
        print('RAW ANSWER:', llm_answer[:200])

    return score


def extract_translation(text: str) -> str:
    """
    Extract the translation from model output.

    Tries multiple extraction methods:
    1. Content within quotes
    2. Content after "Translation:" or similar markers
    3. The entire text if no markers found

    Args:
        text: Raw model output

    Returns:
        Extracted translation text
    """
    # Try to find quoted text
    quote_patterns = [
        r'"([^"]+)"',  # Double quotes
        r"'([^']+)'",  # Single quotes
        r'"([^"]+)"',  # Curly quotes
        r'«([^»]+)»',  # French quotes
    ]

    for pattern in quote_patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Return the longest match (likely the translation)
            return max(matches, key=len)

    # Try to find text after markers
    markers = [
        r'[Tt]ranslation:\s*(.+?)(?:\n|$)',
        r'[Tt]ranslated:\s*(.+?)(?:\n|$)',
        r'[Tt]ranslate[ds]?\s+text:\s*(.+?)(?:\n|$)',
        r'[Oo]utput:\s*(.+?)(?:\n|$)',
        r'[Rr]esult:\s*(.+?)(?:\n|$)',
    ]

    for marker in markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # If no special formatting found, try to extract the first substantial sentence
    # This handles cases where the model just outputs the translation directly
    sentences = re.split(r'[.!?]+', text)
    for sentence in sentences:
        sentence = sentence.strip()
        # Skip very short sentences or sentences that look like meta-commentary in English
        if len(sentence) > 15 and not any(word in sentence.lower() for word in
                                          ['here is', 'here\'s', 'the translation', 'translates to',
                                           'in english', 'in spanish', 'in french', 'in german']):
            return sentence

    # Fall back to using the entire text
    return text.strip()


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Remove common punctuation (but keep word boundaries)
    text = re.sub(r'[,;:!?\.\-—–]', ' ', text)

    # Remove extra spaces again
    text = ' '.join(text.split())

    return text.strip()


def token_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between token sets.

    Args:
        text1: First text (normalized)
        text2: Second text (normalized)

    Returns:
        Jaccard similarity score (0-1)
    """
    tokens1 = set(text1.split())
    tokens2 = set(text2.split())

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    if union == 0:
        return 0.0

    return intersection / union


def character_similarity(text1: str, text2: str) -> float:
    """
    Calculate character-level similarity using SequenceMatcher.

    Args:
        text1: First text (normalized)
        text2: Second text (normalized)

    Returns:
        Similarity ratio (0-1)
    """
    if not text1 or not text2:
        return 0.0

    return SequenceMatcher(None, text1, text2).ratio()


def ngram_similarity(text1: str, text2: str, n=2) -> float:
    """
    Calculate n-gram overlap similarity (default bigrams).

    This captures word order and common phrases better than individual tokens.

    Args:
        text1: First text (normalized)
        text2: Second text (normalized)
        n: N-gram size (default 2 for bigrams)

    Returns:
        N-gram Jaccard similarity score (0-1)
    """
    tokens1 = text1.split()
    tokens2 = text2.split()

    if len(tokens1) < n or len(tokens2) < n:
        # Fall back to token similarity if texts are too short
        return token_similarity(text1, text2)

    # Create n-grams
    ngrams1 = set(tuple(tokens1[i:i+n]) for i in range(len(tokens1) - n + 1))
    ngrams2 = set(tuple(tokens2[i:i+n]) for i in range(len(tokens2) - n + 1))

    if not ngrams1 or not ngrams2:
        return 0.0

    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)

    if union == 0:
        return 0.0

    return intersection / union


def length_ratio_penalty(text1: str, text2: str) -> float:
    """
    Calculate penalty based on length difference.

    Penalizes translations that are too short or too long compared to reference.

    Args:
        text1: Reference text (normalized)
        text2: Translation text (normalized)

    Returns:
        Penalty value (0 = no penalty, 1 = maximum penalty)
    """
    len1 = len(text1.split())
    len2 = len(text2.split())

    if len1 == 0 or len2 == 0:
        return 1.0

    ratio = min(len1, len2) / max(len1, len2)

    # Convert to penalty (1 - ratio)
    # If lengths are similar, penalty is low
    # If one is much longer/shorter, penalty is high
    penalty = 1 - ratio

    # Soften the penalty (translations can vary in length across languages)
    penalty = penalty * 0.6

    return penalty
