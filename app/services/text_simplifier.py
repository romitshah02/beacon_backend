#!/usr/bin/env python3
"""
Text Simplifier using Google's T5-small model
This script loads the T5-small model locally and simplifies input sentences to basic English.
"""

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import sys
import re

def load_model():
    """
    Load the T5-small model and tokenizer for English text simplification.
    Returns: tuple of (model, tokenizer)
    """
    print("Loading T5-small model... This may take a moment on first run.")
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer

def simplify_text(text, model, tokenizer):
    """
    Simplify and summarize the input text using the T5-small model to basic English.
    Args:
        text (str): Input text to simplify
        model: T5 model
        tokenizer: T5 tokenizer
    Returns:
        str: Simplified and summarized text in English
    """
    # Prompts specifically designed for summarization and simplification
    prompts = [
        f"summarize in simple English: {text}",
        f"explain briefly in simple terms: {text}",
        f"rewrite this in fewer, simpler words: {text}",
        f"make this shorter and easier to understand: {text}",
        f"summarize to basic English: {text}",
        f"simplify and shorten: {text}",
        f"explain in simple English in one sentence: {text}"
    ]
    
    best_result = None
    original_length = len(text.split())
    
    for prompt in prompts:
        try:
            inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = inputs.to('cpu')
            model = model.to('cpu')
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=min(256, len(text.split()) * 2),  # Limit output length
                    num_beams=4,
                    length_penalty=0.8,  # Prefer shorter outputs
                    early_stopping=True,
                    do_sample=False,
                    no_repeat_ngram_size=2,
                    temperature=0.7
                )
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = result.strip()
            
            # Clean up the result
            result = clean_output(result)
            
            # Check if result is valid and different from input
            if is_valid_simplified_result(result, text, original_length):
                best_result = result
                break
                
        except Exception as e:
            continue
    
    # If no good result found, try manual simplification
    if best_result is None or best_result.lower() == text.lower():
        best_result = manual_simplify_and_summarize(text)
    
    return best_result

def clean_output(text):
    """
    Clean up the model output to remove unwanted prefixes and ensure English.
    """
    # Remove common unwanted prefixes
    unwanted_prefixes = [
        "simplify to basic english:",
        "rewrite in simple english:",
        "make this easier to read in english:",
        "translate to simple english:",
        "explain in simple english:",
        "simplify:",
        "simplified:",
        "einfacher:",
        "vereinvereinen englische text:",
        "text simplifie simplifie :",
        "this so a 10-year-old can understand:"
    ]
    
    text = text.strip()
    
    for prefix in unwanted_prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    
    # Remove any remaining non-English characters at the beginning
    text = re.sub(r'^[^a-zA-Z\s]+', '', text).strip()
    
    return text

def is_valid_english_result(text):
    """
    Check if the result is valid English text.
    """
    if len(text.strip()) < 5:
        return False
    
    # Check if it contains mostly English words
    english_words = text.lower().split()
    if len(english_words) < 2:
        return False
    
    # Enhanced list of common English words
    common_english_words = [
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
        'what', 'when', 'where', 'why', 'how', 'who', 'which', 'whom', 'whose',
        'if', 'then', 'else', 'because', 'since', 'while', 'before', 'after', 'during',
        'up', 'down', 'out', 'off', 'over', 'under', 'above', 'below', 'between', 'among',
        'here', 'there', 'now', 'then', 'today', 'yesterday', 'tomorrow',
        'good', 'bad', 'big', 'small', 'new', 'old', 'young', 'hot', 'cold', 'warm', 'cool',
        'fast', 'slow', 'easy', 'hard', 'simple', 'complex', 'important', 'necessary'
    ]
    
    english_word_count = 0
    total_words = 0
    
    for word in english_words:
        # Clean the word (remove punctuation)
        clean_word = re.sub(r'[^\w]', '', word.lower())
        if clean_word:
            total_words += 1
            if clean_word in common_english_words or (len(clean_word) > 2 and clean_word.isalpha()):
                english_word_count += 1
    
    if total_words == 0:
        return False
    
    # If more than 60% of words are English-like, consider it valid
    return english_word_count / total_words > 0.6

def is_valid_simplified_result(result, original_text, original_length):
    """
    Check if the result is valid simplified English text that's different from input, and much simpler.
    """
    if not is_valid_english_result(result):
        return False
    result_lower = result.lower().strip()
    original_lower = original_text.lower().strip()
    # Always reject if too similar
    if result_lower == original_lower or result_lower in original_lower or original_lower in result_lower:
        return False
    # Must be at least 30% shorter
    result_length = len(result.split())
    if result_length >= original_length * 0.7:
        return False
    # Must be a short sentence
    if result_length < 2:
        return False
    return True

def manual_simplify_and_summarize(text):
    """
    Aggressive manual simplification and summarization for dyslexia-friendly output.
    """
    # Use a very limited set of simple words
    simple_words = {
        'utilize': 'use', 'implement': 'use', 'facilitate': 'help', 'subsequently': 'then',
        'consequently': 'so', 'nevertheless': 'but', 'furthermore': 'also', 'moreover': 'also',
        'additionally': 'also', 'therefore': 'so', 'thus': 'so', 'hence': 'so', 'accordingly': 'so',
        'thereafter': 'then', 'meanwhile': 'while', 'whilst': 'while', 'whom': 'who', 'wherein': 'where',
        'whereby': 'how', 'whereas': 'but', 'notwithstanding': 'but', 'despite': 'but', 'although': 'but',
        'however': 'but', 'nonetheless': 'but', 'regardless': 'but', 'irrespective': 'but', 'approximately': 'about',
        'subsequently': 'then', 'previously': 'before', 'currently': 'now', 'ultimately': 'finally', 'initially': 'first',
        'frequently': 'often', 'occasionally': 'sometimes', 'rarely': 'seldom', 'immediately': 'now', 'eventually': 'finally',
        'gradually': 'slowly', 'rapidly': 'quickly', 'significantly': 'much', 'considerably': 'much', 'substantially': 'much',
        'extremely': 'very', 'exceptionally': 'very', 'particularly': 'especially', 'specifically': 'especially',
        'primarily': 'mainly', 'essentially': 'basically', 'fundamentally': 'basically', 'consequently': 'so',
        'accordingly': 'so', 'hence': 'so', 'thus': 'so', 'therefore': 'so',
        # Add more as needed
    }
    # Replace complex words
    for complex_word, simple_word in simple_words.items():
        text = re.sub(r'\b' + complex_word + r'\b', simple_word, text, flags=re.IGNORECASE)
    # Remove extra words and keep only the main idea
    sentences = re.split(r'[.!?]', text)
    if sentences:
        main_sentence = sentences[0].strip()
    else:
        main_sentence = text.strip()
    # Use only the most common words
    allowed = set(['the','a','an','and','or','but','in','on','at','to','for','of','with','by','is','are','was','were','be','been','have','has','had','do','does','did','will','would','could','should','may','might','can','this','that','these','those','i','you','he','she','it','we','they','me','him','her','us','them','my','your','his','her','its','our','their','what','when','where','why','how','who','if','then','else','because','since','while','before','after','during','up','down','out','off','over','under','above','below','between','among','here','there','now','then','today','yesterday','tomorrow','good','bad','big','small','new','old','young','hot','cold','warm','cool','fast','slow','easy','hard','simple','important','necessary','use','help','also','so','now','then','about','mainly','first','often','sometimes','seldom','very','especially','basically'])
    words = [w for w in main_sentence.split() if w.lower() in allowed]
    if not words:
        words = [w for w in main_sentence.split() if w.isalpha()]
    # Always output a short, simple sentence
    result = ' '.join(words)
    if len(result) > 60:
        result = ' '.join(result.split()[:15]) + '.'
    result = result.capitalize()
    if not result:
        result = 'This is important.'
    return result

def main():
    """
    Main function to run the text simplifier.
    """
    print("=" * 50)
    print("Text Simplifier using T5-small Model (Basic English)")
    print("=" * 50)
    
    try:
        # Load the model
        model, tokenizer = load_model()
        
        print("\nEnter a sentence to simplify and summarize in basic English (or 'quit' to exit):")
        
        while True:
            # Get input from user
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                print("Please enter some text to simplify.")
                continue
            
            try:
                # Simplify the text
                print("Simplifying and summarizing to basic English...")
                simplified = simplify_text(user_input, model, tokenizer)
                
                print(f"\nOriginal: {user_input}")
                print(f"Simplified Summary: {simplified}")
                
            except Exception as e:
                print(f"Error simplifying text: {e}")
    
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have the required packages installed:")
        print("pip install torch transformers sentencepiece")
        sys.exit(1)

if __name__ == "__main__":
    main() 