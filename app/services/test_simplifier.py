#!/usr/bin/env python3
"""
Test script for the text simplifier
"""

from text_simplifier import load_model, simplify_text

def test_simplifier():
    """Test the text simplifier with a sample sentence"""
    
    # Sample complex sentence
    complex_text = "The intricate mechanism of photosynthesis involves the conversion of solar energy into chemical energy through a series of biochemical reactions."
    
    print("Testing Text Simplifier...")
    print("=" * 50)
    
    try:
        # Load the model
        model, tokenizer = load_model()
        
        # Simplify the text
        print(f"Original text: {complex_text}")
        print("\nSimplifying...")
        
        simplified = simplify_text(complex_text, model, tokenizer)
        
        print(f"\nSimplified text: {simplified}")
        print("\n" + "=" * 50)
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    test_simplifier() 