import wikipedia
from nltk import bigrams
from nltk.tokenize import word_tokenize
from collections import Counter
import string
import math

# 1. Use the Wikipedia python module and access any topic, as you will use that as your corpus, with a word limit of 1000 words. We chose Decision Problem.
# Fetch from Wikipedia page
wikipedia_page = wikipedia.page("Decision problem")
training_text = wikipedia_page.content[:1000]  # Limit to 1000 characters
print("Extracted text:", training_text, "\n")

# 2. Train a Bigram Model
# Tokenization - Converts text to lowercase, tokenizes words, and removes punctuation
tokennized_words = [word for word in word_tokenize(training_text.lower()) if word.isalnum()]
print("Tokens:", tokennized_words, "\n")

# Counts of Unigrams and Bigrams
unigram_frequency = Counter(tokennized_words) 
bigram_frequency = Counter(bigrams(tokennized_words)) 

# Show Unigram Counts
print("Unigram Counts:")
for unigram, count in unigram_frequency.items():
    print(f"{unigram}: {count}")

# Show Bigram Counts
print("\nBigram Counts:")
for bigram, count in bigram_frequency.items():
    print(f"{bigram}: {count}")

# Computes bigram probabilities
print("\nBigram Probabilities:")
def compute_bigram_probabilities(tokennized_words):
    """
    Computes the probability of each bigram occurring based on unigram frequency.
    P(word2 | word1) = Count(word1, word2) / Count(word1)
    """
    bigram_probability = {bigram: count / unigram_frequency[bigram[0]] 
                    for bigram, count in bigram_frequency.items()}
    return bigram_probability

# Train the bigram model
bigram_probability = compute_bigram_probabilities(tokennized_words)

# Show Bigram Probabilities
for bigram, prob in bigram_probability.items():
    print(f"P({bigram[1]} | {bigram[0]}) = {prob:.4f}")

# Predict next word
def predict_next_word(bigram_probability, current_word):
    """
    Predicts the most probable next word based on the given current word.
    """
    candidates = {k[1]: v for k, v in bigram_probability.items() if k[0] == current_word}
    return max(candidates, key=candidates.get) if candidates else None

# Predict Test word
predicted_word = predict_next_word(bigram_probability, "problem")
print(f"\nPredicted next word after 'problem': {predicted_word}")

# Generate text
def generate_bigram_text(bigram_model, start_word, length=10):
    """
    Generates a sequence of words based on the bigram model, starting from a given word.
    """
    current_word = start_word.lower()
    generated_text = [current_word]

    for _ in range(length):
        candidates = {k[1]: v for k, v in bigram_model.items() if k[0] == current_word}
        if not candidates:
            break  # Stop if no valid next word is found
        current_word = max(candidates, key=candidates.get)
        generated_text.append(current_word)
    
    return " ".join(generated_text)

# testing generated text
print("\nGenerated Text:", generate_bigram_text(bigram_probability, "decision", 10))

# Evaluate The Test Sentence
test_sentence = "a decision problem is a computational problem"
test_tokenized_words = [word for word in word_tokenize(test_sentence.lower()) if word.isalnum()]

print("\nEvaluation on Test Sentence:")
print("Test Sentence:", test_sentence)

predicted_words = []
for i in range(len(test_tokenized_words) - 1):  # Iterate through word pairs
    current_word = test_tokenized_words[i]
    actual_next_word = test_tokenized_words[i + 1]
    predicted_next_word = predict_next_word(bigram_probability, current_word)

    predicted_words.append(predicted_next_word)
    print(f"Given '{current_word}', Actual: '{actual_next_word}', Predicted: '{predicted_next_word}'")

# Calculate accuracy of predictions
correct_predictions = sum(1 for i in range(len(predicted_words)) if predicted_words[i] == test_tokenized_words[i + 1])
accuracy = correct_predictions / (len(test_tokenized_words) - 1) * 100
print(f"\nPrediction Accuracy: {accuracy:.2f}%")

# 3. Using a test sentence “The quick brown fox jumps over the lazy dog near the bank of the river.”  OR generate your own test sentence, create a function that will determine the perplexity score for each trained model.
# a.Bigram model perplexity -> Test Sentence “” -> Score:

# Computes bigram perplexity
def compute_bigram_perplexity(bigram_probability, test_sentence):
    tokennized_words = [word for word in word_tokenize(test_sentence.lower()) if word.isalnum()]
    N = len(tokennized_words)
    log_prob_sum = 0

    for i in range(1, N):
        word1, word2 = tokennized_words[i - 1], tokennized_words[i]
        prob = bigram_probability.get((word1, word2), 1e-10)  # Avoid zero probability
        log_prob_sum += math.log2(prob)

    perplexity = 2 ** (-log_prob_sum / N)
    return perplexity

# Test Sentences
test_sentences = [
    "The quick brown fox jumps over the lazy dog near the bank of the river.",
    "In computability theory and computational complexity theory, a decision problem is a computational problem that can be posed as a yes-no question based on the given input values.",
    "A decision problem which can be solved by an algorithm is called decidable."
]

# Computes and Shows the Perplexity Scores
for i, sentence in enumerate(test_sentences, 1):
    perplexity = compute_bigram_perplexity(bigram_probability, sentence)
    print(f"\nBigram Model Perplexity (Sentence: {sentence}): {perplexity:.4f}")

