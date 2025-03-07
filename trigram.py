import wikipedia
from nltk import trigrams
from nltk.tokenize import word_tokenize
from collections import Counter
import string
import math

# 1. Use the Wikipedia python module and access any topic, as you will use that as your corpus, with a word limit of 1000 words. We chose Decision Problem.
# Fetch from Wikipedia page
# Fetch Wikipedia page
wikipedia_page = wikipedia.page("Decision problem")

# Extract first 1000 words
training_text = ' '.join(wikipedia_page.content.split()[:1000])  # Limit to 1000 words

# Display extracted text
print("Extracted Training Text (First 1000 Words):\n")
print(training_text)

# 2. Train a Trigram Model
# Tokenization - Converts text to lowercase, tokenizes words, and removes punctuation
tokennized_words = [word for word in word_tokenize(training_text.lower()) if word.isalnum()]
print("Tokens:", tokennized_words, "\n")

# Counts of Unigrams and Trigrams
unigram_frequency = Counter(tokennized_words) 
trigram_frequency = Counter(trigrams(tokennized_words))

# Show Unigram Counts
print("Unigram Counts:")
for unigram, count in unigram_frequency.items():
    print(f"{unigram}: {count}")

# Show Trigram Counts
print("\nTrigram Counts:")
for trigram, count in trigram_frequency.items():
    print(f"{trigram}: {count}")

# Computes trigram probabilities
print("\nTrigram Probabilities:")
def compute_trigram_probabilities(tokennized_words):
    bigram_frequency = Counter(zip(tokennized_words[:-1], tokennized_words[1:]))
    trigram_probability = {trigram: count / bigram_frequency[(trigram[0], trigram[1])] 
                     for trigram, count in trigram_frequency.items() if (trigram[0], trigram[1]) in bigram_frequency}
    return trigram_probability

# Train the trigram model
trigram_probability = compute_trigram_probabilities(tokennized_words)

# Show Trigram Probabilities
for trigram, prob in trigram_probability.items():
    print(f"P({trigram[2]} | {trigram[0]}, {trigram[1]}) = {prob:.4f}")

# Predict next word
def predict_next_word(trigram_probability, first_word, second_word):
    candidates = {k[2]: v for k, v in trigram_probability.items() if k[0] == first_word and k[1] == second_word}
    return max(candidates, key=candidates.get) if candidates else None

# Predict Test word
predicted_next_word = predict_next_word(trigram_probability, "decision", "problem")
print(f"\nPredicted next word after 'decision problem': {predicted_next_word}")

# Generate text
def generate_trigram_text(trigram_model, start_words, length=10):
    first_word, second_word = start_words.lower().split()
    generated_text = [first_word, second_word]

    for _ in range(length):
        candidates = {k[2]: v for k, v in trigram_model.items() if k[0] == first_word and k[1] == second_word}
        if not candidates:
            break  # Stop if no valid next word is found
        next_word = max(candidates, key=candidates.get)
        generated_text.append(next_word)
        first_word, second_word = second_word, next_word  # Move to the next pair
    
    return " ".join(generated_text)

# testing generated text
print("\nGenerated Text:", generate_trigram_text(trigram_probability, "decision problem", 10))

# Evaluate The Test Sentence
test_sentence = "decision problem is a concept related to algorithms"
test_tokenized_words = [word for word in word_tokenize(test_sentence.lower()) if word.isalnum()]

print("\nEvaluation on Test Sentence:")
print("Test Sentence:", test_sentence)

predicted_words = []
for i in range(len(test_tokenized_words) - 2):  # Iterate through word triplets
    first_word = test_tokenized_words[i]
    second_word = test_tokenized_words[i + 1]
    actual_next_word = test_tokenized_words[i + 2]
    predicted_next_word = predict_next_word(trigram_probability, first_word, second_word)

    predicted_words.append(predicted_next_word)
    print(f"Given '{first_word} {second_word}', Actual: '{actual_next_word}', Predicted: '{predicted_next_word}'")

# Calculate accuracy of predictions
correct_predictions = sum(1 for i in range(len(predicted_words)) if predicted_words[i] == test_tokenized_words[i + 2])
accuracy = correct_predictions / (len(test_tokenized_words) - 2) * 100
print(f"\nPrediction Accuracy: {accuracy:.2f}%")


# 3 Using a test sentence “The quick brown fox jumps over the lazy dog near the bank of the river.” OR generate your own test sentence, create a function that will determine the perplexity score for each trained model.
# b.Trigram model perplexity -> Test Sentence “” -> Score:

# Function to compute trigram perplexity
def compute_trigram_perplexity(trigram_probability, test_sentence):
    tokennized_words = [word for word in word_tokenize(test_sentence.lower()) if word.isalnum()]
    N = len(tokennized_words)
    log_prob_sum = 0

    for i in range(2, N):
        word1, word2, word3 = tokennized_words[i - 2], tokennized_words[i - 1], tokennized_words[i]
        prob = trigram_probability.get((word1, word2, word3), 1e-10)  # Avoid zero probability
        log_prob_sum += math.log2(prob)

    perplexity = 2 ** (-log_prob_sum / N)
    return perplexity

# Test Sentences
test_sentences = [
    "The quick brown fox jumps over the lazy dog near the bank of the river.",
    "In computability theory and computational complexity theory, a decision problem is a computational problem that can be posed as a yes-no question based on the given input values.",
    "A decision problem which can be solved by an algorithm is called decidable."
]

# Compute and Shows the Perplexity Scores
for i, sentence in enumerate(test_sentences, 1):
    perplexity = compute_trigram_perplexity(trigram_probability, sentence)
    print(f"\nTrigram Model Perplexity (Sentence: {sentence}):  {perplexity:.4f}")