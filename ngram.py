import nltk

nltk.download("wordnet")

from nltk.stem import WordNetLemmatizer
from collections import Counter, defaultdict
import math
import copy
import random
import operator


import re


def flatten(lst):
    return [item for sublist in lst for item in sublist]


import re
from nltk.stem import WordNetLemmatizer


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def prepare_data(filename):
    lemmatizer = WordNetLemmatizer()

    # Read the entire file as a single string and remove newlines
    with open(filename, "r", encoding="utf-8") as f:
        text = (
            f.read()
            .replace("\n", " ")
            .replace("-", "")
            .replace("...", "")
            .replace('"', "")
            .lower()
        )

    # Split text into phrases by punctuation
    phrases = re.split(r'[¿?!.;\-"\(\)\[\]]+', text)

    # Remove empty strings and strip extra spaces
    phrases = [phrase.strip() for phrase in phrases if phrase.strip()]

    # Split phrases into words and lemmatize each phrase
    data = [
        [lemmatizer.lemmatize(word) for word in phrase.split()] + ["</s>"]
        for phrase in phrases
    ]

    # Flatten to create the corpus
    corpus = flatten(data)
    vocab = set(corpus)

    for i, sublist in enumerate(data):
        data[i] = [word if word != "wa" else "was" for word in sublist]

    return vocab, data


class NGramLM:
    def __init__(self, N: int, smoothing_k: int = 0):
        self.N = N
        self.vocab = set()
        self.prob = defaultdict(Counter)
        self.counts = defaultdict(Counter)
        self.smoothing_k = smoothing_k

    def train(self, vocab, data):
        if self.N == 1:
            self.counts = Counter(flatten(data))
            for word in self.counts:
                self.prob[word] = self.counts[word] / (self.counts.total())
        else:
            self.vocab = vocab
            self.vocab.add("<s>")
            for sentence in data:
                aux_sent = ["<s>"] * (self.N - 1) + sentence
                for idx in range(self.N - 1, len(aux_sent)):
                    word = aux_sent[idx]
                    context = tuple(aux_sent[idx - self.N + 1 : idx])
                    self.counts[context][word] += 1

            # AFUERA DEL FOR SENTENCE IN DATA
            for context, word_counts in self.counts.items():
                total_count = sum(word_counts.values())
                for word, count in word_counts.items():
                    self.prob[context][word] = count / total_count
        return self.counts, self.prob

    def get_ngram_logprob(self, word, seq_len, context=""):
        if self.N == 1 and word in self.prob.keys():
            return math.log(self.prob[word]) / seq_len
        elif self.N > 1 and not self._is_unseen_ngram(context, word):
            return math.log(self.prob[context][word]) / seq_len
        else:
            # assign a small probability to the unseen ngram
            # to avoid log of zero and to penalise unseen word or context
            return math.log(1 / len(self.vocab)) / seq_len

    def get_ngram_prob(self, word, context=""):
        if self.N == 1 and word in self.prob.keys():
            return self.prob[word]
        elif self.N > 1 and not self._is_unseen_ngram(context, word):
            return self.prob[context][word]
        elif word in self.vocab and self.smoothing_k > 0:
            # probability assigned by smoothing
            return self.smoothing_k / (
                sum(self.counts[context].values()) + self.smoothing_k * len(self.vocab)
            )
        else:
            # unseen word or context
            return 0

    # In this method, the perplexity is measured at the sentence-level, averaging over all sentences.
    # Actually, it is also possible to calculate perplexity by merging all sentences into a long one.
    def perplexity(self, test_data):
        log_ppl = 0
        if self.N == 1:
            for sentence in test_data:
                for word in sentence:
                    log_ppl += self.get_ngram_logprob(word=word, seq_len=len(sentence))
        else:
            for sentence in test_data:
                for i in range(len(sentence) - self.N + 1):
                    context = sentence[i : i + self.N - 1]
                    context = " ".join(context)
                    word = sentence[i + self.N - 1]
                    log_ppl += self.get_ngram_logprob(
                        context=context, word=word, seq_len=len(sentence)
                    )

        log_ppl /= len(test_data)
        ppl = math.exp(-log_ppl)
        return ppl

    def _is_unseen_ngram(self, context, word):
        if context not in self.prob.keys() or word not in self.prob[context].keys():
            return True
        else:
            return False

    # generate the most probable k words
    def generate_next(self, context, k):
        context = (self.N - 1) * "<s> " + context
        context = context.split()
        # ngram_context_list = context[-self.N+1:] #CAMBIADO
        ngram_context_list = context[-(self.N - 1) :]
        # ngram_context = " ".join(ngram_context_list)
        ngram_context = tuple(ngram_context_list)  # CAMBIADO

        # if ngram_context in self.prob.keys():
        if ngram_context in self.prob:  # CAMVIADO
            candidates = self.prob[ngram_context]
            most_probable_words = sorted(
                candidates.items(), key=lambda kv: kv[1], reverse=True
            )
            for i in range(min(k, len(most_probable_words))):
                print(
                    " ".join(context[self.N - 1 :])
                    + " "
                    + most_probable_words[i][0]
                    + "\t P={}".format(most_probable_words[i][1])
                )
        else:
            print(f"Contexto no visto: {ngram_context}")

        # Generate the next n words with greedy search

    def generate_next_n(self, context, n):
        # Prepend <s> tokens to match the N-gram size
        context = (self.N - 1) * "<s> " + context
        context = context.split()
        ngram_context_list = context[-(self.N - 1) :]
        ngram_context = tuple(ngram_context_list)

        for _ in range(n):
            try:
                # Check if current context exists in the probability dictionary
                if ngram_context in self.prob:
                    candidates = self.prob[ngram_context]
                    # Select the most likely next word
                    most_likely_next = max(
                        candidates.items(), key=operator.itemgetter(1)
                    )[0]

                    # Add the chosen word to the context
                    context.append(most_likely_next)
                    ngram_context_list = ngram_context_list[1:] + [most_likely_next]
                    ngram_context = tuple(ngram_context_list)
                else:
                    # If context is unseen, break or skip prediction
                    break
            except Exception as e:
                print(f"Error during generation: {e}")
                break
        return " ".join(context[self.N - 1 :])
        # print(" ".join(context[self.N - 1 :]))
