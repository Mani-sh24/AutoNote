import re
import spacy
from math import ceil
from heapq import nlargest
from collections import Counter
from math import log
NER_THRESHOLD = 2
ACCEPTABLE_SENTENCE_LEN = 10
SUMMARY_LEN = 0.25
NOUN_CHUNK_WEIGHT = 0.46
GARBAGE_TYPES = {
    "DATE",      # "past", "year", "months", "quarterly" — too generic
    "TIME",      # "morning", "afternoon" — not informative
    "CARDINAL",  # plain numbers like "99", "1.5", "ten"
    "ORDINAL",   # "first", "second", "third"
    "QUANTITY",  # "half a point", "18 months"
    "PERCENT",   # "99%"
    "MONEY",     # "$5 million"
    "LANGUAGE",  # "English"
}
nlp = spacy.load("en_core_web_md")
# with open("summary.txt" , "r") as f:
#     text = f.read()
# f.close()
def cleantext(text):
    fillers = r'\b(uh|um|like|basically|kind of|sort of|you know|i mean)\b'
    text = re.sub(r"\[.*?\]", r"", text)
    text = re.sub(r"<.*?>", r"", text)
    text = re.sub(r"\b(\w+(?:'\w+)?)(,\s*\1)+", r'\1', text)
    text = re.sub(fillers, '', text, flags=re.IGNORECASE)
    text = re.sub(r"([!?.,])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def process_text(text):
    doc = nlp(text)
    tokens = []
    sentences = []
    for sent in doc.sents:
        words = sent.text.split()

        # length check
        if len(words) < ACCEPTABLE_SENTENCE_LEN:
            continue

        punct_count = sum(1 for char in sent.text if char in '.,!?;:')
        punct_ratio = punct_count / len(sent.text)
        if punct_ratio > 0.10:
            continue

        short_words = sum(1 for word in words if len(word) <= 2)
        short_ratio = short_words / len(words)
        if short_ratio > 0.40:
            continue
        sentences.append(sent)
        for token in sent:
            if token.is_stop or token.is_punct:
                continue
            if token.pos_ == "INTJ" or token.dep_ == "discourse":
                continue
            tokens.append(token.lemma_.lower())

    return tokens, sentences
def wordFreq(tokens, sentences):
    tf = Counter(tokens)
    max_tf = max(tf.values())

    N = len(sentences)
    doc_freq = Counter()
    for sent in sentences:
        seen = set()
        for token in sent:
            if token.is_stop or token.is_punct:
                continue
            if token.pos_ == "INTJ" or token.dep_ == "discourse":
                continue
            key = token.lemma_.lower()
            if key not in seen:
                doc_freq[key] += 1
                seen.add(key)

    tfidf = {}
    for word, freq in tf.items():
        tf_score = freq / max_tf
        idf_score = log(N / (1 + doc_freq[word]))
        tfidf[word] = tf_score * idf_score

    return tfidf


def sent_score(sentences, word_frequency):
    sentence_scores = {}    

    for sent in sentences:
        words = sent.text.split()
        if len(sent.text.split()) < ACCEPTABLE_SENTENCE_LEN:
            continue
        score = 0

        for token in sent:
            if token.is_stop or token.is_punct:
                continue
            if token.pos_ == "INTJ" or token.dep_ == "discourse":
                continue
            key = token.lemma_.lower()
            tfidfscore = word_frequency.get(key, 0)
            
            if token.ent_type_ and token.ent_type_ not in GARBAGE_TYPES:
                # print(token , token.ent_type_)
                tfidfscore *= NER_THRESHOLD

            score += tfidfscore
        
        noun_chunks = list(sent.noun_chunks)
        noun_chunk_density = len(noun_chunks) / len(words)
        score += noun_chunk_density * NOUN_CHUNK_WEIGHT
        if score > 0:
            sentence_scores[sent] = score

    return sentence_scores


# # ---- MAIN FLOW ----
# cleaned_text = cleantext(text)
# tokens, sentences = process_text(cleaned_text)
# word_frequency = wordFreq(tokens, sentences)
# sentence_scores = sent_score(sentences, word_frequency)
# select_len = max(ceil(len(sentences) * SUMMARY_LEN), 5)
# summary = nlargest(select_len, sentence_scores, key=sentence_scores.get)
# res = ""
# for sent in sorted(summary, key=lambda s: s.start):
#     res += sent.text + " "

# with open("Trans.txt", "w+") as f:
#     f.write(res.strip())

# print("done")