import re
import spacy
from math import ceil
from collections import Counter
from math import log

NER_THRESHOLD = 2
ACCEPTABLE_SENTENCE_LEN = 10
SUMMARY_LEN = 0.20
MAX_SUMMARY_SENTENCES = 3
REDUNDANCY_WEIGHT = 0.35
SIMILARITY_THRESHOLD = 0.80
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
    if not tokens or not sentences:
        return {}

    tf = Counter(tokens)
    max_tf = max(tf.values())

    sentence_count = len(sentences)
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
        idf_score = log((1 + sentence_count) / (1 + doc_freq[word])) + 1
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


def _sentence_terms(sentence):
    return {
        token.lemma_.lower()
        for token in sentence
        if not token.is_stop and not token.is_punct
    }


def _overlap(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 0


def select_summary(sentences, sentence_scores):
    if not sentence_scores:
        return []

    select_len = min(
        MAX_SUMMARY_SENTENCES,
        max(1, ceil(len(sentences) * SUMMARY_LEN)),
    )
    candidates = list(sentence_scores)
    selected = [max(candidates, key=sentence_scores.get)]
    candidates.remove(selected[0])
    max_score = sentence_scores[selected[0]]
    terms = {sentence: _sentence_terms(sentence) for sentence in sentence_scores}
    candidates = [
        sentence for sentence in candidates
        if _overlap(terms[sentence], terms[selected[0]]) < SIMILARITY_THRESHOLD
    ]

    while candidates and len(selected) < select_len:
        def mmr_score(sentence):
            relevance = sentence_scores[sentence] / max_score
            redundancy = max(_overlap(terms[sentence], terms[item]) for item in selected)
            return relevance - REDUNDANCY_WEIGHT * redundancy

        chosen = max(candidates, key=mmr_score)
        selected.append(chosen)
        candidates.remove(chosen)
        candidates = [
            sentence for sentence in candidates
            if _overlap(terms[sentence], terms[chosen]) < SIMILARITY_THRESHOLD
        ]

    return sorted(selected, key=lambda sentence: sentence.start)


def summarise_extractive(content):
    cleaned_text = cleantext(content)
    if not cleaned_text:
        return ""

    tokens, sentences = process_text(cleaned_text)
    word_frequency = wordFreq(tokens, sentences)
    sentence_scores = sent_score(sentences, word_frequency)
    summary = select_summary(sentences, sentence_scores)

    if not summary:
        return cleaned_text

    return " ".join(sentence.text.strip() for sentence in summary)
