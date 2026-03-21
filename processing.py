import re
import spacy
from math import ceil
from heapq import nlargest
text = "run Run lmao hello , hi my name is manish how are you lol "
with open("Trans.txt", "r+") as f:
    text = f.read()
f.close()
nlp = spacy.load("en_core_web_md")


def cleantext(text):
    text = re.sub(r"\[.*?\]", r"", text)
    text = re.sub(r"<.*?>", r"", text)
    text = re.sub(r"([!?.,])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokeniser(text):
    tk = []
    doc = nlp(text)
    for token in doc:
        if token.pos_ == "INTJ":
            continue
        if token.dep_ == "discourse":
            continue
        if token.is_stop:
            continue
        if token.is_punct:
            continue
        tk.append(token.lemma_.lower())
    return tk


def wordFreq(tokens):
    word_frequencies = {}
    for tk in tokens:
        if tk not in word_frequencies:
            word_frequencies[tk] = 1
        else:
            word_frequencies[tk] += 1
    maxf = max(word_frequencies.values())

    for val in word_frequencies.keys():
        word_frequencies[val] = word_frequencies[val] / maxf
    return word_frequencies


def sentence_tokeniser(text):
    st = []
    doc = nlp(text)
    for sent in doc.sents:
        st.append(sent)
    return st


def sent_freq(sentences, word_frequency):
    sentence_scores = {}

    for sent in sentences:
        if len(sent.text.split()) < 6:
            continue

        for word in sent:
            if word.is_stop or word.is_punct or word.pos_ == "INTJ" or word.dep_ == "discourse":
                continue

            key = word.lemma_.lower()

            if key in word_frequency:
                if sent not in sentence_scores:
                    sentence_scores[sent] = word_frequency[key]
                else:
                    sentence_scores[sent] += word_frequency[key]

    return sentence_scores
cleaned_text = cleantext(text=text)

word_tokens = tokeniser(cleaned_text)
word_frequency = wordFreq(word_tokens)
sentence_tokens = sentence_tokeniser(cleaned_text)
sentence_scores = sent_freq(sentence_tokens , word_frequency)

select_len = ceil(len(sentence_tokens)*0.1) 

summary = nlargest(select_len , sentence_scores , key = sentence_scores.get)

print(summary)

