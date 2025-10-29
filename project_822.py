from datasets import load_dataset
import pandas as pd

# Load the dataset
dataset = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")

# Access the 'by_polishing' subset as an example
by_polishing = dataset["by_polishing"]

# Convert to pandas DataFrame for easier exploration
df = pd.DataFrame(by_polishing)

# Print first 5 entries
print(df.head())

print("\nDatasets info for 'by_polishing':")
print(dataset['by_polishing'])

split = dataset["by_polishing"]

# View one sample
print(split[0])
num_human = len(split["original_abstract"])
# Count human-written abstracts
num_human = len(split["original_abstract"])

# Count AI-generated abstracts (4 per row)
num_ai = len(split["allam_generated_abstract"]) \
       + len(split["jais_generated_abstract"]) \
       + len(split["llama_generated_abstract"]) \
       + len(split["openai_generated_abstract"])

print("Number of human abstracts:", num_human)
print("Number of AI-generated abstracts:", num_ai)

# Distribution ratio
total = num_human + num_ai
print("Human %:", round(num_human / total * 100, 2))
print("AI %:", round(num_ai / total * 100, 2))
dataset = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")

import pandas as pd
# Convert to pandas for easier checks
df = pd.DataFrame(split)

# 1. Missing values
print("Missing values per column:")
print(df.isnull().sum())
print("_________________________________________")

# 2. Duplicates
print("\nNumber of duplicate rows:", df.duplicated().sum())

# Also check duplicates in each column separately
for col in df.columns:
    print(f"Duplicates in column {col}: {df[col].duplicated().sum()}")
print("_________________________________________")


# 3. Inconsistencies: empty strings or only spaces
for col in df.columns:
    empty_count = df[col].apply(lambda x: str(x).strip() == "").sum()
    print(f"Empty/blank values in column {col}: {empty_count}")

split2 = dataset["from_title"]

# Count human-written abstracts
num_human = len(split2["original_abstract"])

# Count AI-generated abstracts (4 per row)
num_ai = len(split2["allam_generated_abstract"]) \
       + len(split2["jais_generated_abstract"]) \
       + len(split2["llama_generated_abstract"]) \
       + len(split2["openai_generated_abstract"])

print("Number of human abstracts:", num_human)
print("Number of AI-generated abstracts:", num_ai)

# Distribution ratio
total = num_human + num_ai
print("Human %:", round(num_human / total * 100, 2))
print("AI %:", round(num_ai / total * 100, 2))

import pandas as pd
# Convert to pandas for easier checks
df = pd.DataFrame(split2)

# 1. Missing values
print("Missing values per column:")
print(df.isnull().sum())
print("_________________________________________")

# 2. Duplicates
print("\nNumber of duplicate rows:", df.duplicated().sum())

# Also check duplicates in each column separately
for col in df.columns:
    print(f"Duplicates in column {col}: {df[col].duplicated().sum()}")
print("_________________________________________")


# 3. Inconsistencies: empty strings or only spaces
for col in df.columns:
    empty_count = df[col].apply(lambda x: str(x).strip() == "").sum()
    print(f"Empty/blank values in column {col}: {empty_count}")

    split3 = dataset["from_title_and_content"]

# Count human-written abstracts
num_human = len(split3["original_abstract"])

# Count AI-generated abstracts (4 per row)
num_ai = len(split3["allam_generated_abstract"]) \
       + len(split3["jais_generated_abstract"]) \
       + len(split3["llama_generated_abstract"]) \
       + len(split3["openai_generated_abstract"])

print("Number of human abstracts:", num_human)
print("Number of AI-generated abstracts:", num_ai)

# Distribution ratio
total = num_human + num_ai
print("Human %:", round(num_human / total * 100, 2))
print("AI %:", round(num_ai / total * 100, 2))

#  2.1: 
#pip install nltk
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from datasets import load_dataset

# Download required NLTK resources
nltk.download('stopwords')
#test features
print(df.head())

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("[^؀-ۿ ]+", " ", text)  # remove non-Arabic chars
    return text
    #2.1.1 Normalization


# 2.1.2 aiming to remove altashkeel
def remove_diacritics(text):
    arabic_diacritics = re.compile('[\u0617-\u061A\u064B-\u0652]')
    return re.sub(arabic_diacritics, '', text)

#2.1.3 & 2.1.4
rabic_stopwords = set(stopwords.words("arabic"))
stemmer = ISRIStemmer()

def preprocess_text(text):
    text = str(text)
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in arabic_stopwords]
    tokens = [Stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

print(df.columns)

# to overcome a struggle I had to import this library
from nltk.corpus import stopwords
arabic_stopwords = set(stopwords.words("arabic"))

#enhanced the solution via this line
arabic_stopwords = {
    "في", "من", "على", "عن", "إلى", "و", "كما", "أن", "إن", "ما", "هو", "هي","الذي","هذا", "ذلك"
}

def preprocess_text(text):
    text = str(text)
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in arabic_stopwords]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)


text_columns = [
    'original_abstract',
    'allam_generated_abstract',
    'jais_generated_abstract',
    'llama_generated_abstract',
    'openai_generated_abstract'
]

for col in text_columns:
    if col in df.columns:
        clean_col = col + "_clean"
        df[clean_col] = df[col].apply(preprocess_text)
    else:
        print(f"⚠️ Column '{col}' not found in DataFrame!")

print("✅ Preprocessing complete! Here are the new columns:")
print(df.columns)
df.head(2)

text_columns = [
    'original_abstract',
    'allam_generated_abstract',
    'jais_generated_abstract',
    'llama_generated_abstract',
    'openai_generated_abstract'
]
for col in text_columns:
    clean_col = col + "_clean"
    df[clean_col] = df[col].apply(preprocess_text)
print(" Preprocessing complete! Here are the new columns:")
print(df.columns)
df.head(2)

# 2.2
#pip install matpolt lib
#pip install wordcloud
#pip install seaborn
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
import numpy as np
#pip install scikit-learn

ai_texts = pd.concat([
    df['allam_generated_abstract_clean'],
    df['jais_generated_abstract_clean'],
    df['llama_generated_abstract_clean'],
    df['openai_generated_abstract_clean']
], axis=0).dropna().tolist()

human_texts = df['original_abstract_clean'].dropna().tolist()

def text_stats(texts):
    words = [w for txt in texts for w in txt.split()]
    avg_word_len = np.mean([len(w) for w in words])
    avg_sent_len = np.mean([len(txt.split()) for txt in texts])
    vocab = set(words)
    ttr = len(vocab) / len(words)
    return avg_word_len, avg_sent_len, ttr

stats_human = text_stats(human_texts)
stats_ai = text_stats(ai_texts)

print("\n Statistical Summary:")
print(f"Human-written: Avg word len={stats_human[0]:.2f}, Avg sent len={stats_human[1]:.2f}, TTR={stats_human[2]:.3f}")
print(f"AI-generated : Avg word len={stats_ai[0]:.2f}, Avg sent len={stats_ai[1]:.2f}, TTR={stats_ai[2]:.3f}")

def plot_top_ngrams(texts, n=2, top_k=15):
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(ngram_range=(n, n))
    bag = vec.fit_transform(texts)
    sum_words = bag.sum(axis=0)
    freqs = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    freqs = sorted(freqs, key=lambda x: x[1], reverse=True)[:top_k]
    words, counts = zip(*freqs)
    plt.figure(figsize=(10,4))
    sns.barplot(x=list(counts), y=list(words))
    plt.title(f"Top {top_k} {n}-grams – {n}-grams for {'Human' if texts==human_texts else 'AI'} abstracts")
    plt.show()

print("\n🔤 Top Bigrams for Human-written abstracts:")
plot_top_ngrams(human_texts, n=2)

print("\n🔤 Top Bigrams for AI-generated abstracts:")
plot_top_ngrams(ai_texts, n=2)

#here i will try to show tSentence Length Distribution

#Purpose: to Compare the length of the abstracts  (in words or characters).
# bc AI-generated text might be longer, more repetitive, or more uniform than human-written text.

import matplotlib.pyplot as plt

df["human_length"] = df["original_abstract"].apply(lambda x: len(x.split()))
df["ai_length"] = df["openai_generated_abstract"].apply(lambda x: len(x.split()))

plt.figure(figsize=(8,5))
plt.hist(df["human_length"], bins=30, alpha=0.6, label="Human-written", color='blue')
plt.hist(df["ai_length"], bins=30, alpha=0.6, label="AI-generated", color='orange')
plt.xlabel("Sentence Length (words)")
plt.ylabel("Frequency")
plt.title("Sentence Length Distribution")
plt.legend()
plt.show()

# TTP=unique words/total words
#to show comaring  lexical diversity between AI-generated vs HUMAN-written vocabulary

def type_token_ratio(text):
    words = text.split()
    return len(set(words)) / len(words) if words else 0

df["human_ttr"] = df["original_abstract"].apply(type_token_ratio)
df["ai_ttr"] = df["openai_generated_abstract"].apply(type_token_ratio)

plt.figure(figsize=(6,5))
plt.boxplot([df["human_ttr"], df["ai_ttr"]], labels=["Human", "AI"])
plt.title("Vocabulary Richness (Type–Token Ratio)")
plt.ylabel("TTR Score")
plt.show()


#next to show which words are overused by AI vs humans.
from collections import Counter
import pandas as pd

human_words = " ".join(df["original_abstract"]).split()
ai_words = " ".join(df["openai_generated_abstract"]).split()

human_freq = Counter(human_words)
ai_freq = Counter(ai_words)

common_words = set(list(human_freq.keys())[:100]) & set(list(ai_freq.keys())[:100])

data = []
for w in common_words:
    data.append((w, human_freq[w], ai_freq[w]))

freq_df = pd.DataFrame(data, columns=["word", "human", "ai"]).sort_values("human", ascending=False)[:15]

freq_df.plot(x="word", kind="bar", figsize=(10,5), title="Top Words: Human vs AI", rot=45)
plt.ylabel("Frequency")
plt.show()

#area
freq_df.plot(x="word", kind="area", figsize=(10,5), title="Top Words: Human vs AI", rot=45, alpha=0.6)

#line
freq_df.plot(x="word", kind="line", figsize=(10,5), title="Top Words: Human vs AI", rot=45)



