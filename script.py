import pandas as pd
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import PorterStemmer
import string 
import spacy
from textblob import TextBlob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from matplotlib.patches import Patch
from wordcloud import WordCloud
from PIL import Image

#Importation des données dans source_data
source_data = pd.read_csv('source_data/inaug_speeches.csv', encoding='latin1')

# Normalisation de la colonne 'Date'
def normalize_date(date_str):
    formats = [
        '%A, %B %d, %Y',  # "Thursday, April 30, 1789"
        '%B %d, %Y',       # "January 20, 1997"
        '%d/%m/%Y',        # "30/04/1789"
        '%Y-%m-%d',        # "1789-04-30"
    ]
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt).strftime('%A, %B %d, %Y')
        except:
            continue
    return None

source_data['Date'] = source_data['Date'].apply(normalize_date)


def preprocess_text(text):

    # Tokenisation
    tokens = wordpunct_tokenize(text.lower())

    # Suppression des stop words
    nltk.download('stopwords')
    stop_words = nltk.corpus.stopwords.words('english')

    tokens = [word for word in tokens if word not in stop_words]

    # Suppression de la ponctuation
    tokens = [word for word in tokens if word not in string.punctuation]

    # Suppression des tokens non alphabétiques (chiffres, symboles)
    tokens = [word for word in tokens if word.isalpha()]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

source_data['cleaned_text'] = source_data['text'].apply(preprocess_text)
def analyze_text(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    
    return doc

# Exemple d'analyse de texte
sample_text = source_data['cleaned_text'].iloc[0]
analyze_text = analyze_text(sample_text)

def analyze_sentiment(text):
    """
    Analyse de sentiment avec TextBlob et VADER
    Retourne un dictionnaire avec les scores
    """
    # Initialisation des analyseurs
    sentiment = TextBlob(text).sentiment

    # Analyse avec TextBlob
    blob_sentiment = {
        'polarity': sentiment.polarity,
        'subjectivity': sentiment.subjectivity
    }

    return {
        'textblob': blob_sentiment
    }

# Exemple d'analyse de sentiment de chaque discours
for index, row in source_data.iterrows():
    sentiment_scores = analyze_sentiment(row['cleaned_text'])

#Calcul des mots les plus fréquents
from collections import Counter
all_tokens = []
for text in source_data['cleaned_text']:
    all_tokens.extend(text.split())
word_freq = Counter(all_tokens)
most_common = word_freq.most_common(20)

#Tf-IDF
def compute_tfidf(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    words = vectorizer.get_feature_names_out()
    tfidf_matrix = X.toarray()

    return tfidf_matrix, words

# Exemple de calcul du TF-IDF pour les discours
tfidf_matrix, feature_names = compute_tfidf(source_data['cleaned_text'])
df_tfidf = pd.DataFrame(tfidf_matrix, index=[f"Doc{i+1}" for i in range(len(source_data['cleaned_text']))], columns=feature_names)
avg_tfidf = np.mean(tfidf_matrix, axis=0)

#Visualisation du nuage de mots (format coeur)
img = Image.open("coeur.png")
img_rgba = img.convert("RGBA")
alpha = np.array(img_rgba)[:, :, 3]
if alpha.max() > 0:
    mask = alpha
else:
    mask = np.array(img_rgba.convert('L'))

if mask.mean() > 127:
    mask = 255 - mask
mask = np.where(mask > 128, 255, 0).astype(np.uint8)
mask = 255 - mask

plt.figure(figsize=(4, 4))
plt.imshow(mask, cmap='gray')
plt.axis('off')
plt.title('Masque en forme de cœur')
plt.show()

wordcloud = WordCloud(
    background_color='white',
    stopwords=nltk.corpus.stopwords.words('english'),
    max_words=50,
    mask=mask,
    contour_width=1,
    contour_color='firebrick'
).generate(' '.join(all_tokens))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Visualisation des scores TF-IDF tops
top_n = 20
top_indices = np.argsort(avg_tfidf)[-top_n:]
top_features = [feature_names[i] for i in top_indices]
top_scores = avg_tfidf[top_indices]

plt.figure(figsize=(10, 5))
plt.bar(top_features, top_scores, color='skyblue')
plt.xlabel("Mots")
plt.ylabel("Score TF-IDF moyen")
plt.title("Importance des mots selon TF-IDF")
plt.xticks(rotation=45)
plt.show()

# Dictionnaire des partis par président
party_dict = {
    'Washington': 'Autre',
    'Adams': 'Autre',
    'Jefferson': 'Autre',
    'Madison': 'Autre',
    'Monroe': 'Autre',
    'Jackson': 'Democrat',
    'Van Buren': 'Democrat',
    'Harrison': 'Autre',
    'Polk': 'Democrat',
    'Taylor': 'Autre',
    'Pierce': 'Democrat',
    'Buchanan': 'Democrat',
    'Lincoln': 'Republican',
    'Grant': 'Republican',
    'Hayes': 'Republican',
    'Garfield': 'Republican',
    'Cleveland': 'Democrat',
    'McKinley': 'Republican',
    'Roosevelt': 'Republican',
    'Taft': 'Republican',
    'Wilson': 'Democrat',
    'Harding': 'Republican',
    'Coolidge': 'Republican',
    'Hoover': 'Republican',
    'Truman': 'Democrat',
    'Eisenhower': 'Republican',
    'Kennedy': 'Democrat',
    'Johnson': 'Democrat',
    'Nixon': 'Republican',
    'Carter': 'Democrat',
    'Reagan': 'Republican',
    'Bush': 'Republican',
    'Clinton': 'Democrat',
    'Obama': 'Democrat',
    'Trump': 'Republican',
    'Biden': 'Democrat',
}
crisis_periods = [
    (1812, 1815, "Guerre de 1812"),
    (1861, 1865, "Guerre Civile"),
    (1917, 1918, "Première Guerre Mondiale"),
    (1929, 1939, "Grande Dépression"),
    (1941, 1945, "Seconde Guerre Mondiale"),
    (1950, 1953, "Guerre de Corée"),
    (1964, 1975, "Guerre du Vietnam"),
    (2001, 2003, "11 Septembre / Guerre en Afghanistan"),
    (2008, 2010, "Crise Financière"),
    (2020, 2021, "COVID-19"),
]
party_colors = {
    'Democrat': 'blue',
    'Republican': 'red',
    'Autre': 'grey',
}

def get_party(full_name):
    for key in party_dict:
        if key.lower() in full_name.lower():
            return party_dict[key]
    return 'Autre'

source_data['party'] = source_data['Name'].apply(get_party)

# Sentiment et année
source_data['sentiment'] = source_data['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
def get_tonality(polarity):
    if polarity > 0.05:
        return 'POSITIF'
    elif polarity < -0.05:
        return 'NÉGATIF'
    else:
        return 'NEUTRE'
source_data['tonality'] = source_data['sentiment'].apply(get_tonality)
source_data['year'] = source_data['Date'].apply(lambda x: pd.to_datetime(x, format='%A, %B %d, %Y').year)

sentiment_by_year = source_data.groupby('year')['sentiment'].mean()
party_by_year = source_data.groupby('year')['party'].first()

# Lissage
smoothed = uniform_filter1d(sentiment_by_year.values, size=5)

# Visualisation
plt.figure(figsize=(14, 6))

# Fond coloré selon le parti
for i in range(len(party_by_year) - 1):
    year_start = party_by_year.index[i]
    year_end = party_by_year.index[i + 1]
    party = party_by_year.iloc[i]
    color = party_colors.get(party, 'grey')
    plt.axvspan(year_start, year_end, alpha=0.15, color=color)

crisis_added = False
for (start, end, label) in crisis_periods:
    plt.axvspan(start, end, alpha=0.25, color='yellow',
                label='Crise / Guerre' if not crisis_added else "")
    # Annotation du nom de la crise
    plt.text((start + end) / 2, plt.ylim()[1] if plt.ylim()[1] != 1.0 else 0.18,
             label, fontsize=6.5, ha='center', va='top', color='darkgoldenrod', rotation=90)
    crisis_added = True
# Courbe brute et lissée
plt.plot(sentiment_by_year.index, sentiment_by_year.values, marker='o', linestyle='-', alpha=0.3, color='black')
# Légende
legend_patches = [Patch(color=color, alpha=0.4, label=party) for party, color in party_colors.items()]
legend_patches += [Patch(color='yellow', alpha=0.5, label='Crise / Guerre')]
plt.legend(handles=legend_patches, loc='upper left', fontsize=8)

plt.xlabel("Année")
plt.ylabel("Sentiment moyen")
plt.title("Évolution du sentiment dans les discours d'inauguration")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()