from Daten import df_all
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import re
import string

# Stoppwörter vorbereiten
nltk.download('stopwords')
stop_words = stopwords.words('english')

# 1. Titel + Text zusammenführen und klein schreiben
df_all['full_text'] = (df_all['title'].fillna('') + ' ' + df_all['text'].fillna('')).str.lower()

# 2. Bereinigen: Sonderzeichen, Zahlen, Leerzeichen
def clean_text(text):
    text = re.sub(r'\d+', '', text)  # Zahlen entfernen
    text = text.translate(str.maketrans('', '', string.punctuation))  # Satzzeichen entfernen
    text = text.strip()
    return text

df_all['clean_text'] = df_all['full_text'].apply(clean_text)

# 3. TF-IDF-Vektorisierung
vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=10000)
X = vectorizer.fit_transform(df_all['clean_text'])  # X = Merkmalsmatrix
y = df_all['rating']  # Zielvariable

# 4. Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

if __name__ == '__main__':
    # Ausgabe
    print("Trainingsdaten:", X_train.shape)
    print("Testdaten:", X_test.shape)
