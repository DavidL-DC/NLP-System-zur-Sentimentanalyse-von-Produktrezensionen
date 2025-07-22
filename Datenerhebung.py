import gzip
import json
import pandas as pd
import os

def load_reviews(filepath, category_name, limit=10000):
    data = []
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            review = json.loads(line)
            data.append({
                'category': category_name,
                'title': review.get('title', ''),
                'text': review.get('text', ''),
                'rating': int(review.get('rating', 0))
            })
    return pd.DataFrame(data)

# Dateinamen und Kategorienamen definieren
files = {
    "Video_Games": "Video_Games.jsonl.gz",
    "Pet_Supplies": "Pet_Supplies.jsonl.gz",
    "Automotive": "Automotive.jsonl.gz"
}

# Daten kombinieren
df_list = []
for category, filename in files.items():
    if os.path.exists(filename):
        print(f"Lade Daten aus {filename} ...")
        df = load_reviews(filename, category, limit=10000)
        df_list.append(df)
    else:
        print(f"Datei nicht gefunden: {filename}")

df_all = pd.concat(df_list, ignore_index=True)

if __name__ == '__main__':
    # Vorschau
    print("Vorschau auf die Daten:")
    print(df_all.head())

    # Bewertungshäufigkeiten pro Kategorie
    print("\nVerteilung der Bewertungen pro Kategorie:")
    print(df_all.groupby(['category', 'rating']).size().unstack().fillna(0).astype(int))