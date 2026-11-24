# Workstream 2: Clustering & Jaccard Recommender (Ansatz C + E1)

Book Recommender System, Gruppe 1, ML HWS 2026/27.
Arbeitsebene: **Buchebene** (jede Zeile = ein Buch mit aggregierten Werten).

---

## Kontext und User Stories

### Ansatz C: Content-Based Clustering

Ein Nutzer hat gerade ein Buch fertig gelesen und sucht "etwas Aehnliches". Die Plattform kennt sein Bewertungsverhalten nicht, vielleicht ist er ein neuer Nutzer. Stattdessen schaut das System: In welcher "Buecherfamilie" liegt das gelesene Buch? Und empfiehlt andere Buecher aus derselben Familie.

Anwendung: Funktioniert auch beim Cold Start Problem (neuer Nutzer, keine Historie). Ein Verlag will sein Sortiment verstehen und herausfinden, welche natuerlichen Buchgruppen es gibt, die ueber die klassischen Genregrenzen hinausgehen.

### Ansatz E1: Kollaborativer Jaccard Recommender

"Kunden, die dieses Buch gekauft haben, kauften auch..." Die Empfehlung basiert nicht auf Bucheigenschaften, sondern auf dem Verhalten anderer User. Wenn viele Leute sowohl Buch A als auch Buch B bewertet haben, sind die Buecher offenbar verwandt, egal ob sie in der gleichen Kategorie stehen.

Anwendung: Personalisierte Empfehlungen basierend auf kollektivem Nutzerverhalten. Findet Zusammenhaenge, die rein inhaltlich nicht sichtbar sind (z.B. dass Thriller Leser auch True Crime Sachbuecher moegen).

### Unterschied C vs. E1

Ansatz C empfiehlt basierend auf **Buchmerkmalen** (Kategorie, Preis, Score). Ansatz E1 empfiehlt basierend auf **Nutzerverhalten** (welche Buecher werden von denselben Leuten gelesen). Beides hat Staerken und Schwaechen:

- C funktioniert auch ohne Nutzerdaten (Cold Start), versteht aber nicht, dass jemand, der Thriller liebt, vielleicht auch True Crime Sachbuecher mag.
- E1 findet genau solche ueberraschenden Verbindungen, braucht aber genug Nutzerdaten pro Buch.

Dieser Vergleich ist gutes Material fuer die Praesentations-Diskussion.

---

## Datensatz

Zwei CSV Dateien, ueber `Title` gejoint:

- **Books_rating.csv**: Id, Title, Price, User_id, profileName, review/helpfulness, review/score (1-5), review/time, review/summary, review/text
- **books_data.csv**: Title, description, authors, publisher, publishedDate, categories, ratingsCount, image, previewLink, infoLink

Ca. 3 Millionen Reviews, ca. 212.000 einzigartige Buecher.

### Fehlende Werte Strategie (speziell fuer WS2)

WS2 arbeitet auf Buchebene. Die Aggregation loest viele Probleme:
- `Price` (84 % fehlend): Optional. Fuer Clustering nuetzlich, aber bei so viel fehlend eher droppen oder mit Median fuellen.
- `categories` (18 % fehlend): "Unknown" als eigene Kategorie.
- `authors`, `publisher`: "Unknown" oder droppen.
- `ratingsCount`: Nicht noetig, wir berechnen `review_count` selbst.
- `description`, `image`, URLs: Nicht nutzbar (Freitext bzw. keine Features).

---

## Teil 1: Ansatz C (Clustering)

### Schritt 1: Daten auf Buchebene aggregieren

Kein Subsampling noetig! Die Aggregation reduziert die Datenmenge von 3 Mio auf ca. 212.000 Buecher.

```python
books = df.groupby('Title').agg({
    'review/score': ['mean', 'std', 'count'],
    'Price': 'first',
    'categories': 'first',
    'authors': 'first',
    'publisher': 'first'
})
books.columns = ['avg_score', 'score_std', 'review_count', 'price', 'category', 'author', 'publisher']

# score_std ist NaN bei Buechern mit nur 1 Review:
books['score_std'] = books['score_std'].fillna(0)
```

**Optional aber empfohlen:** Nur Buecher mit mindestens 5 Reviews behalten.
```python
books = books[books['review_count'] >= 5]
print(f"Buecher nach Filter: {len(books)}")
```
Begruendung: Ein Durchschnittsscore aus 1-2 Reviews ist kaum belastbar. Der Schwellenwert (5, 10, etc.) ist eine Entscheidung die ihr begruenden muesst.

### Schritt 2: Preprocessing fuer Clustering

**Features auf Buchebene:**

| Feature | Herkunft | Warum nuetzlich fuers Clustering |
|---|---|---|
| `avg_score` | Durchschnittsscore des Buches | Unterscheidet beliebte von unbeliebten Buechern |
| `review_count` | Anzahl Reviews | Unterscheidet bekannte von nischigen Buechern |
| `score_std` | Streuung der Bewertungen | Unterscheidet einhellig bewertete von kontroversen Buechern |
| `price` | Buchpreis (falls vorhanden) | Preissegment als Unterscheidungsmerkmal |
| `main_category` (OneHot) | Kategorie | Der offensichtlichste Unterschied zwischen Buechern |
| `book_age` (optional) | Alter seit Veroeffentlichung | Neuerscheinungen vs. Klassiker |
| `avg_review_length` | Durchschnittliche Reviewlaenge pro Buch | Buecher die lange Reviews bekommen loesen staerkere Reaktionen aus |
| `avg_helpfulness` | Durchschnittliche Helpfulness der Reviews | Engagiertere Leserschaft |

**Kategorien encoden:**
```python
# Hauptkategorie extrahieren (erster Eintrag)
books['main_category'] = books['category'].str.strip("[]'\"").str.split(',').str[0].str.strip("' ")
books['main_category'] = books['main_category'].fillna('Unknown')

# Top 20 behalten, Rest = "Other"
top_cats = books['main_category'].value_counts().nlargest(20).index
books['main_category'] = books['main_category'].where(
    books['main_category'].isin(top_cats), 'Other'
)

# OneHot Encoding
books_encoded = pd.get_dummies(books, columns=['main_category'])
```

**Normalisierung (ZWINGEND fuer Clustering!):**
```python
from sklearn.preprocessing import StandardScaler

feature_cols = [...]  # Alle numerischen + OneHot Spalten
scaler = StandardScaler()
X_books = scaler.fit_transform(books_encoded[feature_cols])
```

Ohne Normalisierung clustert K-Means fast nur nach `review_count` (groesster Wertebereich) und ignoriert alles andere. Features mit grossen Wertebereichen dominieren die Distanzberechnung.

**Spalten die NICHT als Feature reingehen:**
- Title (Identifikator)
- author, publisher (zu viele Auspraegungen, ausser ihr macht Top-N OneHot)
- Roh-category (bereits als main_category encoded)

### Schritt 3: K-Means Clustering

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Elbow Methode: SSE fuer verschiedene k berechnen
sse = []
k_range = range(2, 16)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_books)
    sse.append(km.inertia_)

# Elbow Plot
plt.figure(figsize=(10, 6))
plt.plot(k_range, sse, marker='o')
plt.xlabel('Anzahl Cluster (k)')
plt.ylabel('SSE (Sum of Squared Errors)')
plt.title('Elbow Methode')
plt.show()
# Knick im Plot = guter Wert fuer k
```

**K-Means mehrfach laufen lassen:** `n_init=10` bedeutet 10 Durchlaeufe mit zufaelligen Startpunkten, der mit dem kleinsten SSE wird gewaehlt. Das ist wichtig, weil K-Means abhaengig von der Initialisierung ist.

```python
# Finales Clustering mit gewaehltem k
best_k = ...  # Aus Elbow Plot ablesen
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
books['cluster'] = km_final.fit_predict(X_books)
```

### Schritt 4: DBSCAN (Ergaenzung zu K-Means)

DBSCAN als zweiten Clustering Algorithmus zeigen. Vorteil: Findet Outlier automatisch (Noise Punkte), braucht kein k.

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# eps bestimmen ueber k-Distance Plot:
nn = NearestNeighbors(n_neighbors=4)
nn.fit(X_books)
distances, indices = nn.kneighbors(X_books)
distances = np.sort(distances[:, -1])

plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.xlabel('Datenpunkte (sortiert)')
plt.ylabel('Distanz zum 4. Nachbarn')
plt.title('k-Distance Plot (MinPts=4)')
plt.show()
# Knick = guter Wert fuer eps

# DBSCAN ausfuehren
db = DBSCAN(eps=..., min_samples=4)  # eps aus dem Plot ablesen
books['cluster_dbscan'] = db.fit_predict(X_books)

# Noise Punkte haben Label -1
n_noise = (books['cluster_dbscan'] == -1).sum()
n_clusters = len(set(books['cluster_dbscan'])) - (1 if -1 in books['cluster_dbscan'].values else 0)
print(f"Cluster: {n_clusters}, Noise: {n_noise}")
```

**Vergleich K-Means vs. DBSCAN** in der Praesentation diskutieren:
- K-Means: Einfach, aber k muss gewaehlt werden, Outlier-sensitiv
- DBSCAN: Findet Outlier, beliebige Clusterformen, aber eps-sensitiv

### Schritt 5: Cluster interpretieren

```python
# Pro Cluster: Durchschnittswerte aller Features
cluster_summary = books.groupby('cluster').agg({
    'avg_score': 'mean',
    'review_count': 'mean',
    'score_std': 'mean',
    'price': 'mean',
    'main_category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
}).round(2)

print(cluster_summary)
```

**Cluster benennen:**
- z.B. "Cluster 0 = guenstige, viel bewertete Belletristik"
- z.B. "Cluster 3 = teure, wenig bewertete Fachbuecher"

**Visualisierungen:**
```python
# Balkendiagramm: Clustergroessen
books['cluster'].value_counts().sort_index().plot(kind='bar')
plt.title('Anzahl Buecher pro Cluster')
plt.show()

# Scatter Plot mit eingefaerbten Clustern (2 Features waehlen)
plt.scatter(books['avg_score'], books['review_count'],
            c=books['cluster'], cmap='viridis', alpha=0.5)
plt.xlabel('Durchschnittsscore')
plt.ylabel('Anzahl Reviews')
plt.title('Cluster Visualisierung')
plt.colorbar(label='Cluster')
plt.show()
```

### Schritt 6: Empfehlung demonstrieren (Clustering-basiert)

Ein konkretes Beispiel im Notebook zeigen:

```python
# User hat Buch X gelesen und mit 5 Sternen bewertet
example_book = "Harry Potter and the Sorcerer's Stone"  # Beispiel
example_cluster = books.loc[example_book, 'cluster']

# Andere Buecher im gleichen Cluster, sortiert nach avg_score
recommendations = books[
    (books['cluster'] == example_cluster) &
    (books.index != example_book)
].sort_values('avg_score', ascending=False).head(10)

print(f"Buch: {example_book}")
print(f"Cluster: {example_cluster}")
print(f"\nEmpfehlungen aus dem gleichen Cluster:")
print(recommendations[['avg_score', 'review_count', 'main_category']])
```

Das muss kein fertiges System sein. Ein nachvollziehbares Beispiel reicht.

---

## Teil 2: Ansatz E1 (Jaccard Recommender)

### Erklaerung: Was ist die User-Book-Matrix?

Eine riesige Tabelle. Die Zeilen sind User, die Spalten sind Buecher. In jeder Zelle steht entweder 1 (User hat dieses Buch bewertet) oder 0 (hat er nicht).

```
              Buch A    Buch B    Buch C    Buch D
User 1          1         0         1         0
User 2          1         1         0         0
User 3          0         0         1         1
User 4          1         1         1         0
```

Die meisten Zellen sind 0, weil kein User alle Buecher bewertet hat. Die Matrix ist extrem duenn besetzt (sparse).

### Erklaerung: Was macht Jaccard?

Jaccard vergleicht zwei Buecher, indem es schaut: Wie viele User haben BEIDE Buecher bewertet, geteilt durch wie viele User MINDESTENS EINES der beiden bewertet haben?

```
Jaccard(Buch A, Buch C) = |User die A UND C bewertet haben|
                          / |User die A ODER C (oder beide) bewertet haben|
```

Beispiel mit der Matrix oben:
- Buch A bewertet von: User 1, 2, 4
- Buch C bewertet von: User 1, 3, 4
- Beide bewertet: User 1, 4 (= 2)
- Mindestens eins bewertet: User 1, 2, 3, 4 (= 4)
- Jaccard(A, C) = 2/4 = 0.5

Je hoeher der Jaccard Wert, desto aehnlicher sind die Buecher (gemessen daran, dass dieselben Leute sie lesen).

### Erklaerung: Warum Jaccard und nicht SMC?

SMC (Simple Matching Coefficient) zaehlt auch die gemeinsamen Nullen mit. Das heisst, wenn zwei Buecher beide von 99 % der User NICHT bewertet wurden, wuerde SMC sagen "sehr aehnlich". Das ist Unsinn, weil "beide nicht gelesen" keine Aussage ueber Aehnlichkeit ist.

Jaccard ignoriert die gemeinsamen Nullen und schaut nur auf die positiven Uebereinstimmungen. Genau das passt zu asymmetrischen binaeren Attributen (im Kurs gelernt).

### Schritt 1: Daten vorbereiten fuer Jaccard

Die volle Matrix (3M User x 212k Buecher) ist zu gross. Subsamplen:

```python
# Nur Buecher mit mindestens N Reviews behalten
min_reviews_per_book = 50
popular_books = df.groupby('Title').size()
popular_books = popular_books[popular_books >= min_reviews_per_book].index

# Nur User mit mindestens M Reviews behalten
min_reviews_per_user = 10
active_users = df.groupby('User_id').size()
active_users = active_users[active_users >= min_reviews_per_user].index

# Filtern
df_filtered = df[
    (df['Title'].isin(popular_books)) &
    (df['User_id'].isin(active_users))
]

print(f"Buecher: {df_filtered['Title'].nunique()}")
print(f"User: {df_filtered['User_id'].nunique()}")
print(f"Reviews: {len(df_filtered)}")
```

Ziel: Eine handhabbare Matrix. Lieber strenger filtern und eine kleinere, dichtere Matrix haben.

### Schritt 2: User-Book-Matrix bauen

```python
# Binaere Matrix: Hat User das Buch bewertet? (1/0)
user_book_matrix = df_filtered.pivot_table(
    index='User_id',
    columns='Title',
    values='review/score',
    aggfunc='count',  # Nur ob bewertet, nicht wie
    fill_value=0
)

# Alles > 0 auf 1 setzen (binaer)
user_book_matrix = (user_book_matrix > 0).astype(int)

print(f"Matrix Groesse: {user_book_matrix.shape}")
print(f"Dichte: {user_book_matrix.sum().sum() / user_book_matrix.size:.4f}")
# Dichte wird sehr niedrig sein (< 1 %)
```

### Schritt 3: Jaccard Similarity berechnen

```python
from sklearn.metrics import pairwise_distances

# Jaccard Distance = 1 - Jaccard Similarity
# sklearn berechnet Distanz, wir brauchen Similarity
book_matrix = user_book_matrix.T  # Spalten = User, Zeilen = Buecher

jaccard_dist = pairwise_distances(book_matrix.values, metric='jaccard')
jaccard_sim = 1 - jaccard_dist

# Als DataFrame fuer leichtere Handhabung
jaccard_df = pd.DataFrame(
    jaccard_sim,
    index=book_matrix.index,
    columns=book_matrix.index
)
```

**Achtung:** Bei vielen Buechern kann die pairwise Berechnung lange dauern. Ggf. auf die Top 1000-2000 meistbewerteten Buecher einschraenken.

### Schritt 4: Empfehlungen generieren

```python
def recommend_jaccard(book_title, jaccard_df, n=10):
    """Empfehle die n aehnlichsten Buecher basierend auf Jaccard Similarity."""
    if book_title not in jaccard_df.index:
        return f"Buch '{book_title}' nicht in der Matrix."

    # Aehnlichkeiten fuer dieses Buch, absteigend sortiert
    similarities = jaccard_df[book_title].sort_values(ascending=False)

    # Sich selbst ausschliessen (Similarity = 1.0)
    similarities = similarities[similarities.index != book_title]

    return similarities.head(n)

# Beispiel:
example_book = "Harry Potter and the Sorcerer's Stone"  # Anpassen an echten Titel
recs = recommend_jaccard(example_book, jaccard_df, n=10)
print(f"\nTop 10 Empfehlungen fuer: {example_book}\n")
print(recs)
```

### Schritt 5: Ergebnisse interpretieren und visualisieren

```python
# Top Empfehlungen als Tabelle mit Zusatzinfos
rec_titles = recs.index.tolist()
rec_details = books.loc[books.index.isin(rec_titles), 
                        ['avg_score', 'review_count', 'main_category']]
rec_details['jaccard_similarity'] = recs.values[:len(rec_details)]
rec_details = rec_details.sort_values('jaccard_similarity', ascending=False)
print(rec_details)
```

**Fragen fuer die Interpretation:**
- Sind die Empfehlungen inhaltlich plausibel? (gleiche Kategorie? gleicher Autor?)
- Gibt es ueberraschende Empfehlungen? (unterschiedliche Kategorien, aber trotzdem hohe Jaccard Similarity, das waere ein Zeichen dass der kollaborative Ansatz etwas findet, was Clustering nicht sehen kann)
- Wie hoch sind die Jaccard Werte? (bei sehr duenner Matrix oft sehr niedrig)

---

## Teil 3: Vergleich C vs. E1 (fuer die Praesentation)

### Direkter Vergleich an einem Beispiel

```python
# Fuer dasselbe Buch: Was empfiehlt Clustering, was empfiehlt Jaccard?
example_book = "..."

# Clustering Empfehlung (Ansatz C)
cluster_id = books.loc[example_book, 'cluster']
cluster_recs = books[
    (books['cluster'] == cluster_id) & (books.index != example_book)
].sort_values('avg_score', ascending=False).head(10)

# Jaccard Empfehlung (Ansatz E1)
jaccard_recs = recommend_jaccard(example_book, jaccard_df, n=10)

print("=== Clustering Empfehlungen ===")
print(cluster_recs[['avg_score', 'review_count', 'main_category']].head(10))

print("\n=== Jaccard Empfehlungen ===")
# Jaccard Empfehlungen mit Buchdetails anreichern
jaccard_titles = jaccard_recs.index.tolist()
print(books.loc[books.index.isin(jaccard_titles),
                ['avg_score', 'review_count', 'main_category']])
```

### Ueberschneidung pruefen

```python
# Wie viele Buecher empfehlen beide Ansaetze?
cluster_set = set(cluster_recs.index[:10])
jaccard_set = set(jaccard_recs.index[:10])
overlap = cluster_set & jaccard_set

print(f"Clustering Top 10: {len(cluster_set)}")
print(f"Jaccard Top 10: {len(jaccard_set)}")
print(f"Ueberschneidung: {len(overlap)}")
print(f"Gemeinsam empfohlen: {overlap}")
```

### Diskussionspunkte fuer die Praesentation

| Aspekt | Clustering (C) | Jaccard (E1) |
|---|---|---|
| Basis | Bucheigenschaften (Kategorie, Score, Preis) | Nutzerverhalten (wer liest was) |
| Cold Start | Funktioniert (braucht keine User Historie) | Braucht genuegend Bewertungen pro Buch |
| Ueberraschende Empfehlungen | Nein, bleibt in der "Buecherfamilie" | Ja, findet genre-uebergreifende Muster |
| Interpretierbarkeit | Hoch (Cluster = Buchgruppen) | Mittel (warum aehnlich? Weil gleiche Leute lesen) |
| Skalierbarkeit | Gut (nur Buchfeatures) | Problematisch bei vielen Usern/Buechern |
| Sparse Data | Kein Problem | Grosses Problem (duenne Matrix = niedrige Similarity) |

### Optionale Erweiterung: Cosine Similarity auf numerischen Buchfeatures

Statt Buecher in harte Cluster zu stecken, fuer jedes Buch die N aehnlichsten direkt per Cosine Similarity finden. Das ist feiner als Clustering (kontinuierliche Aehnlichkeit statt harte Gruppenzugehoerigkeit).

```python
from sklearn.metrics.pairwise import cosine_similarity

# Auf den gleichen normalisierten Buchfeatures wie beim Clustering
cosine_sim = cosine_similarity(X_books)
cosine_df = pd.DataFrame(cosine_sim, index=books.index, columns=books.index)

def recommend_cosine(book_title, cosine_df, n=10):
    if book_title not in cosine_df.index:
        return f"Buch '{book_title}' nicht gefunden."
    sims = cosine_df[book_title].sort_values(ascending=False)
    sims = sims[sims.index != book_title]
    return sims.head(n)
```

Wichtig: Cosine Similarity NUR auf numerischen Features, nicht auf Freitext (Textverarbeitung war nicht im Kurs).

---

## Diskussion (fuer die Praesentation)

Fragen die beantwortet werden muessen:

- Welche natuerlichen Buchgruppen hat das Clustering gefunden?
- Sind die Cluster inhaltlich sinnvoll? Passen die Buecher in jedem Cluster zusammen?
- Wie hat sich die Wahl von k (K-Means) bzw. eps (DBSCAN) auf das Ergebnis ausgewirkt?
- K-Means vs. DBSCAN: Was hat besser funktioniert und warum?
- Jaccard: Wie duenn ist die Matrix? Wie hoch sind die Similarity Werte?
- Welcher Ansatz liefert die plausibleren Empfehlungen?
- Gibt es Buecher, die von beiden Ansaetzen empfohlen werden? Was sagt das?
- Was koennte man verbessern? (Mehr Daten, Textanalyse, Hybrid aus C und E1)
- Sparse Data Problem: Wie geht man damit um, dass die meisten User nur wenige Buecher bewertet haben?

---

## Checkliste: Nichts vergessen

### Clustering (Ansatz C)
- [ ] Daten auf Buchebene aggregiert
- [ ] Mindest-Reviewanzahl Filter angewendet und begruendet
- [ ] Features ausgewaehlt und begruendet
- [ ] Kategorien encoded (Top 20, Rest = Other)
- [ ] Normalisierung (ZWINGEND vor Clustering)
- [ ] K-Means mit Elbow Methode (k bestimmt)
- [ ] K-Means mehrfach mit zufaelligen Startpunkten (n_init=10)
- [ ] DBSCAN mit k-Distance Plot (eps bestimmt)
- [ ] Cluster interpretiert und benannt
- [ ] Visualisierungen (Elbow Plot, Cluster Scatter, Clustergroessen)
- [ ] Empfehlung anhand eines konkreten Beispiels demonstriert
- [ ] K-Means vs. DBSCAN Vergleich

### Jaccard Recommender (Ansatz E1)
- [ ] Daten gefiltert (min Reviews pro Buch, min Reviews pro User)
- [ ] User-Book-Matrix gebaut (binaer: bewertet ja/nein)
- [ ] Matrix Groesse und Dichte dokumentiert
- [ ] Jaccard Similarity berechnet
- [ ] Empfehlungsfunktion gebaut
- [ ] Konkretes Beispiel gezeigt und interpretiert
- [ ] Erklaert warum Jaccard und nicht SMC

### Vergleich und Diskussion
- [ ] Gleiches Beispielbuch in beiden Ansaetzen
- [ ] Ueberschneidung der Empfehlungen geprueft
- [ ] Staerken und Schwaechen diskutiert
- [ ] Cold Start Problem erwaehnt
- [ ] Sparse Data Problem erwaehnt
- [ ] Optional: Cosine Similarity auf numerischen Features als dritter Vergleich
