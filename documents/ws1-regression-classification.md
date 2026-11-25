# Workstream 1: Regression & Classification (Ansatz A + B)

Book Recommender System, Gruppe 1, ML HWS 2026/27.
Arbeitsebene: **Reviewebene** (jede Zeile = eine Bewertung eines Users zu einem Buch).

---

## Kontext und User Stories

### Ansatz A: Rating Prediction (Regression)

Ein Online Buchhaendler will vorhersagen, wie ein bestimmter Kunde ein Buch bewerten wuerde, das er noch nicht gelesen hat. Statt einfach den Gesamtdurchschnitt eines Buches anzuzeigen ("4.2 Sterne"), zeigt die Plattform eine personalisierte Prognose: "Basierend auf deinem bisherigen Leseverhalten wuerdest du dieses Buch wahrscheinlich mit 3.8 Sternen bewerten."

Anwendung: Personalisierte Sortierung in einem Buchshop. Buecher, fuer die das Modell einen hohen Score vorhersagt, erscheinen weiter oben.

### Ansatz B: Binaere Klassifikation ("Empfehlen ja/nein")

Eine Bibliotheks App will ihren Nutzern eine simple "Fuer dich" Liste anzeigen. Die Frage ist nicht "wie viele Sterne", sondern einfach: Wird dieser Nutzer dieses Buch moegen, ja oder nein? Die App filtert das gesamte Sortiment und zeigt nur Buecher, bei denen das Modell "ja" sagt.

Anwendung: Newsletter Personalisierung. Ein Buchhaendler verschickt woechentlich Empfehlungen und will nicht 20 Buecher schicken, sondern genau die 5, die der Kunde wahrscheinlich gut findet.

### Zusammenhang A und B

Man koennte B theoretisch aus A ableiten (Vorhersage >= 4 bedeutet "empfehlen"). Aber die Modelle optimieren auf unterschiedliche Ziele: Regression minimiert RMSE (exakter Score), Klassifikation maximiert F1 (saubere Trennlinie). Unterschiedliche Algorithmen mit unterschiedlichen Zielen fuehren zu unterschiedlichen Ergebnissen. Deshalb lohnt es sich, beides zu probieren und zu vergleichen.

---

## Datensatz

Zwei CSV Dateien, ueber `Title` gejoint:

- **Books_rating.csv**: Id, Title, Price, User_id, profileName, review/helpfulness, review/score (1-5), review/time, review/summary, review/text
- **books_data.csv**: Title, description, authors, publisher, publishedDate, categories, ratingsCount, image, previewLink, infoLink

Ca. 3 Millionen Reviews, ca. 212.000 einzigartige Buecher.

### Fehlende Werte (bekanntes Problem)

Nur 5,81 % aller Zeilen haben alle Attribute gefuellt. Die groessten Luecken:

| Spalte | Fehlend % | Strategie |
|---|---|---|
| `Price` | 83,9 % | **Spalte droppen.** Zu viel fehlend, Imputation bei 84 % waere mehr Erfindung als Daten. Preis ist kein starker Praediktor fuer Bewertungen. |
| `ratingsCount` | 45,4 % | **Spalte droppen.** Stammt aus Google Books API, nicht aus unserem Datensatz. Wir berechnen `book_review_count` selbst (100 % vorhanden). |
| `description` | 21,3 % | **Spalte droppen.** Freitext, ohne Textverarbeitung (nicht im Kurs) nicht als Feature nutzbar. |
| `image`, `previewLink`, `infoLink` | 11-18 % | **Spalten droppen.** URLs, keine Features. |
| `profileName` | 18,7 % | **Spalte droppen.** Wir nutzen `User_id`, nicht den Namen. |
| `publisher` | 26,1 % | **Optional behalten** mit "Unknown" als eigene Kategorie. Oder droppen. |
| `categories` | 18,4 % | **Behalten**, fehlende mit "Unknown" fuellen. |
| `authors` | 13,0 % | **Optional behalten** mit "Unknown". Oder droppen. |
| `publishedDate` | 11,8 % | **Optional** fuer `book_age`. Fehlende Zeilen ohne dieses Feature lassen. |
| `User_id` | 18,7 % | **Zeilen ohne User_id droppen.** Ohne User keine userbezogenen Features. |
| `review/score` | ~0 % | **Zeilen ohne Score droppen.** Das ist die Zielvariable. |

Nach dem Aufraeuemen bleiben ca. 80 % der Daten (rund 2,4 Mio Zeilen). Davon wird dann ein Subsample gezogen.

---

## Datensatz aufbereiten: "erst bereinigen, dann sampeln"

- Korrektheit der Stratifikation: Das Sampling stratifiziert nach review/score. Wenn noch Zeilen ohne Score drin sind, werden sie von groupby still ignoriert – das macht den Sampling-Code zwar nicht falsch, aber es ist sauberer, das explizit zu droppen.

- Repräsentativität: Wenn Zeilen ohne User_id erst nach dem Sampling rausfallen, kann die Klassenverteilung im Sample leicht kippen.

Praktischer Kompromiss: Man muss nicht alle 3 Mio. Zeilen durch den vollen Preprocessing-Stack jagen. Die minimale Bereinigung vor dem Sampling reicht:

# Minimal-Bereinigung VOR dem Subsampling
df = df.dropna(subset=['review/score', 'User_id'])
df = df.drop(columns=['Price', 'ratingsCount', 'description', 'image',
                       'previewLink', 'infoLink', 'profileName'])

# DANN sampeln
df_sample = df.groupby('review/score', group_keys=False).apply(
    lambda x: x.sample(frac=0.02, random_state=42)
).reset_index(drop=True)

---

## Schritt 1: Subsampling

Aus den bereinigten Daten ein kleineres, repraesentatives Sample ziehen.

```python
df_sample = df.groupby('review/score', group_keys=False).apply(
    lambda x: x.sample(frac=0.02, random_state=42)
)
```

- `frac=0.02` = 2 % pro Score Klasse, ergibt ca. 50.000-60.000 Zeilen
- `groupby('review/score')` macht es stratified: die Score Verteilung bleibt proportional gleich
- `random_state=42` macht es reproduzierbar

**Danach pruefen:**
```python
print(df_sample.shape)
print(df_sample['review/score'].value_counts(normalize=True))
# Verteilung muss mit dem Original uebereinstimmen
```

Warum nicht einfach die ersten 60.000 Zeilen? Weil die vielleicht alle aus dem gleichen Zeitraum oder der gleichen Kategorie kommen. Stratified Random Sampling ist sauberer.

# Feedback zum Code von Schritt 1
- Was gut ist:

groupby('review/score') mit frac=... ist sauberes stratifiziertes Sampling – genau richtig
group_keys=False verhindert einen störenden Multi-Index – korrekt
random_state=42 sorgt für Reproduzierbarkeit – unbedingt beibehalten
Der Prüfcode mit value_counts(normalize=True) ist wichtig – gut, dass er explizit erwähnt wird

- Was ich verbessern würde:

reset_index(drop=True) anhängen – nach groupby().apply() kann der Index chaotisch sein, das kann später beim .join() im Feature Engineering zu Bugs führen

frac=0.02 ist knapp – 2 % von 2,4 Mio. = ~48.000 Zeilen. Bei sehr seltenen Usern oder Büchern im Sample sind user_avg_score oder book_avg_score aus nur 1–2 Reviews berechnet – instabil. frac=0.03 (ca. 72.000 Zeilen) wäre robuster, bei modernen Rechnern kein Problem.

Review/score als Integer casten – der Score kann im CSV als Float (1.0, 2.0...) vorliegen, was beim Groupby zu Problemen führen kann:

df['review/score'] = df['review/score'].astype(int)
Prüfung nach dem Sampling erweitern – neben der Score-Verteilung auch die Größe prüfen:

print(df_sample.shape)
print(df_sample['review/score'].value_counts())           # Absolute Zahlen
print(df_sample['review/score'].value_counts(normalize=True))  # Anteile
Zusammenfassung: Der Kern-Code ist korrekt und durchdacht. Die drei kleinen Ergänzungen (reset_index, frac leicht erhöhen, Score zu int casten) machen ihn robuster für die späteren Schritte.

---

## Schritt 2: Preprocessing

### 2a: Spalten bereinigen und Datentypen fixen

**review/helpfulness parsen** (Format "2/3"):
```python
split = df_sample['review/helpfulness'].str.split('/', expand=True).astype(float)
df_sample['helpful_yes'] = split[0]
df_sample['helpful_total'] = split[1]
df_sample['helpfulness_ratio'] = df_sample['helpful_yes'] / df_sample['helpful_total']
# Division durch 0 behandeln:
df_sample['helpfulness_ratio'] = df_sample['helpfulness_ratio'].fillna(0)
```

**review/time umwandeln** (Unix Timestamp):
```python
df_sample['review_date'] = pd.to_datetime(df_sample['review/time'], unit='s')
df_sample['review_year'] = df_sample['review_date'].dt.year
```

**categories bereinigen** (ggf. String der wie Liste aussieht, z.B. `"['Fiction']"`):
```python
# Pruefen wie das Format aussieht:
print(df_sample['categories'].head(20))
# Falls noetig parsen, z.B.:
# df_sample['categories'] = df_sample['categories'].str.strip("[]'\"")
```

### 2b: Missing Values behandeln

| Spalte | Strategie | Code |
|---|---|---|
| `categories` (fehlend) | "Unknown" als eigene Kategorie | `df_sample['categories'].fillna('Unknown')` |
| `review/score` (fehlend) | Zeile entfernen | `df_sample.dropna(subset=['review/score'])` |
| `helpfulness_ratio` (NaN) | Mit 0 fuellen | Bereits oben gemacht |
| `User_id` (fehlend) | Zeile entfernen | `df_sample.dropna(subset=['User_id'])` |

Jede Entscheidung in einer Markdown Zelle im Notebook begruenden. Das ist Praesentationsmaterial.

### 2c: Feature Engineering

Neue Spalten bauen, die fuer die Modelle nuetzlicher sind als die Rohdaten:

| Feature | Berechnung | Idee dahinter |
|---|---|---|
| `helpfulness_ratio` | `helpful_yes / helpful_total` | Reviews mit hoher Helpfulness koennten reflektierter sein |
| `review_length` | Anzahl Woerter in `review/text` | Ausfuehrliche Reviews koennten informativer sein |
| `summary_length` | Anzahl Woerter in `review/summary` | Wie bei review_length |
| `user_avg_score` | Durchschnitt aller Scores dieses Users | Manche User sind generell strenger |
| `user_review_count` | Anzahl Reviews dieses Users | Vielreviewer vs. Einmalbewerter |
| `user_score_std` | Standardabweichung der Scores dieses Users | Wie stark streut der User? |
| `user_positive_ratio` | Anteil Score >= 4 an allen Reviews des Users | Direkter als avg_score fuer Ansatz B |
| `book_avg_score` | Durchschnitt aller Scores dieses Buches | Allgemeine Beliebtheit |
| `book_review_count` | Anzahl Reviews dieses Buches | Bekannte vs. unbekannte Buecher |
| `book_score_std` | Standardabweichung der Scores dieses Buches | Kontroverse Buecher sind schwerer vorherzusagen |
| `book_positive_ratio` | Anteil Score >= 4 an allen Reviews des Buches | Direkter fuer Ansatz B |
| `review_year` | Jahr aus dem Timestamp | Zeitliche Trends |
| `main_category` | Erster Eintrag aus `categories` | Vereinfacht die Kategorie auf ein Feature |
| `book_age` | Differenz Review-Datum minus publishedDate (optional) | Klassiker vs. Neuerscheinungen |

```python
# Beispiel: User Aggregate berechnen
user_stats = df_sample.groupby('User_id')['review/score'].agg(
    user_avg_score='mean',
    user_review_count='count',
    user_score_std='std'
)
user_stats['user_score_std'] = user_stats['user_score_std'].fillna(0)
user_stats['user_positive_ratio'] = df_sample.groupby('User_id')['review/score'].apply(
    lambda x: (x >= 4).mean()
)
df_sample = df_sample.join(user_stats, on='User_id')

# Beispiel: Book Aggregate berechnen
book_stats = df_sample.groupby('Title')['review/score'].agg(
    book_avg_score='mean',
    book_review_count='count',
    book_score_std='std'
)
book_stats['book_score_std'] = book_stats['book_score_std'].fillna(0)
df_sample = df_sample.join(book_stats, on='Title')
```

**Achtung Data Leakage:** `user_avg_score` und `book_avg_score` berechnet auf dem gesamten Datensatz (inkl. Testdaten) ist streng genommen Leakage. Fuer ein Uni Projekt akzeptabel, aber in der Diskussion erwaehnen.

### 2d: Kategorische Variablen encoden

```python
# Top 20 Kategorien behalten, Rest zu "Other"
top_cats = df_sample['main_category'].value_counts().nlargest(20).index
df_sample['main_category'] = df_sample['main_category'].where(
    df_sample['main_category'].isin(top_cats), 'Other'
)

# OneHotEncoder oder pd.get_dummies
# Bei Pipeline: OneHotEncoder in die Pipeline einbauen
# Ohne Pipeline: pd.get_dummies(df_sample, columns=['main_category'])
```

Optional: Gleiche Logik fuer `authors` und `publisher` (Top N, Rest = "Other"). Oder weglassen wenn zu viele Auspraegungen.

### 2e: Normalisierung

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# In einer Pipeline (empfohlen):
# Wird automatisch nur auf Trainingsdaten gefittet

# Ohne Pipeline:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # transform, NICHT fit_transform!
```

Wichtig fuer KNN (distanzbasiert). Decision Trees brauchen es nicht, aber es schadet nicht.

### 2f: Feature Auswahl (finale Spalten fuer X)

**Rein:**
- helpfulness_ratio, review_length, summary_length
- user_avg_score, user_review_count, user_score_std, user_positive_ratio
- book_avg_score, book_review_count, book_score_std, book_positive_ratio
- review_year
- OneHot codierte main_category Spalten
- Optional: book_age, publisher_encoded, author_encoded

**Nicht rein:**
- Title, User_id, profileName (Identifikatoren, keine Features)
- review/text, review/summary (Rohtext, nicht verarbeitbar ohne NLP)
- review/score (das ist die Zielvariable!)
- Price (zu viel fehlend, gedroppt)
- description, image, previewLink, infoLink (nicht nutzbar)

---

## Schritt 3: Zielvariable und Train/Test Split

### Ansatz A (Regression)

```python
y_reg = df_sample['review/score']  # Werte 1 bis 5, kontinuierlich
```

### Ansatz B (Klassifikation)

```python
y_clf = (df_sample['review/score'] >= 4).astype(int)
# 1 = empfehlen (Score 4 oder 5)
# 0 = nicht empfehlen (Score 1, 2 oder 3)

# Verteilung pruefen:
print(y_clf.value_counts(normalize=True))
# Vermutlich stark unbalanciert (viel mehr 1er als 0er)
```

### Train/Test Split

```python
from sklearn.model_selection import train_test_split

# Fuer Regression:
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# Fuer Klassifikation (mit stratify wegen Imbalance):
X_train, X_test, y_train, y_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)
```

### Erklaerung: Warum Train/Test/Validation (3-Way Split)

Normalerweise: Trainingsdaten (zum Lernen) und Testdaten (zum Bewerten).
Beim Hyperparameter Tuning braucht man ein drittes Set (Validierungsdaten), weil man sonst die Testdaten zum Tuning missbraucht und das Ergebnis zu optimistisch wird.

**Loesung: Cross Validation ersetzt das dritte Set.**
GridSearchCV nimmt die Trainingsdaten, teilt sie intern in k Folds auf und probiert alle Parameterkombinationen durch. Die Testdaten werden nie angefasst.

```
Gesamte Daten
|-- Trainingsdaten (80%)
|   +-- GridSearchCV teilt intern auf:
|       |-- Fold 1: Train / Validate
|       |-- Fold 2: Train / Validate
|       |-- ...
|       +-- Fold 5: Train / Validate
|       => Bestes Parameterset gefunden
+-- Testdaten (20%)
    +-- Einmal am Ende: wie gut ist das beste Modell wirklich?
```

Die Testdaten werden nur ein einziges Mal angefasst, ganz am Schluss.

---

## Schritt 4: Modelle trainieren

### Ansatz A: Regression

```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

models_reg = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'KNN Regressor': KNeighborsRegressor()
}

for name, model in models_reg.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Metriken berechnen (Schritt 6)
```

**Baseline Regression:** Immer den Durchschnittsscore vorhersagen.
```python
import numpy as np
baseline_pred = np.full(len(y_test), y_train.mean())
```

### Ansatz B: Klassifikation

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

models_clf = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN Classifier': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}
```

**Baseline Klassifikation:** Immer die Mehrheitsklasse vorhersagen.
```python
from collections import Counter
majority_class = Counter(y_train).most_common(1)[0][0]
baseline_pred = np.full(len(y_test), majority_class)
```

Jedes echte Modell muss besser sein als die Baseline, sonst hat es nichts gelernt.

### Imbalanced Data behandeln (nur Ansatz B)

Wenn die Klassenverteilung z.B. 75/25 ist:

```python
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# In der Pipeline:
pipe = ImbPipeline([
    ('scaler', StandardScaler()),
    ('sampler', RandomOverSampler(random_state=42)),
    ('model', RandomForestClassifier(random_state=42))
])
```

Resampling nur auf den Trainingsdaten, nie auf den Testdaten!
Die imblearn Pipeline (nicht die sklearn Pipeline) handhabt das automatisch korrekt.

---

## Schritt 5: Hyperparameter Tuning

Hyperparameter = Einstellungen VOR dem Training (k bei KNN, max_depth bei Trees, n_estimators bei Random Forest). Das Modell lernt sie nicht selbst.

### GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

# Beispiel: Decision Tree Classifier
param_grid = {
    'max_depth': [3, 5, 10, 20, None],
    'min_samples_leaf': [1, 5, 10, 20],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    scoring='f1',           # Metrik die optimiert wird
    cv=5,                   # 5-Fold Cross Validation
    n_jobs=-1,              # Alle CPU Kerne nutzen
    return_train_score=True
)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)   # Beste Parameterkombination
print(grid_search.best_score_)    # Bester CV Score
```

5 x 4 x 2 = 40 Kombinationen, jeweils 5-Fold CV = 200 Modelle werden trainiert.

### Beispiel Hyperparameter Grids pro Modell

**KNN:**
```python
param_grid_knn = {
    'n_neighbors': [1, 3, 5, 7, 11, 15, 20],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
```

**Random Forest:**
```python
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}
```

**Ridge Regression:**
```python
param_grid_ridge = {
    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
}
```

### Pipeline mit Tuning

Eine Pipeline kettet Preprocessing und Modell zusammen. Vorteil: Bei jeder CV Fold wird das Preprocessing nur auf den Trainingsdaten gefittet (kein Data Leakage).

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

# Parameter mit doppeltem Unterstrich ansprechen:
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [5, 10, 20, None]
}

grid_search = GridSearchCV(pipe, param_grid, scoring='f1', cv=5)
grid_search.fit(X_train, y_train)
```

Der Doppelunterstrich (`model__max_depth`) sagt: "max_depth gehoert zum Schritt model in der Pipeline."

---

## Schritt 6: Evaluation

### Ansatz A: Regression Metriken

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.3f}")  # Durchschnittliche Abweichung in Scorepunkten
print(f"MAE:  {mae:.3f}")   # Gleiche Skala wie RMSE, aber ohne Bestrafung grosser Fehler
print(f"R2:   {r2:.3f}")    # 1.0 = perfekt, 0.0 = nicht besser als Durchschnitt
```

Scatter Plot: Vorhergesagte vs. tatsaechliche Scores (Diagonale = perfekt).

### Ansatz B: Klassifikation Metriken

```python
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, accuracy_score, f1_score
)

y_pred = best_model.predict(X_test)

# Classification Report (Precision, Recall, F1 pro Klasse)
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
# Als Heatmap visualisieren mit matplotlib/seaborn

# ROC Kurve (braucht Wahrscheinlichkeiten)
y_proba = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
# Diagonale = Zufall, je steiler die Kurve desto besser
```

**Metrik Erklaerungen:**
- **Confusion Matrix**: Zeigt wo das Modell richtig und falsch lag (TP, TN, FP, FN)
- **Precision**: Von allen als "empfehlen" markierten, wie viele stimmen?
- **Recall**: Von allen echten "empfehlen" Faellen, wie viele gefunden?
- **F1**: Kompromiss aus Precision und Recall, nur hoch wenn beide hoch
- **Accuracy**: Nur bei balancierten Klassen sinnvoll! Bei Imbalance irrefuehrend.
- **ROC/AUC**: Trennschaerfe ueber alle Schwellenwerte, braucht predict_proba()

### Cross Validation fuer robustere Zahlen

```python
from sklearn.model_selection import cross_val_score

# Regression:
scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print(f"RMSE: {rmse_scores.mean():.3f} +/- {rmse_scores.std():.3f}")

# Klassifikation:
scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1')
print(f"F1: {scores.mean():.3f} +/- {scores.std():.3f}")
```

### Learning Curve

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_train, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='f1'  # oder 'neg_mean_squared_error' fuer Regression
)
# Plotten: Performance ueber wachsende Trainingsgroesse
# Niedrig und instabil = mehr Trainingsdaten noetig
```

### Ergebnistabelle

Alle Modelle untereinander mit Metriken, inklusive Baseline:

| Modell | RMSE | MAE | R2 | (fuer Regression) |
|---|---|---|---|---|
| Baseline (Durchschnitt) | ... | ... | 0.0 | |
| Linear Regression | ... | ... | ... | |
| Ridge | ... | ... | ... | |
| Decision Tree | ... | ... | ... | |
| KNN | ... | ... | ... | |

| Modell | Accuracy | Precision | Recall | F1 | AUC | (fuer Klassifikation) |
|---|---|---|---|---|---|---|
| Baseline (Mehrheitsklasse) | ... | ... | ... | ... | 0.5 | |
| Decision Tree | ... | ... | ... | ... | ... | |
| Random Forest | ... | ... | ... | ... | ... | |
| KNN | ... | ... | ... | ... | ... | |
| Naive Bayes | ... | ... | ... | ... | ... | |

---

## Schritt 7: Feature Importance und Interpretation

```python
# Decision Tree / Random Forest:
importances = best_tree.feature_importances_
feature_names = X.columns
# Balkendiagramm: Top 10 wichtigste Features

# Linear Regression / Ridge:
coefficients = best_linear.coef_
# Welche Features haben die groessten Gewichte?
```

---

## Schritt 8: Diskussion (fuer die Praesentation)

Fragen die beantwortet werden muessen:

- Welches Modell hat am besten funktioniert, und warum?
- Wie schwer war die Aufgabe? (RMSE von 0.8 bei Scores 1-5: gut oder schlecht?)
- Was waren die wichtigsten Features? (feature_importances_)
- Hat Ansatz B bessere Empfehlungen geliefert als die Schwelle aus Ansatz A?
- Was hat nicht funktioniert und warum?
- Was koennte man mit mehr Zeit verbessern? (Textanalyse der Reviews, groesseres Sample, andere Algorithmen)
- Data Leakage Diskussion: user_avg_score und book_avg_score auf dem gesamten Datensatz berechnet, was streng genommen Leakage ist
- Imbalanced Data: Wie hat sich Resampling auf die Ergebnisse ausgewirkt?
- Overfitting: Gibt es grosse Unterschiede zwischen Train und Test Performance?

---

## Checkliste: Nichts vergessen

- [ ] Daten geladen und gejoint
- [ ] Unnoetige Spalten gedroppt (Price, ratingsCount, description, image, URLs, profileName)
- [ ] Subsampling (stratified, random_state gesetzt)
- [ ] Missing Values behandelt und begruendet
- [ ] Features engineered (user_avg_score, book_avg_score, etc.)
- [ ] Kategorische Variablen encoded (Top 20, Rest = Other)
- [ ] Normalisierung (StandardScaler oder MinMaxScaler)
- [ ] Train/Test Split (80/20, stratify bei Klassifikation)
- [ ] Baseline definiert (Durchschnitt bzw. Mehrheitsklasse)
- [ ] Mindestens 3 Modelle trainiert pro Ansatz
- [ ] Imbalanced Data behandelt (Ansatz B)
- [ ] GridSearchCV fuer Hyperparameter Tuning
- [ ] Pipeline gebaut (Preprocessing + Modell)
- [ ] Evaluation mit passenden Metriken (RMSE/R2 bzw. F1/ROC)
- [ ] Cross Validation fuer robustere Schaetzung
- [ ] Learning Curve
- [ ] Feature Importance
- [ ] Ergebnistabelle mit allen Modellen + Baseline
- [ ] Diskussion vorbereitet
