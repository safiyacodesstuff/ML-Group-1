# ML Toolkit – Gruppe 1: Book Recommender

Arbeitsgrundlage aus dem Kurs Machine Learning (HWS 2026/27, Yannik Hahn).
Alles was wir kennen und einsetzen dürfen/sollen.

---

## Einordnung & Prozess

- **ML (Mitchell)**: Programm lernt aus Erfahrung E bei Aufgabe T, gemessen mit Metrik P.
- **Supervised** (Classification, Regression) = predictive tasks. **Unsupervised** (Clustering) = descriptive tasks.
- **Classification** sagt nominale Klassen voraus (binär, multi-class, multi-label), **Regression** kontinuierliche Werte.
- **Data Mining Process (Fayyad)**: Selection (inkl. Exploration/Profiling) → Preprocessing → Transformation → Data Mining (iterieren) → Interpretation.
- **CRISP-DM**: Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment.
- 70–80 % des Aufwands steckt in der Datenaufbereitung.

---

## Methoden

### Clustering (Unsupervised)

| Methode | Kern-Idee | Wichtige Parameter | Stärken / Schwächen |
|---|---|---|---|
| K-Means | Zentroide, iterative Zuweisung | k (Elbow-Methode), SSE | Einfach, effizient / k muss gewählt werden, Outlier-sensitiv, abhängig von Startpunkten |
| K-Medoids | Wie K-Means, aber Median statt Mittelwert | k | Robuster gegen Outlier |
| DBSCAN | Dichte-basiert: Core-, Border-, Noise-Punkte | eps, MinPts | Findet Outlier, beliebige Formen / Probleme bei variierender Dichte |
| Hierarchisch | Agglomerativ, Dendrogram | Linkage (Single, Complete, Average) | Kein k nötig / O(N³), max ~10k Datenpunkte |

- **Cluster-Ziel**: Intra-Cluster-Distanz minimieren, Inter-Cluster-Distanz maximieren. Qualität bei K-Means: SSE.
- **K-Means**: mehrfach mit zufälligen Startpunkten laufen lassen, kleinstes SSE wählen. SSE verschiedener k ist nicht direkt vergleichbar (sinkt immer mit k) → Elbow.
- **DBSCAN**: Start mit MinPts = 4, eps über k-Distance-Plot (Knick = Beginn der Noise-Punkte).

### Klassifikation (Supervised)

| Methode | Kern-Idee | Wichtige Parameter | Stärken / Schwächen |
|---|---|---|---|
| KNN | Mehrheitsvote der k Nachbarn (Lazy Learning), optional distanzgewichtet | k (1–20 testen), Distanzmaß | Flexibel, nicht-achsenparallele Grenzen / Langsam, braucht numerische Daten |
| Decision Tree | Greedy Top-down Splits nach Purity (Hunt's Algorithm, Eager Learning) | max_depth, Split-Kriterium (GINI/Entropy), Pruning | Interpretierbar, ignoriert irrelevante Attribute / Overfitting, keine Attribut-Interaktionen |
| Random Forest | Ensemble vieler Bäume: Bagging (zufällige Datenteilmengen) + zufällige Attributteilmenge pro Split | Anzahl Bäume, max_depth | Meist besser als ein Baum / weniger interpretierbar |
| Naïve Bayes | Bayes-Theorem, Unabhängigkeitsannahme | Laplace-Korrektur | Schnell, robust gegen irrelevante Attribute / Unabhängigkeitsannahme oft verletzt, redundante Attribute |

**Decision Trees im Detail**
- **Splits nach Datentyp**: nominal (Multiway oder binär), ordinal (Reihenfolge erhalten), continuous (Schwellenwert oder Diskretisierung).
- **Purity-Maße**: GINI Index, Entropy / Information Gain (bevorzugt viele kleine Partitionen), GainRATIO (korrigiert das). Bester Split = größter Purity Gain.
- **Pre-Pruning**: Mindestgröße pro Leaf, Mindest-Gain, max. Tiefe.

**Naïve Bayes im Detail**
- Prior P(C) aus Trainingsdaten, P(A|C) als Produkt der Einzelwahrscheinlichkeiten, Vergleich über unnormalisierten Posterior.
- Zero-Count-Problem → Laplace Estimator (+1).
- Missing Values: im Training nicht mitzählen, bei Klassifikation Attribut weglassen.
- Numerische Attribute: Diskretisierung oder Verteilungsannahme.

### Regression (Supervised)

| Methode | Kern-Idee | Wichtige Parameter | Stärken / Schwächen |
|---|---|---|---|
| KNN Regression | Durchschnitt der k Nachbarn | k | Einfach / Nur Interpolation, langsam |
| Regression Tree | Splits nach MSE-Reduktion, Durchschnitt im Leaf | max_depth (Pre-Pruning) | Nichtlinear / Overfitting, nur Interpolation |
| Linear Regression | y = w⊤x, Least Squares | Ridge: α (Regularisierung, α=0 = normal) | Schnell, extrapoliert / Nur lineare Zusammenhänge |
| Polynomial Regression | Erweiterung Linear Reg. | Grad m | Nichtlinear / Overfitting bei hohem m, v.a. in dünn besetzten Bereichen |
| Local Regression | KNN + lokale lineare Regression | k | Oft besser als reines KNN / Langsam |

## Distanz- und Ähnlichkeitsmaße

- **Euclidean Distance** – Standard für numerische Daten. Normalisierung vorher nötig.
- **Jaccard Coefficient** – Für binäre Daten (z.B. "hat User Buch bewertet ja/nein"). Ignoriert gemeinsame Nullen.
- **SMC** – Für symmetrische binäre Attribute. Zählt auch gemeinsame Nullen.
- **Cosine Similarity** – Für Textvektoren / aufgabenspezifisch (im Kurs erwähnt).

## Preprocessing

- **Datentypen**: Binary, Categorical/Nominal, Ordinal, Continuous
- **Normalisierung**: Attribute auf gleiche Skala bringen (z.B. [0,1]), sonst dominieren große Wertebereiche
- **Missing Values**: Entfernen, Imputation, oder bei Naïve Bayes einfach ignorieren
- **Diskretisierung**: Equal-Interval- oder Equal-Frequency-Binning, Binarisierung
- **Feature Engineering**: Dimensionsreduktion, Feature Subset Selection, Text-zu-Term-Vektor, Embeddings
- **Datenintegration**: mehrere Quellen zusammenführen (Join)
- **Imbalanced Data**: Stratified Sampling, Downsampling (verwirft Daten), Upsampling (Kopien → Gefahr des Auswendiglernens), synthetisch (SMOTE)

## Evaluation

### Metriken Klassifikation

Basis ist die **Confusion Matrix** (TP, TN, FP, FN).

| Metrik | Wann nutzen |
|---|---|
| Accuracy | Nur bei balancierten Klassen (sonst irreführend) |
| Precision (TP/(TP+FP)) / Recall (TP/(TP+FN)) | Wenn interessante Klasse selten ist |
| F1 | Harmonisches Mittel aus Precision und Recall, nur hoch wenn beide hoch |
| Cost-Sensitive Evaluation | Wenn Fehlerarten unterschiedlich teuer sind (Kostenmatrix) |
| ROC / AUC | Modellvergleich, braucht Konfidenzwerte (KNN, Naïve Bayes). Diagonale = Zufall, steiler = besser |

### Metriken Regression

| Metrik | Eigenschaft |
|---|---|
| MAE | Durchschnittliche Abweichung, gleiche Skala wie Zielgröße |
| MSE | Bestraft große Fehler stärker |
| RMSE | Wie MSE, aber gleiche Skala wie MAE |

### Evaluation Setup

- **Goldene Regel**: Nie auf Trainingsdaten testen, Train/Test strikt trennen (Data Leakage vermeiden). 100 % Accuracy → skeptisch sein.
- **Holdout**: 80/20 Split, bei Imbalance stratified.
- **Random Subsampling**: Holdout wiederholen und mitteln, Test-Sets können überlappen.
- **k-Fold Cross-Validation**: k=10 Standard, jeder Datenpunkt genau einmal im Testset, stratified bei Imbalance.
- **Faustregel**: Default ist Cross-Validation. Holdout mit festem Split bei großen Daten (> ~5000), langer Rechenzeit oder wenn Reproduzierbarkeit zählt (Projekt).
- **Learning Curve**: Performance über wachsende Trainingsgröße. Niedrig und instabil → mehr Trainingsdaten.

## Hyperparameter Tuning

- **Hyperparameter vs. Parameter**: Hyperparameter werden vor dem Training gesetzt (k, max_depth, Grad m), Parameter lernt das Modell (Splits, Gewichte, Wahrscheinlichkeiten).
- **Grid Search**: Systematisch alle Kombinationen. Teuer bei vielen Parametern (Curse of Dimensionality).
- **Random Search**: Zufällige Kombinationen, besser bei vielen Parametern.
- **Bayesian Optimization**: Lernt aus bisherigen Ergebnissen, wo als Nächstes gesucht wird.
- **3-Way Split**: Train / Validation (zum Tunen) / Test (nur finale Evaluation). Nie auf dem Testset tunen, sonst zu optimistische Schätzung.
- **Cross-Validation für Model Selection**: Alternative zum festen Validation-Set.
- **Nested Cross-Validation**: Inner Loop = HPO, Outer Loop = Generalisierungsfehler. Finales Modell oft auf allen Daten neu trainiert.
- **Feature Selection**: Forward (Features hinzufügen) oder Backward (entfernen). Decision Trees machen es automatisch, KNN und Naïve Bayes brauchen es explizit.

## Schlüsselkonzepte

- **Overfitting**: Gut auf Train, schlecht auf Test. Ursachen: Noise/Outlier, zu wenig Daten, zu hohe Komplexität. Gegenmittel: mehr Daten, Pruning, Ensembles, Regularisierung.
- **Underfitting**: Modell zu simpel, Train- und Testfehler hoch.
- **Occam's Razor**: Bei gleicher Güte das einfachere Modell wählen.
- **Bias-Variance Tradeoff**: Bias = zu einfach (Underfitting), Variance = zu flexibel (Overfitting). Optimale Komplexität = minimaler Gesamtfehler.
- **Baseline**: Einfachste sinnvolle Lösung (z.B. immer Durchschnitt vorhersagen). Jedes Modell muss die Baseline schlagen.
