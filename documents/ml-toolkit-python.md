# ML Toolkit Python – Gruppe 1: Book Recommender

Praktisches Gegenstück zu `ml-toolkit.md` (Theorie). Quelle: Exercise-Notebooks e01–e06.
Pakete: pandas, matplotlib, scikit-learn (sklearn), scipy, imblearn.

---

## 1. Daten laden & erkunden (pandas)

- **Laden**: `pd.read_csv`, `pd.read_excel` → DataFrame. ARFF: `scipy.io.arff.loadarff` → Bytes-Strings mit `.str.decode("utf-8")` umwandeln.
- **Ansehen**: `head()`, `tail()`, `index`, `describe()` (`include='all'` auch für nicht-numerische Spalten).
- **Sortieren**: `sort_values(by=..., ascending=...)`.
- **Projektion** (Spalten wählen): `df['col']`, `df[['a','b']]`.
- **Selektion** (Zeilen filtern): `df[df['col'] > x]`, Negation mit `~`.
- **Aggregation**: `groupby('col')` + `mean()`, `sum()`, `std()`, `min()`, `max()`.
- **Zeilen iterieren**: `itertuples()`, `iterrows()`.
- **Spalte hinzufügen**: `assign(...)`, `df['neu'] = ...`, `join()` (über Index, auch für mehrere Dateien).
- **Original sichern**: `df.copy()` vor dem Preprocessing.

## 2. Visualisierung (matplotlib)

- **Plot-Typen**: `hist` (bins variieren), `scatter`, `bar`, `line`, `pie`. Anzeige erst mit `plt.show()`.
- **Mehrere Plots**: `plt.subplots(rows, cols)`, `tight_layout()`.
- **Details**: `annotate()`, Gruppen per `groupby` einzeln plotten, Legende, Achsenbeschriftung.
- **Cluster einfärben**: `scatter(..., c=labels)`.
- **Klassenverteilung prüfen**: `value_counts()` + Balkendiagramm.

## 3. Grundprinzip sklearn

- **Daten**: `X` = 2D (n_samples, n_features), `target` = 1D (n_samples). Bei nur einem Feature: `reshape(-1, 1)`.
- **Estimator**: `fit(X, y)` lernt, `predict(X)` wendet an. Clustering: `fit_predict(X)` bzw. `labels_`.
- **Transformer** (Encoder, Scaler): `fit()`, `transform()`, `fit_transform()`.
- **Zielvariable** getrennt von den Features halten, Features-DataFrame enthält sie nicht.
- **Regel**: Auf Testdaten dasselbe Preprocessing mit `transform()` anwenden (nicht neu fitten).

## 4. Preprocessing (sklearn.preprocessing)

| Zweck | Klasse | Hinweis |
|---|---|---|
| Kategorien → Zahlen (mit Ordnung) | `OrdinalEncoder` | Impliziert eine Reihenfolge |
| Kategorien → 0/1-Spalten | `OneHotEncoder` | Spalte pro Wert, `get_feature_names_out()` |
| Zielvariable → Zahlen | `LabelEncoder` | `classes_` zeigt Zuordnung |
| Normalisieren auf [0,1] | `MinMaxScaler` | |
| Standardisieren (Mittelwert 0, Varianz 1) | `StandardScaler` | |
| Binning | `KBinsDiscretizer` | `n_bins`, `encode='ordinal'` |
| Polynom-Features | `PolynomialFeatures(degree)` | Für Polynomial Regression |

## 5. Modelle

### Clustering (`sklearn.cluster`, `scipy.cluster.hierarchy`)

| Methode | Klasse | Wichtige Parameter |
|---|---|---|
| K-Means | `KMeans` | `n_clusters`; immer Euclidean; Cluster-IDs variieren wegen zufälliger Initialisierung |
| Hierarchisch (nur Cluster) | `AgglomerativeClustering` | `n_clusters`, `linkage` (single, complete, average), `metric` |
| Hierarchisch (Dendrogram) | `linkage(data, method, metric)` + `dendrogram(Z)` | Metric: euclidean, cityblock, jaccard, cosine; `truncate_mode='lastp'`, `p`, `color_threshold`, `orientation` |
| DBSCAN | `DBSCAN` | `eps`, `min_samples`, `metric` (euclidean, cosine, manhattan); Noise = Label -1 |

- Hierarchisch: agglomerativ (bottom-up) oder divisiv (top-down).

### Klassifikation (Supervised)

| Methode | Klasse | Wichtige Parameter |
|---|---|---|
| KNN | `KNeighborsClassifier` | `n_neighbors` (Default 5), `weights` ('uniform'/'distance'), `metric` (euclidean, manhattan) |
| Decision Tree | `DecisionTreeClassifier` | `criterion` ('gini'/'entropy'), `max_depth`, `min_samples_leaf` |
| Random Forest | `RandomForestClassifier` | `n_estimators`, `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf` |
| Naïve Bayes | `GaussianNB` (kontinuierlich), `MultinomialNB` (diskret) | |

- **Baum visualisieren**: `tree.plot_tree(dt, feature_names=..., class_names=...)`.
- **Decision Boundary** zeigt Unterschied: Bäume achsenparallel, KNN flexibel.
- **Lazy vs. Eager**: KNN speichert nur Daten, Bäume lernen ein Modell.

### Regression (Supervised)

| Methode | Klasse |
|---|---|
| Linear | `LinearRegression` (`coef_`, `intercept_`) |
| Ridge | `Ridge` |
| KNN | `KNeighborsRegressor` |
| Regression Tree | `DecisionTreeRegressor` |
| Polynomial | `Pipeline([PolynomialFeatures(degree), LinearRegression])` |

- Vorgehen: Zusammenhang per Scatter-Plot prüfen, linear oder polynomial wählen, Train/Test-Split, fitten, Vorhersage plotten.
- Regressions-Metriken: `mean_squared_error` (RMSE = Wurzel daraus), `r2_score` (R²).

## 6. Evaluation (sklearn.model_selection / metrics)

- **Train/Test-Split**: `train_test_split(X, y, test_size, random_state, stratify)`. Default `test_size=0.25`, `random_state` für Reproduzierbarkeit, `stratify=y` bei Imbalance.
- **Warum nie auf Trainingsdaten testen**: Trainingsfehler sagt nichts über neue Daten (1-NN hätte Fehler 0).
- **Cross-Validation**: `cross_val_score(estimator, X, y, cv, scoring)` liefert Score pro Fold. Es erzeugt kein Modell → danach `fit()` aufrufen.
- **Stratified**: `StratifiedKFold(n_splits, shuffle=True, random_state)`.
- **Vorhersagen aus CV**: `cross_val_predict` (genau eine Vorhersage pro Datenpunkt, für Fehleranalyse).
- **Scoring-Werte**: 'accuracy', 'f1_micro', 'f1_macro' u.a.
- **Klassifikations-Metriken**: `confusion_matrix`, `accuracy_score`, `classification_report` (Precision, Recall, F1 pro Klasse).
- **ROC**: `roc_curve` braucht Konfidenzwerte → `predict_proba()` statt `predict()`. Diagonale = Zufall. Mit CV: pro Fold berechnen und mitteln (AUC mit Standardabweichung).
- **Kostenbasiert**: eigene Kostenmatrix + `make_scorer(func, greater_is_better=False)`.

## 7. Pipelines

- **`Pipeline([(name, transformer), ..., (name, estimator)])`**: Preprocessing wird in jedem CV-Fold korrekt nur auf den Trainingsdaten gefittet.
- **Parameter ansprechen**: `stepname__parametername` (zwei Unterstriche), z.B. `classification__n_neighbors`; ändern mit `set_params`.
- **Gemischte Datentypen**: `ColumnTransformer` (z.B. Scaler auf numerische, OneHotEncoder auf kategorische Spalten), eingebettet in die Pipeline.
- **Debugging**: eigener Zwischenschritt (Transformer, der Daten nur ausgibt).

## 8. Imbalanced Data (imblearn)

- Balancing nur auf dem **Trainingsset**, nie auf den Testdaten.
- **Undersampling**: `RandomUnderSampler`. **Oversampling**: `RandomOverSampler`. Methode: `fit_resample()`.
- In Pipelines den **imblearn-Pipeline** verwenden (`imblearn.pipeline.Pipeline`), nicht den sklearn-Pipeline.

## 9. Hyperparameter-Tuning

- **Grid Search**: `GridSearchCV(estimator, parameter_grid, scoring, cv)`. Grid ist ein Dict, Key = Parametername, Value = Liste. Verhält sich wie ein Estimator, `fit()` testet alle Kombinationen. Default `cv` = 3-Fold.
- **Random Search**: `RandomizedSearchCV`.
- **Ergebnisse**: `cv_results_`, `best_score_`, `best_params_`.
- **Auf Pipelines anwenden**, auch Preprocessing-Parameter tunen.
- **Nested CV**: GridSearchCV als Estimator in `cross_val_score` (Inner = Tuning, Outer = Generalisierungsfehler). Für das finale Modell trotzdem `fit()` auf der Grid Search aufrufen.
- Alternativ: Grid Search nur auf Trainingsset, finales Modell auf Testset bewerten.
- **Weitere Hilfen**: `n_jobs=-1` (parallel), `return_train_score`.
