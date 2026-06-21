# Projet RFID Analytics — 

Projet — Big Data  
Réalisé en groupe dans le cadre du cours TIC (Technologies de l'Information et de la Communication)

---

## Présentation du projet

Ce projet a pour objectif d'améliorer la qualité des lectures RFID dans une chaîne logistique e-commerce.

Les lecteurs RFID placés sur un tapis convoyeur lisent les tags des cartons qui passent devant eux. Le problème principal est que les antennes captent aussi les tags des **cartons voisins** (amont et aval), ce qui génère des **lectures parasites** et fausse les données.

Notre application Flask permet de :

- Importer et visualiser des fichiers de lecture RFID
- Lancer une **analyse descriptive** pour mesurer le taux de lecture et identifier les tags manquants
- Entraîner des **modèles de machine learning** (KNN, SVM, Random Forest) pour classer les lectures en IN (valides) ou OUT (parasites)
- Comparer les résultats de plusieurs expériences
- Exporter les résultats en CSV ou PDF
- Consulter l'historique de toutes les analyses

---

## Stack technique

| Composant | Technologie |
|---|---|
| Backend | Python 3, Flask |
| Machine Learning | Scikit-learn |
| Manipulation de données | Pandas, NumPy |
| Visualisation | Matplotlib |
| Base de données | SQLite |
| Frontend | HTML, CSS, JavaScript (Vanilla) |
| Export PDF | ReportLab |

---

## Structure du projet

```
projet_rfid/
│
├── app.py                  # Point d'entrée Flask — toutes les routes
├── data.py                 # Gestion de la base de données SQLite
├── modeles.py              # Configuration des modèles ML (KNN, SVM, RF)
├── pretraitement.py        # Chargement et prétraitement des données RFID
├── ml.py                   # Entraînement KNN et SVM + graphiques
├── rf.py                   # Entraînement Random Forest + feature importances
├── export.py               # Génération des rapports PDF (ReportLab)
├── utils.py                # Fonctions utilitaires (fichiers, dossiers)
│
├── templates/              # Pages HTML (Jinja2)
│   ├── layout.html         # Template de base commun
│   ├── index.html          # Page d'import des fichiers
│   ├── datasets.html       # Liste des dossiers importés
│   ├── analyse_predictive.html  # Interface de lancement des expériences
│   ├── ml_p.html           # Affichage des résultats ML
│   ├── historique.html     # Historique des analyses
│   ├── detail_historique.html   # Détail d'une analyse
│   └── ...
│
├── static/
│   └── style.css           # Styles CSS globaux
│
├── data/                   # Dossiers de données importés (non versionné)
├── historique.db           # Base SQLite générée automatiquement
└── requirements.txt        # Dépendances Python
```

---

## Installation

### Prérequis

- Python 3.9 ou supérieur
- pip

### Étapes

1. **Cloner le dépôt**

```bash
git clone https://github.com/votre-compte/projet-rfid.git
cd projet-rfid
```

2. **Créer un environnement virtuel** (recommandé)

```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Lancer l'application**

```bash
python app.py
```

5. **Ouvrir dans le navigateur**

```
http://localhost:5000
```

---

## Format des données attendues

Le projet fonctionne avec trois types de fichiers à placer dans un même dossier :

| Fichier | Description |
|---|---|
| `ano_APTags*.csv` | Lectures RFID brutes (EPC, RSSI, LogTime, Reader, Ant…) |
| `ano_supply-process*.csv` | Fenêtres temporelles de passage (refListId, ciuchStart, ciuchStop) |
| `reflist_*.olpn` | Listes des EPCs attendus par run |

Le dossier entier s'importe directement depuis la page d'accueil de l'application.

---

## Modèles disponibles

### Analyse Descriptive
Pas de ML — mesure le taux de lecture, les runs complets et les tags manquants. À utiliser en premier pour diagnostiquer la qualité RFID.

### K-Nearest Neighbors (KNN)
Classe chaque EPC selon ses k voisins les plus proches dans l'espace RSSI. Simple et interprétable, bon pour une baseline.

### Support Vector Machine (SVM)
Trouve l'hyperplan optimal séparant les EPCs IN des parasites OUT. Robuste si bien paramétré.

### Random Forest (RF)
Combine plusieurs arbres de décision. Le plus robuste au bruit RFID. Fournit les importances des variables et un score OOB.

---

## Fonctionnalités principales

- **Import de dossiers** complets (APTags + supply-process + reflists)
- **Analyse descriptive** avec tolérance temporelle configurable
- **Analyse prédictive** avec comparaison de plusieurs expériences en parallèle
- **Graphiques** : matrice de confusion, courbe ROC, distribution des scores, importance des variables
- **Historique** : sauvegarde automatique de chaque analyse en base SQLite
- **Export** : CSV, PDF (rapport complet) ou ZIP (graphiques uniquement)
- **Relance** : modification des hyperparamètres et relance directement depuis les résultats

---

## Auteurs

Projet réalisé par des étudiants en 2ème année cycle ingénieur — Spécialité Big Data  
**ESIGELEC-- 2024/2025**

---

## Remarques

- Le fichier `data/temp_combined.csv` est ignoré par Git (trop volumineux)
- La base `historique.db` est générée automatiquement au premier lancement
- Le dossier `data/` n'est pas versionné — les jeux de données sont à importer manuellement
