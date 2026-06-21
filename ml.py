"""
ml.py — Entraînement KNN/SVM + séparation train/test + métriques réalistes

Ce module contient :
- la préparation des graphiques pour l'interface web,
- l'évaluation des prédictions,
- la séparation anti-fuite EPC train/test,
- le calcul d'erreur selon k pour KNN.
"""

import io
import base64
import copy

import matplotlib
from sklearn.ensemble import RandomForestClassifier
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd

from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,       # BLOC 4 — courbe ROC
    auc              # BLOC 4 — aire sous la courbe
)

from modeles import MODELES


#COULEURS

C = {
    'bg'   : '#141619',
    'panel': '#1a1d22',
    'text' : '#e8eaed',
    'muted': '#7a8090',
    'grid' : '#2a2e35',
    'in'   : '#00d4ff',
    'out'  : '#ffab40',
}


#BASE64

def _b64(fig):
    # Convertit une figure Matplotlib en image base64 pour l'interface web.
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor=C['bg'])
    buf.seek(0)
    s = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return s


#MATRICE CONFUSION 

def _g_confusion(y_true, y_pred):
    # Génère un graphique de matrice de confusion à partir des vraies et prédictions.
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(C['bg'])
    ax.set_facecolor(C['panel'])

    # Couleurs explicites par case :
    # Diagonale (bonnes prédictions) → vert foncé
    # Hors diagonale (erreurs)       → rouge foncé
    couleurs = [
        ['#1a5c38', '#7a1f1f'],   # Réel OUT : [TP_out, FP_in]
        ['#7a1f1f', '#1a5c38'],   # Réel IN  : [FN_out, TP_in]
    ]

    for i in range(2):
        for j in range(2):
            # Dessiner la case colorée manuellement
            rect = plt.Rectangle(
                [j - 0.5, i - 0.5], 1, 1,
                color=couleurs[i][j]
            )
            ax.add_patch(rect)
            # Texte toujours blanc — lisible sur fond foncé
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    fontsize=22, fontweight='bold',
                    color='white')

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Prédit OUT', 'Prédit IN'], color=C['text'], fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Réel OUT', 'Réel IN'], color=C['text'], fontsize=11)
    ax.set_title('Matrice de confusion', color=C['text'], fontsize=13, pad=12)
    ax.tick_params(colors=C['muted'])

    # Légende
    from matplotlib.patches import Patch
    legende = [
        Patch(color='#1a5c38', label='Bonne prédiction'),
        Patch(color='#7a1f1f', label='Erreur'),
    ]
    ax.legend(handles=legende, loc='upper right',
              facecolor=C['panel'], labelcolor=C['text'],
              fontsize=9, framealpha=0.8)

    plt.tight_layout()
    return _b64(fig)

#DISTRIBUTION RSSI 

def _g_rssi(X, y):
    # Affiche la distribution du RSSI moyen pour les classes IN et OUT.
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(C['bg'])
    ax.set_facecolor(C['panel'])
    ax.hist(X.loc[y == 1, 'rssi_moyen'], bins=20, alpha=0.75, color=C['in'],
            label='IN', edgecolor='white', linewidth=0.5)
    ax.hist(X.loc[y == 0, 'rssi_moyen'], bins=20, alpha=0.75, color=C['out'],
            label='OUT', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('RSSI moyen', color=C['text'])
    ax.set_ylabel('Nb EPC', color=C['text'])
    ax.set_title('Distribution RSSI', color=C['text'])
    ax.legend(facecolor=C['panel'], labelcolor=C['text'])
    ax.tick_params(colors=C['muted'])
    ax.grid(axis='y', linestyle='--', alpha=0.3, color=C['grid'])
    plt.tight_layout()
    return _b64(fig)


#MÉTRIQUES

def _g_metriques(m):
    # Crée un graphique horizontal des principales métriques de performance.
    noms    = ['Accuracy', 'F1-Score', 'Précision', 'Rappel']
    valeurs = [m['acc_test'], m['f1'] * 100, m['precision'] * 100, m['recall'] * 100]
    cols    = [C['in'], '#00e676', C['out'], '#ff5555']
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor(C['bg'])
    ax.set_facecolor(C['panel'])
    bars = ax.barh(noms, valeurs, color=cols, alpha=0.85, height=0.5)
    for bar, val in zip(bars, valeurs):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', color=C['text'])
    ax.set_xlim(0, 115)
    ax.set_xlabel('Score (%)', color=C['text'])
    ax.set_title('Métriques du modèle', color=C['text'])
    ax.tick_params(colors=C['muted'])
    ax.grid(axis='x', linestyle='--', alpha=0.3, color=C['grid'])
    ax.invert_yaxis()
    plt.tight_layout()
    return _b64(fig)


def evaluer_predictions(y_train, y_train_pred, y_test, y_test_pred):
    # Calcule les métriques de classification pour l'entraînement et le test.
    acc_train = round(accuracy_score(y_train, y_train_pred) * 100, 1)
    acc_test  = round(accuracy_score(y_test,  y_test_pred)  * 100, 1)
    f1        = f1_score(y_test,  y_test_pred,  zero_division=0)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall    = recall_score(y_test, y_test_pred,  zero_division=0)
    return {
        'acc_train': acc_train,
        'acc_test' : acc_test,
        'ecart'    : round(abs(acc_train - acc_test), 1),
        'f1'       : float(f1),
        'precision': float(precision),
        'recall'   : float(recall),
    }


#HELPERS SPLIT EPC

def _split_train_test_epc(X, y):
    """
    Split anti-leakage :
    - si colonne 'Epc' présente dans X -> split par EPC
    - sinon -> split classique par lignes
    """
    # Cette fonction évite que les mêmes tags EPC apparaissent à la fois dans train et test.
    if isinstance(X, np.ndarray):
        return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    if 'Epc' in X.columns:
        feats = X.copy()
        epcs  = feats['Epc'].astype(str)
        df_epc = (
            pd.DataFrame({'Epc': epcs, 'target': y.values})
            .drop_duplicates(subset=['Epc'])
            .reset_index(drop=True)
        )
        if df_epc['target'].nunique() < 2 or len(df_epc) < 2:
            train_mask = np.random.rand(len(feats)) >= 0.3
            test_mask  = ~train_mask
            return (
                feats.loc[train_mask].drop(columns=['Epc']),
                feats.loc[test_mask].drop(columns=['Epc']),
                y.loc[train_mask],
                y.loc[test_mask],
            )
        train_epcs, test_epcs = train_test_split(
            df_epc['Epc'].values, test_size=0.3,
            random_state=42, stratify=df_epc['target'].values
        )
        train_mask = feats['Epc'].isin(train_epcs)
        test_mask  = feats['Epc'].isin(test_epcs)
        return (
            feats.loc[train_mask].drop(columns=['Epc']),
            feats.loc[test_mask].drop(columns=['Epc']),
            y.loc[train_mask],
            y.loc[test_mask],
        )

    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# GRAPHIQUE ERREUR VS K (KNN)

def graphique_erreur_vs_k(X_train, X_test, y_train, y_test, scaler):
    # Crée un graphique d'erreur en fonction de k pour analyser le comportement du KNN.
    k_values  = range(1, 21)
    err_train = []
    err_test  = []
    X_train_sc = scaler.transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_sc, y_train)
        err_train.append(1 - knn.score(X_train_sc, y_train))
        err_test.append(1  - knn.score(X_test_sc,  y_test))
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(C['bg'])
    ax.set_facecolor(C['panel'])
    ax.plot(k_values, err_train, color=C['in'],  marker='o', label='Erreur Train')
    ax.plot(k_values, err_test,  color=C['out'], marker='s', label='Erreur Test')
    ax.set_xlabel('k (nombre de voisins)', color=C['text'])
    ax.set_ylabel("Taux d'erreur",         color=C['text'])
    ax.set_title('Erreur vs k — KNN',      color=C['text'])
    ax.legend(facecolor=C['panel'], labelcolor=C['text'])
    ax.tick_params(colors=C['muted'])
    ax.grid(linestyle='--', alpha=0.3, color=C['grid'])
    ax.set_xticks(list(k_values))
    plt.tight_layout()
    return _b64(fig)


def graphique_confusion(y_true, y_pred):
    # Wrapper simple pour exposer la matrice de confusion.
    return _g_confusion(y_true, y_pred)


# BLOC 4 — Courbe ROC (Receiver Operating Characteristic)
# La courbe ROC montre le compromis entre :
#   - Taux de vrais positifs (TPR = tags IN bien classés)
#   - Taux de faux positifs (FPR = parasites classés IN par erreur)
# L'aire sous la courbe (AUC) résume la qualité en un seul chiffre :
#   AUC = 1.0 → parfait | AUC = 0.5 → aléatoire
# Disponible pour KNN et SVM.

def graphique_roc(model, X_test_scaled, y_test):
    """
    Génère la courbe ROC pour KNN ou SVM.
    Nécessite que le modèle ait une méthode predict_proba ou decision_function.
    """
    # La courbe ROC montre le compromis entre sensibilité et spécificité.
    try:
        # SVC sans probability=True n'a pas predict_proba, on utilise decision_function
        if hasattr(model, 'predict_proba'):
            scores = model.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(model, 'decision_function'):
            scores = model.decision_function(X_test_scaled)
        else:
            return None   # Pas de score de confiance disponible

        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc     = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(C['bg'])
        ax.set_facecolor(C['panel'])

        # Courbe ROC
        ax.plot(fpr, tpr, color=C['in'],  linewidth=2,
                label=f'ROC (AUC = {roc_auc:.3f})')
        # Ligne de référence aléatoire
        ax.plot([0, 1], [0, 1], color=C['muted'], linewidth=1,
                linestyle='--', label='Aléatoire (AUC = 0.5)')

        ax.set_xlabel('Taux de Faux Positifs (FPR)', color=C['text'])
        ax.set_ylabel('Taux de Vrais Positifs (TPR)', color=C['text'])
        ax.set_title('Courbe ROC', color=C['text'])
        ax.legend(facecolor=C['panel'], labelcolor=C['text'], fontsize=10)
        ax.tick_params(colors=C['muted'])
        ax.grid(linestyle='--', alpha=0.3, color=C['grid'])
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        return _b64(fig)

    except Exception:
        return None   # En cas d'erreur, on renvoie None sans bloquer


def graphique_scores(model, X_test_scaled, y_test):
    """Affiche la distribution des scores de classification pour chaque classe."""
    # Compare la séparation des scores entre les classes IN et OUT.
    try:
        if hasattr(model, 'predict_proba'):
            scores = model.predict_proba(X_test_scaled)[:, 1]
            xlabel = 'Probabilité prédite (IN)'
            threshold = 0.5
        elif hasattr(model, 'decision_function'):
            scores = model.decision_function(X_test_scaled)
            xlabel = 'Decision function'
            threshold = 0.0
        else:
            return None

        y_test_arr = np.asarray(y_test)
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(C['bg'])
        ax.set_facecolor(C['panel'])

        ax.hist(scores[y_test_arr == 1], bins=25, alpha=0.7, color=C['in'], label='IN', edgecolor='white', linewidth=0.5)
        ax.hist(scores[y_test_arr == 0], bins=25, alpha=0.7, color=C['out'], label='OUT', edgecolor='white', linewidth=0.5)
        ax.axvline(threshold, color=C['muted'], linestyle='--', linewidth=1)

        ax.set_xlabel(xlabel, color=C['text'])
        ax.set_ylabel('Nombre d\'échantillons', color=C['text'])
        ax.set_title('Distribution des scores de classification', color=C['text'])
        ax.legend(facecolor=C['panel'], labelcolor=C['text'])
        ax.tick_params(colors=C['muted'])
        ax.grid(axis='y', linestyle='--', alpha=0.3, color=C['grid'])

        plt.tight_layout()
        return _b64(fig)

    except Exception:
        return None

# ENTRAINEMENT

def entrainer_modele(model_id, X_train, y_train, custom_params=None):
    # Entraîne un modèle KNN ou SVM sur les données d'entraînement.

    modele_info = MODELES.get(model_id)
    if not modele_info:
        raise ValueError(f"Modèle inconnu : {model_id}")

    params = copy.deepcopy(modele_info["params"])

    if custom_params:
        for k, v in custom_params.items():
            if k in params:
                params[k]["default"] = v

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if model_id == "knn":

        # ── BLOC 1 : Paramètres avancés KNN
        # On lit chaque paramètre depuis le dict params (avec valeur par défaut
        # si le paramètre n'existe pas, pour compatibilité avec l'ancien code).
        n_jobs_val = params.get("n_jobs", {}).get("default", -1)
        algorithm  = params.get("algorithm",  {}).get("default", "auto")
        p_val      = params.get("p",          {}).get("default", 2)

        model = KNeighborsClassifier(
            n_neighbors = params["n_neighbors"]["default"],
            metric      = params["metric"]["default"],
            weights     = params["weights"]["default"],
            algorithm   = algorithm,   # Bloc 1 — algo de recherche
            p           = int(p_val),       # Bloc 1 — exposant Minkowski
             n_jobs      = int(n_jobs_val), # Bloc 1 — parallélisation
        )

    elif model_id == "svm":

        #BLOC 1 : Paramètres avancés SVM
        degree_val  = params.get("degree",      {}).get("default", 3)
        tol_val     = params.get("tol",         {}).get("default", 0.001)
        prob_val    = params.get("probability", {}).get("default", "true")
        probability = prob_val == "true" or prob_val is True

        # "none" → None en Python (scikit-learn attend None, pas la string "none")
        max_iter_val = params.get("max_iter",    {}).get("default", -1)
        dfs_val      = params.get("decision_function_shape", {}).get("default", "ovr")
        cw_val       = params.get("class_weight", {}).get("default", "none")
        class_weight_final = None if cw_val == "none" else cw_val
        model = SVC(
            C                       = params["C"]["default"],
            kernel                  = params["kernel"]["default"],
             gamma                   = params["gamma"]["default"],
            degree                  = int(degree_val),
            tol                     = float(tol_val),
            probability             = probability,   # ← active predict_proba pour ROC
            max_iter                = int(max_iter_val),
            class_weight            = class_weight_final,
            decision_function_shape = dfs_val,
        )


    else:
        raise ValueError(f"Non implémenté : {model_id}")

    model.fit(X_train_scaled, y_train)
    return model, scaler


#GRIDSEARCH SVM

def gridsearch_svm(X_train, y_train, param_grid, scoring="accuracy", cv=5):
    """
    Lance un GridSearchCV sur SVM.
    Retourne (meilleur pipeline, meilleurs params, meilleur score %).
    """
    # Recherche automatique des meilleurs hyperparamètres SVM.
    grid = dict(param_grid)

    # Conversions de types
    if "class_weight" in grid:
        grid["class_weight"] = [None if v == "None" else v for v in grid["class_weight"]]
    for key in ("probability", "shrinking"):
        if key in grid:
            grid[key] = [
                True if v == "True" else (False if v == "False" else v)
                for v in grid[key]
            ]

    pipeline      = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
    grid_prefixed = {f"svc__{k}": v for k, v in grid.items()}

    gs = GridSearchCV(pipeline, grid_prefixed, scoring=scoring, cv=cv, n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)

    best_params = {k.replace("svc__", ""): v for k, v in gs.best_params_.items()}
    return gs.best_estimator_, best_params, round(gs.best_score_ * 100, 2)


#VALIDATION CROISÉE SVM

def crossval_svm(model, scaler, X, y, cv=5, scoring="accuracy", shuffle=True, random_state=42):
    """
    Validation croisée stratifiée sur le modèle SVM.
    Retourne (moyenne %, écart-type %).
    """
    # Mesure la robustesse du modèle sur plusieurs plis.
    kf       = StratifiedKFold(n_splits=cv, shuffle=shuffle,
                               random_state=random_state if shuffle else None)
    pipeline = Pipeline([("scaler", scaler), ("svc", model)])
    scores   = cross_val_score(pipeline, X, y, cv=kf, scoring=scoring, n_jobs=-1)
    return round(scores.mean() * 100, 2), round(scores.std() * 100, 2)