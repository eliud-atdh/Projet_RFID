"""
rf.py
-----
Entraînement et évaluation du modèle Random Forest.

Ce module transforme les paramètres fournis par l'interface,
entraîne le modèle et génère les graphiques de performance.
"""

from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from ml import (
    _b64,
    _g_confusion,
    _g_rssi,
    _g_metriques,
    C,
    evaluer_predictions,
)


def _g_importance(clf, feature_names):
    # Génère un graphique de l'importance des features du modèle Random Forest.
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(6, 4))

    # ── CORRECTION : appliquer le thème sombre comme les autres graphiques ──
    fig.patch.set_facecolor(C['bg'])
    ax.set_facecolor(C['panel'])

    ax.bar(range(len(importances)), importances[indices], color=C['in'])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(
        [feature_names[i] for i in indices],
        rotation=45, ha='right',
        color=C['text']          # ── CORRECTION : labels en clair
    )
    ax.set_title("Importance des features", color=C['text'])
    ax.tick_params(colors=C['muted'])
    ax.yaxis.label.set_color(C['text'])
    ax.grid(axis='y', linestyle='--', alpha=0.3, color=C['grid'])

    # ── CORRECTION : couleur des graduations de l'axe Y ──
    for spine in ax.spines.values():
        spine.set_edgecolor(C['grid'])

    plt.tight_layout()
    return _b64(fig)


def entrainer_rf(X_train, y_train, params):
    # Transforme les paramètres fournis par l'interface et entraîne un RandomForestClassifier.
    def to_int(val, default):
        try:
            return int(val) if val and str(val).strip() not in ['None', 'null'] else default
        except (ValueError, TypeError):
            return default
    def to_bool(val, default):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            val = val.strip().lower()
            if val in ['true', '1', 'yes']:
                return True
            elif val in ['false', '0', 'no']:
                return False
        return default
    def to_max_features(val, default):
        if val in ['sqrt', 'log2']:
            return val
        try:
            int_val = int(val)
            if int_val > 0:
                return int_val
        except:
            pass
        return default
    def to_criterion(val, default):
        if val in ['gini', 'entropy', 'log_loss']:
            return val
        return default
    def to_class_weight(val, default):
        if isinstance(val, str):
            val = val.strip().lower()
            if val in ['balanced', 'balanced_subsample']:
                return val
            if val == 'none':
                return None
        return default
    def infer_class_weight(val, y_train):
        # Détermine automatiquement class_weight si les classes sont déséquilibrées.
        cw = to_class_weight(val, None)
        if cw is not None:
            return cw
        try:
            labels, counts = np.unique(y_train, return_counts=True)
        except Exception:
            return None
        if len(labels) == 2 and counts.min() / counts.max() < 0.5:
            return 'balanced'
        return None

    class_weight_val = infer_class_weight(params.get('class_weight'), y_train)

    clf = RandomForestClassifier(
        # Random Forest configuré avec les paramètres transformés.
        n_estimators      = to_int(params.get('n_estimators'),      100),
        max_depth         = to_int(params.get('max_depth'),         None),
        min_samples_split = to_int(params.get('min_samples_split'), 2),
        min_samples_leaf  = to_int(params.get('min_samples_leaf'),  1),
        max_features      = to_max_features(params.get('max_features'), 'sqrt'),
        bootstrap         = to_bool(params.get('bootstrap'),        True),
        criterion         = to_criterion(params.get('criterion'),   'gini'),
        class_weight      = class_weight_val,
        oob_score         = to_bool(params.get('oob_score'),        False),
        random_state      = 42,
        n_jobs            = -1
    )
    clf.fit(X_train, y_train)
    return clf


def calculer_resultats_rf(clf, X_train, X_test, y_train, y_test, X_all, y_all, feature_names):
    # Calcule les prédictions, les métriques et les graphiques pour Random Forest.

    y_train_pred = clf.predict(X_train)
    y_test_pred  = clf.predict(X_test)

    try:
        # Validation croisée sur toutes les données pour estimer la stabilité du modèle.
        cv_scores = cross_val_score(clf, X_all, y_all, cv=5, scoring='f1')
        cv_moyen  = cv_scores.mean()
        cv_ecart  = cv_scores.std()
    except:
        cv_moyen, cv_ecart = None, None

    oob = clf.oob_score_ if hasattr(clf, "oob_score_") and clf.oob_score else None

    feature_importance_chart = _g_importance(clf, feature_names)
    confusion_chart          = _g_confusion(y_test, y_test_pred)

    metriques = evaluer_predictions(y_train, y_train_pred, y_test, y_test_pred)
    metriques.update({
        'cv_score' : cv_moyen,
        'cv_ecart' : cv_ecart,
        'oob_score': oob
    })

    return {
        **metriques,
        'n_arbres' : clf.n_estimators,
        'graphiques': {
            'importance'    : feature_importance_chart,
            'confusion'     : confusion_chart,
            'metriques'     : _g_metriques(metriques),
        }
    }