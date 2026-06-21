"""
modeles.py
----------
Définition des modèles ML disponibles (KNN, SVM, Random Forest).
Chaque modèle expose : nom, description, paramètres avec valeurs par défaut,
types, contraintes min/max et options pour les selects.

La configuration est utilisée par l'interface pour générer les forms
et par le traitement prédictif pour construire les modèles.

BLOC 1 : Ajout de paramètres avancés pour KNN et SVM
  - KNN : algorithm, leaf_size, p (exposant Minkowski)
  - SVM : max_iter, class_weight, decision_function_shape
"""

# Liste des modèles disponibles et de leurs paramètres.
MODELES = {

    
    # KNN — K-Nearest Neighbors
    "knn": {
        "id": "knn",
        "nom": "K-Nearest Neighbors",
        "sigle": "KNN",
        "description": (
            "Classifie chaque EPC selon les k lectures RFID les plus proches "
            "en termes de RSSI. Simple, interprétable, mais sensible au bruit."
        ),
        "couleur": "#00d4ff",
        "params": {

            #Paramètres de base
            "n_neighbors": {
                "label": "Nombre de voisins (k)",
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 50,
            },

            "metric": {
                "label": "Métrique de distance",
                "type": "select",
                "default": "euclidean",
                "options": ["euclidean", "manhattan", "minkowski"],
            },

            "weights": {
                "label": "Pondération des voisins",
                "type": "select",
                "default": "uniform",
                "options": ["uniform", "distance"],
            },

            #Paramètres avancés KNN
            "algorithm": {
                "label": "Algorithme de recherche",
                "type": "select",
                "default": "auto",
                "options": ["auto", "ball_tree", "kd_tree", "brute"],
            },

            
            "p": {
                "label": "Exposant Minkowski (p)",
                "type": "int",
                "default": 2,
                "min": 1,
                "max": 5,
            },

            "n_jobs": {
                "label": "Parallélisation (n_jobs)",
                "type": "int",
                "default": -1,
                "min": -1,
                "max": 16,
            },
        },
    },

    # SVM — Support Vector Machine

    "svm": {
        "id": "svm",
        "nom": "Support Vector Machine",
        "sigle": "SVM",
        "description": (
            "Sépare les lectures IN/OUT par un hyperplan optimal dans l'espace "
            "des features RSSI. Robuste, efficace sur données non linéaires."
        ),
        "couleur": "#ffab40",
        "params": {

            #Paramètres de base
            "C": {
                "label": "Paramètre de régularisation (C)",
                "type": "float",
                "default": 1.0,
                "min": 0.01,
                "max": 100.0,
            },

            "kernel": {
                "label": "Noyau (kernel)",
                "type": "select",
                "default": "rbf",
                "options": ["rbf", "linear", "poly", "sigmoid"],
            },

            "gamma": {
                "label": "Gamma",
                "type": "select",
                "default": "scale",
                "options": ["scale", "auto"],
            },

            #Paramètres avancés SVM
            "max_iter": {
                "label": "Itérations maximum",
                "type": "int",
                "default": -1,
                "min": -1,
                "max": 10000,
            },

            "class_weight": {
                "label": "Pondération des classes",
                "type": "select",
                "default": "none",
                "options": ["none", "balanced"],
            },

            "decision_function_shape": {
                "label": "Forme de la fonction de décision",
                "type": "select",
                "default": "ovr",
                "options": ["ovr", "ovo"],
            },

            "degree": {
                "label": "Degré polynôme (kernel=poly)",
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 10,
            },

            "tol": {
                "label": "Tolérance convergence (tol)",
                "type": "float",
                "default": 0.001,
                "min": 0.0001,
                "max": 0.1,
            },

            "probability": {
                "label": "Activer predict_proba (courbe ROC)",
                "type": "select",
                "default": "true",
                "options": ["true", "false"],
            },
        },
    },

    # Analyse Descriptive
    "descriptif": {
        "id": "descriptif",
        "nom": "Analyse Descriptive",
        "sigle": "DESC",
        "description": (
            "Lance uniquement l'analyse descriptive RFID sur les données sélectionnées, "
            "sans entraînement de modèle prédictif."
        ),
        "couleur": "#7c3aed",
        "params": {
        },
    },


    # Random Forest
    "random": {
        "id": "random",
        "nom": "Random Forest",
        "sigle": "RF",
        "description": (
            "Combinaison de plusieurs arbres de décision entraînés sur des "
            "sous-échantillons aléatoires. Très robuste au bruit RFID. "
            "Fournit les importances des variables."
        ),
        "couleur": "#5d40ff",
        "params": {

            "n_estimators": {
                "label": "Nombre d'arbres",
                "type": "int",
                "default": 100,
                "min": 10,
                "max": 500,
            },

            "max_depth": {
                "label": "Profondeur maximale des arbres",
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 100,
            },

            "min_samples_split": {
                "label": "Échantillons min pour diviser un nœud",
                "type": "int",
                "default": 2,
                "min": 2,
                "max": 20,
            },

            "min_samples_leaf": {
                "label": "Échantillons min dans une feuille",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 20,
            },

            "max_features": {
                "label": "Features par split (max_features)",
                "type": "select",
                "default": "sqrt",
                "options": ["sqrt", "log2"],
            },

            "criterion": {
                "label": "Critère de qualité",
                "type": "select",
                "default": "gini",
                "options": ["gini", "entropy", "log_loss"],
            },

            "bootstrap": {
                "label": "Tirage avec remise (bootstrap)",
                "type": "select",
                "default": "true",
                "options": ["true", "false"],
            },

            "oob_score": {
                "label": "Score Out-of-Bag (oob_score)",
                "type": "select",
                "default": "false",
                "options": ["true", "false"],
            },

            "class_weight": {
                "label": "Pondération des classes",
                "type": "select",
                "default": "none",
                "options": ["none", "balanced"],
            },

            "n_jobs": {
                "label": "Parallélisation (n_jobs)",
                "type": "int",
                "default": -1,
                "min": -1,
                "max": 16,
            },
        },
    },
}


#Fonctions utilitaire

def get_modele(model_id: str) -> dict:
    """Retourne un modèle par son id."""
    return MODELES.get(model_id)


def get_tous_modeles() -> list:
    """Retourne la liste de tous les modèles."""
    return list(MODELES.values())


def detail_modele(model_id: str) -> dict:
    """Retourne les détails d'un modèle."""
    return MODELES.get(model_id)


def construire_models_config():
    # Prépare une configuration JSON-friendly des modèles pour le front-end.
    config = {}

    for model_id, info in MODELES.items():

        params_list = []

        for param_name, param_info in info["params"].items():

            p = {
                "name": param_name,
                "label": param_info["label"],
                "default": param_info["default"],
            }

            if param_info["type"] in ("int", "float"):

                p["type"] = "number"
                p["min"] = param_info.get("min", 0)
                p["max"] = param_info.get("max", 9999)

            else:

                p["type"] = "select"
                p["options"] = [
                    {
                        "value": o,
                        "label": str(o),
                    }
                    for o in param_info.get("options", [])
                ]

            params_list.append(p)

        #Paramètres communs
        params_list += [
            {
                "name": "tolerance_in",
                "label": "Marge temporelle IN (s)",
                "default": 0,
                "type": "number",
                "min": 0,
                "max": 600,
            },
            {
                "name": "tolerance_out",
                "label": "Marge temporelle OUT (s)",
                "default": 0,
                "type": "number",
                "min": 0,
                "max": 600,
            },
        ]

        config[model_id] = {
            "label": info["nom"],
            "sigle": info["sigle"],
            "color": info["couleur"],
            "params": params_list,
        }

    return config
