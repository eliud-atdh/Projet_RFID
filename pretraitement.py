"""
pretraitement.py
----------------
Traitement des données RFID brutes et préparation du dataset ML.

Ce module normalise les colonnes, lit les fichiers APTags et supply,
calcule les indicateurs descriptifs, et produit les features par EPC
pour les modèles prédictifs.
"""

import pandas as pd
import os
import glob
from rf import entrainer_rf, calculer_resultats_rf
from ml import (
    graphique_confusion, graphique_roc, graphique_scores,
    evaluer_predictions, entrainer_modele, _split_train_test_epc
)
RSSI_SEUIL = -65


def normaliser_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    # Normalise les noms de colonnes venant de différents exports RFID.
    # Cela permet de traiter les colonnes RSSI/Epc/LogTime de façon uniforme.
    df = df.copy()
    df.columns = df.columns.str.strip()
    mapping = {}
    standard = {
        'rssi': 'RSSI',
        'epc': 'Epc',
        'logtime': 'LogTime',
        'reader': 'Reader',
        'ant': 'Ant',
        'timestamp': 'TimeStamp',
        'emitpower': 'EmitPower',
        'frequency': 'Frequency',
        'latency': 'Latency',
        'reflistid': 'refListId',
        'ciuchstart': 'ciuchStart',
        'ciuchstop': 'ciuchStop',
    }
    lower_to_orig = {col.lower(): col for col in df.columns}
    for name, target in standard.items():
        if name in lower_to_orig:
            mapping[lower_to_orig[name]] = target
    if mapping:
        df = df.rename(columns=mapping)
    return df


def _valider_colonnes(df: pd.DataFrame, required_columns):
    # Vérifie que le DataFrame contient les colonnes indispensables.
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes : {missing}. Colonnes disponibles : {list(df.columns)}"
        )


def lancer_traitement(
    filepath: str = None,
    dossier: str = None,
    tolerance_in: int = 0,
    tolerance_out: int = 0,
    aptags_files: list = None
) -> dict:
    """
    Lance le traitement descriptif RFID.
    Par défaut : comparaison STRICTE t_start <= LogTime <= t_stop (tolerance=0).
    tolerance_in  : marge en secondes ajoutée AVANT t_start
    tolerance_out : marge en secondes ajoutée APRÈS t_stop
    """
    if filepath is not None and not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier introuvable : {os.path.basename(filepath)}")

    if dossier is None:
        if filepath is None:
            raise ValueError("Le dossier doit être spécifié si aucun fichier n'est fourni.")
        dossier = os.path.dirname(os.path.abspath(filepath))

    if aptags_files is None:
        aptags_files = (
            glob.glob(os.path.join(dossier, 'ano_APTags*.csv')) +
            glob.glob(os.path.join(dossier, 'ano_APTags-*.csv')) +
            glob.glob(os.path.join(dossier, '*APTags*.csv'))
        )
        aptags_files = sorted(set(aptags_files))
        if not aptags_files:
            aptags_files = [filepath]
    else:
        aptags_files = [os.path.abspath(f) for f in aptags_files if f]
        if not aptags_files:
            if filepath is None:
                raise ValueError("Aucun fichier APTags fourni.")
            aptags_files = [filepath]

    if filepath is None and aptags_files:
        filepath = aptags_files[0]

    # Lecture et concaténation des fichiers APTags valides.
    dfs_tags = []
    for f in aptags_files:
        try:
            df_tmp = pd.read_csv(f, sep=',')
            df_tmp = normaliser_colonnes(df_tmp)
            if all(c in df_tmp.columns for c in ['RSSI', 'Epc', 'LogTime']):
                dfs_tags.append(df_tmp)
            else:
                pass
        except Exception:
            pass

    if not dfs_tags:
        raise ValueError("Aucun fichier APTags valide trouvé dans le dossier.")

    df_tags = pd.concat(dfs_tags, ignore_index=True)
    df_tags['RSSI'] = pd.to_numeric(df_tags['RSSI'], errors='coerce')
    df_tags['Epc']  = df_tags['Epc'].astype(str).str.strip()
    df_tags = df_tags.dropna(subset=['RSSI']).copy()

    df_tags['LogTime'] = pd.to_datetime(
        df_tags['LogTime'],
        format='%Y-%m-%d-%H:%M:%S',
        errors='coerce'
    )
    df_tags = df_tags.dropna(subset=['LogTime'])

    # 2. Charger reflist_*.olpn
    reflist_files = glob.glob(os.path.join(dossier, 'reflist_*.olpn'))
    if not reflist_files:
        raise ValueError("Aucun fichier reflist_*.olpn trouvé dans data/")

    ref_dict = {}
    for f in reflist_files:
        # Construire un dictionnaire des EPC attendus par reflist.
        ref_id = os.path.splitext(os.path.basename(f))[0]
        df_r   = pd.read_csv(f, sep=',')
        df_r.columns = df_r.columns.str.strip()

        epc_col = None
        for col in df_r.columns:
            if df_r[col].astype(str).str.contains('epc_', case=False).any():
                epc_col = col
                break
        if epc_col is None:
            epc_col = df_r.columns[-1]
        epcs = set(df_r[epc_col].astype(str).str.strip().tolist())
        ref_dict[ref_id] = epcs

    # 3. Charger supply-process
    supply_files = (
        glob.glob(os.path.join(dossier, 'ano_supply-process*.csv')) +
        glob.glob(os.path.join(dossier, 'ano_supplyprocess*.csv')) +
        glob.glob(os.path.join(dossier, '*supply*.csv'))
    )
    supply_files = sorted(set(supply_files))
    if not supply_files:
        raise ValueError("Aucun fichier supply-process (*.csv) trouvé dans le dossier.")

    supply_dfs = []
    for f in supply_files:
        # Lecture des fichiers supply-process et fusion en un seul DataFrame.
        df_s = pd.read_csv(f, sep=',')
        df_s.columns = df_s.columns.str.strip()
        supply_dfs.append(df_s)

    df_supply = pd.concat(supply_dfs, ignore_index=True)

    def parse_ciuch(col):
        return pd.to_datetime(
            col.astype(str).str.replace(',', '.', regex=False),
            format='%d/%m/%Y %H:%M:%S.%f',
            errors='coerce'
        )

    df_supply['t_start']   = parse_ciuch(df_supply['ciuchStart'])
    df_supply['t_stop']    = parse_ciuch(df_supply['ciuchStop'])
    df_supply['refListId'] = df_supply['refListId'].astype(str).str.strip()
    df_supply = df_supply.dropna(subset=['t_start', 't_stop']).reset_index(drop=True)

    nb_runs = len(df_supply)

   
    # 5. Calcul run par run
    # On garde uniquement les lectures avec RSSI suffisant.
    df_tags_filtre = df_tags[df_tags['RSSI'] >= RSSI_SEUIL].copy()

    taux_lecture_runs = []
    runs_complets = 0
    runs_avec_attendus = 0
    tags_manquants_set = set()
    durees = []

    for _, run_row in df_supply.iterrows():
        ref_id = run_row['refListId']
        t_start = run_row['t_start']
        t_stop = run_row['t_stop']

        epc_attendus = ref_dict.get(ref_id, set())

        if not epc_attendus:
            continue

        runs_avec_attendus += 1

        t_s = (
            t_start - pd.Timedelta(seconds=tolerance_in)
            if tolerance_in > 0 else t_start
        )

        t_e = (
            t_stop + pd.Timedelta(seconds=tolerance_out)
            if tolerance_out > 0 else t_stop
        )

        mask = (
            (df_tags_filtre['LogTime'] >= t_s) &
            (df_tags_filtre['LogTime'] <= t_e)
        )

        epc_lus = set(df_tags_filtre.loc[mask, 'Epc'].unique())

        epc_corrects = epc_attendus & epc_lus

        tags_manquants_set |= (epc_attendus - epc_lus)

        taux_lecture_runs.append(
            len(epc_corrects) / len(epc_attendus)
        )

        if epc_corrects == epc_attendus:
            runs_complets += 1

        duree = (t_stop - t_start).total_seconds()
        durees.append(duree)

    # Calcul des indicateurs globaux de performance sur tous les runs.
    read_rate = (
        round(
            100 * sum(taux_lecture_runs) / len(taux_lecture_runs),
            1
        )
        if taux_lecture_runs else 0.0
    )

    accuracy = (
        round(
            100 * runs_complets / runs_avec_attendus,
            1
        )
        if runs_avec_attendus > 0 else 0.0
    )

    marge_moy = (
        round(sum(durees) / len(durees), 2)
        if durees else 0.0
    )

    marge_min = (
        round(min(durees), 2)
        if durees else 0.0
    )

    marge_max = (
        round(max(durees), 2)
        if durees else 0.0
    )

    nb_epc_attendus = sum(len(v) for v in ref_dict.values())
    nb_epc_lus = df_tags_filtre['Epc'].nunique()
    nb_lectures = len(df_tags_filtre)

    n_manquants = len(tags_manquants_set)

    tags_manquants = [
        {'Epc': e}
        for e in sorted(tags_manquants_set)
    ]

    return {
        'fichier': os.path.basename(filepath),
        'accuracy': accuracy,
        'read_rate': read_rate,
        'nb_epc_attendus': nb_epc_attendus,
        'nb_epc_lus': nb_epc_lus,
        'nb_lectures': nb_lectures,
        'nb_runs': nb_runs,
        'n_manquants': n_manquants,
        'tags_manquants': tags_manquants,
        'runs_complets': runs_complets,
        'marge_moy': marge_moy,
        'marge_min': marge_min,
        'marge_max': marge_max,
        'df_tags': df_tags,
        'df_supply': df_supply,
        'ref_dict': ref_dict,
    }


def preparer_donnees(df_tags, ref_dict, df_supply, tolerance_in: int = 0, tolerance_out: int = 0, min_lectures_out: int = 10):
    """
    Construit le dataset ML (features par EPC).
    Applique la tolérance temporelle après correction du décalage horaire.

    Args:
        tolerance_in  : marge en secondes ajoutée AVANT t_start
        tolerance_out : marge en secondes ajoutée APRÈS t_stop
        min_lectures_out : EPCs OUT (parasites) avec < N lectures sont supprimés (filtrage du bruit)
    """
    if df_tags.empty or df_supply.empty:
        raise ValueError("Données tags ou supply vides.")

    tags_min   = df_tags['LogTime'].min()
    supply_min = df_supply['t_start'].min()

    # Correction automatique du décalage horaire
    if abs((supply_min - tags_min).total_seconds()) > 300:
        decalage = supply_min - tags_min
        df_tags['LogTime'] += decalage
    lignes = []

    for idx, run in df_supply.iterrows():
        ref_id  = str(run['refListId']).strip()
        t_start = run['t_start']
        t_stop  = run['t_stop']

        epcs_attendus = ref_dict.get(ref_id, set())
        if not epcs_attendus:
            continue

        t_start_tol = t_start - pd.Timedelta(seconds=tolerance_in)
        t_stop_tol  = t_stop  + pd.Timedelta(seconds=tolerance_out)

        mask   = (df_tags['LogTime'] >= t_start_tol) & (df_tags['LogTime'] <= t_stop_tol)
        df_run = df_tags.loc[mask].copy()

        if df_run.empty:
            continue

        feats = (
            df_run.groupby("Epc")
            .apply(lambda g: pd.Series({
                "RSSI_moyen":  g["RSSI"].mean(),
                "RSSI_max":    g["RSSI"].max(),
                "RSSI_min":    g["RSSI"].min(),
                "RSSI_ecart":  g["RSSI"].max() - g["RSSI"].min(),
                "nb_lectures": len(g),
                "nb_antennes": g["Ant"].nunique() if "Ant" in g.columns else 1,
            }))
            .reset_index()
        )

        feats["target"] = feats["Epc"].apply(lambda e: 1 if e in epcs_attendus else 0)
        lignes.append(feats)

    if not lignes:
        raise ValueError("Toujours aucune correspondance après correction de décalage.")

    feats_all    = pd.concat(lignes, ignore_index=True)
    
    # Filtrage du bruit : supprimer les EPCs OUT (parasites) avec peu de lectures
    feats_all = feats_all[
        (feats_all["target"] == 1) | (feats_all["nb_lectures"] >= min_lectures_out)
    ].copy()
    
    nb_valides   = int((feats_all["target"] == 1).sum())
    nb_parasites = int((feats_all["target"] == 0).sum())

    X = feats_all[["Epc", "RSSI_moyen", "RSSI_max", "RSSI_min", "RSSI_ecart", "nb_lectures", "nb_antennes"]].copy()
    X = X.fillna(-100)
    y = feats_all["target"]

    return X, y, feats_all

def preparer_et_entrainer(modele_id: str, input_path: str, custom_params=None,
                          dossier_ref: str = None, aptags_files: list = None):
    from pretraitement import preparer_donnees

    if not os.path.isfile(input_path):
        raise ValueError(f"Le chemin doit pointer vers un fichier CSV : {input_path}")

    custom_params = dict(custom_params or {})

    tolerance_in  = int(custom_params.pop('tolerance_in',  0))
    tolerance_out = int(custom_params.pop('tolerance_out', 0))

    if dossier_ref is None:
        dossier_ref = os.path.dirname(input_path)

    result_desc = lancer_traitement(
        input_path,
        dossier=dossier_ref,
        tolerance_in=tolerance_in,
        tolerance_out=tolerance_out,
        aptags_files=aptags_files
    )

    X, y, feats_all = preparer_donnees(
        result_desc['df_tags'],
        result_desc['ref_dict'],
        result_desc['df_supply'],
        tolerance_in=tolerance_in,
        tolerance_out=tolerance_out
    )

    feature_cols = ["RSSI_moyen", "RSSI_max", "RSSI_min", "RSSI_ecart",
                    "nb_lectures", "nb_antennes"]

    X_ml = X[feature_cols].copy()

    X_train, X_test, y_train, y_test = _split_train_test_epc(X, y)

    descriptif = {
        "read_rate"      : result_desc.get("read_rate"),
        "accuracy"       : result_desc.get("accuracy"),
        "nb_epc_attendus": result_desc.get("nb_epc_attendus"),
        "nb_epc_lus"     : result_desc.get("nb_epc_lus"),
        "nb_lectures"    : result_desc.get("nb_lectures"),
        "nb_runs"        : result_desc.get("nb_runs"),
        "runs_complets"  : result_desc.get("runs_complets"),
        "n_manquants"    : result_desc.get("n_manquants"),
        "marge_moy"      : result_desc.get("marge_moy"),
        "marge_min"      : result_desc.get("marge_min"),
        "marge_max"      : result_desc.get("marge_max"),
    }

    if modele_id == "random":
        clf = entrainer_rf(X_train, y_train, custom_params)
        resultats = calculer_resultats_rf(
            clf, X_train, X_test, y_train, y_test,
            X_ml, y, feature_cols
        )
        resultats["modele"]       = "random"
        resultats["nb_epcs"]      = len(X)
        resultats["nb_valides"]   = int((y == 1).sum())
        resultats["nb_parasites"] = int((y == 0).sum())
        resultats["descriptif"]   = descriptif
        return resultats

    else:
        model, scaler = entrainer_modele(modele_id, X_train, y_train, custom_params)

        X_train_scaled = scaler.transform(X_train)
        X_test_scaled  = scaler.transform(X_test)
        y_test_pred    = model.predict(X_test_scaled)
        y_train_pred   = model.predict(X_train_scaled)

        stats = evaluer_predictions(y_train, y_train_pred, y_test, y_test_pred)

        resultats = {
            "modele"      : modele_id,
            "acc_test"    : stats['acc_test'],
            "acc_train"   : stats['acc_train'],
            "ecart"       : stats['ecart'],
            "f1"          : stats['f1'],
            "precision"   : stats['precision'],
            "recall"      : stats['recall'],
            "nb_epcs"     : len(X),
            "nb_valides"  : int((y == 1).sum()),
            "nb_parasites": int((y == 0).sum()),
            "graphiques"  : {
                "confusion": graphique_confusion(y_test, y_test_pred),
                "roc"      : graphique_roc(model, X_test_scaled, y_test),
                "scores"   : graphique_scores(model, X_test_scaled, y_test),
            },
            "descriptif"  : descriptif,
        }
        return resultats
    


def lancer_analyse_complete(nom_dossier, tolerance_in, tolerance_out, modele_id, custom_params, upload_folder):
    import os
    
    dossier_path = os.path.join(upload_folder, nom_dossier)
    
    aptags = sorted([
        f for f in os.listdir(dossier_path)
        if 'APTags' in f and f.endswith('.csv')
    ])
    
    if not aptags:
        raise ValueError(f"Aucun fichier APTags trouvé dans {nom_dossier}")
    
    premier_aptag = os.path.join(dossier_path, aptags[0])

    # Conversion des types
    for k, v in custom_params.items():
        try:
            custom_params[k] = float(v) if '.' in str(v) else int(v)
        except (ValueError, TypeError):
            pass

    # 1. Analyse descriptive
    resultat_descriptif = lancer_traitement(
        premier_aptag,
        dossier=dossier_path,
        tolerance_in=tolerance_in,
        tolerance_out=tolerance_out
    )

    # 2. Analyse prédictive
    custom_params['tolerance_in']  = tolerance_in
    custom_params['tolerance_out'] = tolerance_out

    resultat_predictif = preparer_et_entrainer(
        modele_id,
        premier_aptag,
        custom_params,
        dossier_ref=dossier_path
    )

    return resultat_descriptif, resultat_predictif