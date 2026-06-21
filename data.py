"""
data.py
-------
Gestion de la base de données SQLite pour l'historique des traitements.

Ce module initialise la base, stocke les résultats des analyses
et fournit les données pour l'affichage ou l'export depuis l'historique.
"""

import sqlite3
import json
from datetime import datetime

DB_NAME = "historique.db"


def init_db():
    # Initialise la base SQLite si elle n'existe pas encore.
    # Crée les tables de modèles, traitements, métriques, paramètres et graphiques.
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS modeles (
            id_modele TEXT PRIMARY KEY,
            nom       TEXT,
            sigle     TEXT
        )
    """)

    c.execute("INSERT OR IGNORE INTO modeles VALUES ('knn',    'K-Nearest Neighbors',    'KNN')")
    c.execute("INSERT OR IGNORE INTO modeles VALUES ('svm',    'Support Vector Machine', 'SVM')")
    c.execute("INSERT OR IGNORE INTO modeles VALUES ('random', 'Random Forest',          'RF')")
    c.execute("INSERT OR IGNORE INTO modeles VALUES ('descriptif', 'Analyse Descriptive',       'DESC')")

    c.execute("""
        CREATE TABLE IF NOT EXISTS traitements (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            date_debut   TEXT,
            date_fin     TEXT,
            duree        REAL,
            id_modele    TEXT,
            fichier      TEXT,
            acc_train    REAL,
            acc_test     REAL,
            nb_epcs      INTEGER,
            nb_valides   INTEGER,
            nb_parasites INTEGER,
            FOREIGN KEY (id_modele) REFERENCES modeles(id_modele)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS descriptif_metrics (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            id_traitement INTEGER,
            read_rate     REAL,
            accuracy      REAL,
            nb_epc_attendus INTEGER,
            nb_epc_lus    INTEGER,
            nb_lectures   INTEGER,
            nb_runs       INTEGER,
            runs_complets INTEGER,
            n_manquants   INTEGER,
            marge_moy     REAL,
            marge_min     REAL,
            marge_max     REAL,
            tags_manquants TEXT,
            FOREIGN KEY (id_traitement) REFERENCES traitements(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS metriques (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            id_traitement INTEGER,
            f1            REAL,
            precision     REAL,
            recall        REAL,
            cv_score      REAL,
            oob_score     REAL,
            n_arbres      INTEGER,
            FOREIGN KEY (id_traitement) REFERENCES traitements(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS parametres (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            id_traitement INTEGER,
            nom           TEXT,
            valeur        TEXT,
            FOREIGN KEY (id_traitement) REFERENCES traitements(id)
        )
    """)


    c.execute("""
    CREATE TABLE IF NOT EXISTS graphiques (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        id_traitement INTEGER,
        nom           TEXT,
        image_base64  TEXT,
        FOREIGN KEY (id_traitement) REFERENCES traitements(id)
    )
""")
    

    conn.commit()
    conn.close()


def save_traitement(result: dict):
    # Enregistre une nouvelle ligne dans l'historique des traitements.
    # Cette fonction découple le traitement des données de l'interface,
    # et stocke aussi bien les métriques ML que les paramètres et graphiques.
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()

    c.execute("""
        INSERT INTO traitements (
            date_debut, date_fin, duree,
            id_modele, fichier,
            acc_train, acc_test,
            nb_epcs, nb_valides, nb_parasites
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result.get("date_debut"),
        result.get("date_fin"),
        result.get("duree"),
        result.get("modele"),
        result.get("fichier"),
        result.get("acc_train",    0),
        result.get("acc_test",     0),
        result.get("nb_epcs",      0),
        result.get("nb_valides",   0),
        result.get("nb_parasites", 0),
    ))

    id_traitement = c.lastrowid

    c.execute("""
        INSERT INTO metriques (
            id_traitement, f1, precision,
            recall, cv_score, oob_score, n_arbres
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        id_traitement,
        result.get("f1"),
        result.get("precision"),
        result.get("recall"),
        result.get("cv_score"),
        result.get("oob_score"),
        result.get("n_arbres"),
    ))

    descriptif_metrics = result.get("descriptif_metrics")
    if not descriptif_metrics and result.get("modele") == 'descriptif':
        descriptif_metrics = {
            'read_rate'      : result.get('read_rate'),
            'accuracy'       : result.get('accuracy'),
            'nb_epc_attendus': result.get('nb_epc_attendus'),
            'nb_epc_lus'     : result.get('nb_epc_lus'),
            'nb_lectures'    : result.get('nb_lectures'),
            'nb_runs'        : result.get('nb_runs'),
            'runs_complets'  : result.get('runs_complets'),
            'n_manquants'    : result.get('n_manquants'),
            'marge_moy'      : result.get('marge_moy'),
            'marge_min'      : result.get('marge_min'),
            'marge_max'      : result.get('marge_max'),
            'tags_manquants' : result.get('tags_manquants', []),
        }

    if descriptif_metrics:
        tags_json = json.dumps(descriptif_metrics.get('tags_manquants', []))
        c.execute("""
            INSERT INTO descriptif_metrics (
                id_traitement, read_rate, accuracy,
                nb_epc_attendus, nb_epc_lus, nb_lectures,
                nb_runs, runs_complets, n_manquants,
                marge_moy, marge_min, marge_max,
                tags_manquants
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            id_traitement,
            descriptif_metrics.get('read_rate'),
            descriptif_metrics.get('accuracy'),
            descriptif_metrics.get('nb_epc_attendus'),
            descriptif_metrics.get('nb_epc_lus'),
            descriptif_metrics.get('nb_lectures'),
            descriptif_metrics.get('nb_runs'),
            descriptif_metrics.get('runs_complets'),
            descriptif_metrics.get('n_manquants'),
            descriptif_metrics.get('marge_moy'),
            descriptif_metrics.get('marge_min'),
            descriptif_metrics.get('marge_max'),
            tags_json,
        ))

    for nom, valeur in result.get("custom_params", {}).items():
        c.execute("""
            INSERT INTO parametres (id_traitement, nom, valeur)
            VALUES (?, ?, ?)
        """, (id_traitement, nom, str(valeur)))


    # Sauvegarder chaque graphique comme une ligne dans la table
    for nom, image_b64 in result.get("graphiques", {}).items():
        if image_b64:  # on vérifie que l'image existe vraiment
            c.execute("""
            INSERT INTO graphiques (id_traitement, nom, image_base64)
            VALUES (?, ?, ?)
        """, (id_traitement, nom, image_b64))


    conn.commit()
    conn.close()


def get_one(traitement_id: int):
    # Récupère toutes les données liées à un traitement pour affichage ou export.
    # Cette fonction reconstitue l'objet complet utilisé par les pages
    # de détail, le téléchargement CSV/PDF et la comparaison d'analyses.
    # On joint la table modeles pour obtenir le nom et le sigle du modèle.
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()

    c.execute("""
        SELECT t.*, m.nom, m.sigle
        FROM traitements t
        JOIN modeles m ON t.id_modele = m.id_modele
        WHERE t.id = ?
    """, (traitement_id,))
    row = c.fetchone()

    c.execute("SELECT * FROM metriques   WHERE id_traitement = ?", (traitement_id,))
    metriques = c.fetchone()

    c.execute("SELECT nom, valeur FROM parametres WHERE id_traitement = ?", (traitement_id,))
    parametres = dict(c.fetchall())

    c.execute("SELECT nom, image_base64 FROM graphiques WHERE id_traitement = ?", (traitement_id,))
    graphiques = dict(c.fetchall())  # ex: {'confusion': 'base64...', 'roc': 'base64...'}

    c.execute("SELECT read_rate, accuracy, nb_epc_attendus, nb_epc_lus, nb_lectures, nb_runs, runs_complets, n_manquants, marge_moy, marge_min, marge_max, tags_manquants FROM descriptif_metrics WHERE id_traitement = ?", (traitement_id,))
    desc_row = c.fetchone()
    descriptif = None
    if desc_row:
        descriptif = {
            'read_rate'      : desc_row[0],
            'accuracy'       : desc_row[1],
            'nb_epc_attendus': desc_row[2],
            'nb_epc_lus'     : desc_row[3],
            'nb_lectures'    : desc_row[4],
            'nb_runs'        : desc_row[5],
            'runs_complets'  : desc_row[6],
            'n_manquants'    : desc_row[7],
            'marge_moy'      : desc_row[8],
            'marge_min'      : desc_row[9],
            'marge_max'      : desc_row[10],
            'tags_manquants' : json.loads(desc_row[11] or '[]'),
        }

    conn.close()
    return row, metriques, parametres, graphiques, descriptif


def get_filtered(q: str = "", modele: str = "", tri: str = "recent") -> list:
    # Recherche et trie les enregistrements de l'historique selon les filtres fournis.
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()

    conditions = []
    params     = []

    if q:
        conditions.append("(t.fichier LIKE ? OR t.id_modele LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like])

    if modele:
        conditions.append("t.id_modele = ?")
        params.append(modele)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    order = {
        "recent"  : "t.id DESC",
        "acc_desc": "t.acc_test DESC",
        "acc_asc" : "t.acc_test ASC",
    }.get(tri, "t.id DESC")

    sql = f"""
        SELECT t.*, m.nom, m.sigle
        FROM traitements t
        JOIN modeles m ON t.id_modele = m.id_modele
        {where}
        ORDER BY {order}
    """

    c.execute(sql, params)
    rows = c.fetchall()
    conn.close()
    return rows


def get_stats() -> dict:
    # Calcule des statistiques globales sur l'historique des traitements.
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()

    c.execute("SELECT COUNT(*) FROM traitements")
    total = c.fetchone()[0]

    if total == 0:
        conn.close()
        return {"total": 0, "meilleur": None, "modele_top": None, "moyenne": None}

    c.execute("SELECT MAX(acc_test) FROM traitements")
    meilleur = c.fetchone()[0]

    c.execute("SELECT AVG(acc_test) FROM traitements")
    moyenne = round(c.fetchone()[0] or 0, 1)

    c.execute("""
        SELECT id_modele, COUNT(*) as cnt
        FROM traitements
        GROUP BY id_modele
        ORDER BY cnt DESC
        LIMIT 1
    """)
    row = c.fetchone()
    modele_top = row[0].upper() if row else None

    conn.close()
    return {
        "total"     : total,
        "meilleur"  : meilleur,
        "modele_top": modele_top,
        "moyenne"   : moyenne,
    }


def delete_one(traitement_id: int):
    # Supprime un traitement et toutes les tables associées pour garder la base propre.
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute("DELETE FROM graphiques WHERE id_traitement = ?", (traitement_id,))
    c.execute("DELETE FROM parametres WHERE id_traitement = ?", (traitement_id,))
    c.execute("DELETE FROM metriques WHERE id_traitement = ?", (traitement_id,))
    c.execute("DELETE FROM descriptif_metrics WHERE id_traitement = ?", (traitement_id,))
    c.execute("DELETE FROM traitements WHERE id = ?", (traitement_id,))
    conn.commit()
    conn.close()