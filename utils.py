"""
utils.py
--------
Fonctions utilitaires pour la gestion des fichiers et métadonnées.

Ce module fournit :
- la validation des uploads,
- la navigation dans le dossier data/,
- la récupération des informations de taille/date des fichiers.
"""

import os
from datetime import datetime

# Dossier principal de stockage des datasets importés.
UPLOAD_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data'
)

# Taille maximale autorisée pour un fichier uploadé : 10 Mo.
TAILLE_MAX_OCTETS = 10 * 1024 * 1024

# On crée le dossier si nécessaire au démarrage.
os.makedirs(UPLOAD_FOLDER, exist_ok=True)




def _fmt_taille(octets: int) -> str:
    # Convertit un nombre d'octets en chaîne lisible pour l'utilisateur.
    if octets < 1024:
        return f"{octets} B"
    elif octets < 1024 * 1024:
        return f"{octets / 1024:.1f} KB"
    return f"{octets / (1024 * 1024):.1f} MB"


def get_metadonnees(filename):
    # Retourne le nom, la taille et la date de modification d'un fichier.
    chemin = os.path.join(UPLOAD_FOLDER, filename)
    taille_octets = os.path.getsize(chemin)
    if taille_octets < 1024:
        taille = f"{taille_octets} B"
    elif taille_octets < 1024 * 1024:
        taille = f"{taille_octets / 1024:.1f} KB"
    else:
        taille = f"{taille_octets / (1024 * 1024):.1f} MB"
    date = datetime.fromtimestamp(
        os.path.getmtime(chemin)
    ).strftime('%d/%m/%Y %H:%M')
    return {'nom': filename, 'taille': taille, 'date': date}




def get_dossiers() -> list:
    # Liste tous les dossiers de datasets disponibles dans data/.
    # Retourne le nombre de fichiers, le nombre de CSV APTags et la taille totale.
    try:
        dossiers = []
        for entree in sorted(os.listdir(UPLOAD_FOLDER)):
            if entree.startswith('.') or entree.startswith('temp_'):
                continue
            chemin = os.path.join(UPLOAD_FOLDER, entree)
            if not os.path.isdir(chemin):
                continue

            fichiers_internes = [
                f for f in os.listdir(chemin)
                if not f.startswith('.') and os.path.isfile(os.path.join(chemin, f))
            ]
            nb_total   = len(fichiers_internes)
            nb_csv     = sum(1 for f in fichiers_internes if 'APTags' in f)
            taille_tot = sum(
                os.path.getsize(os.path.join(chemin, f)) for f in fichiers_internes
            )
            date = datetime.fromtimestamp(
                os.path.getmtime(chemin)
            ).strftime('%d/%m/%Y %H:%M')

            dossiers.append({
                'nom'     : entree,
                'date'    : date,
                'nb_csv'  : nb_csv,
                'nb_total': nb_total,
                'taille'  : _fmt_taille(taille_tot),
            })
        return dossiers
    except Exception:
        return []
    

def get_fichiers_dossier(nom_dossier: str) -> list:
    # Retourne la liste des fichiers contenus dans un dossier sélectionné.
    # Les fichiers sont triés par date de modification décroissante.
    try:
        chemin = os.path.join(UPLOAD_FOLDER, nom_dossier)
        if not os.path.isdir(chemin):
            return []

        fichiers_bruts = [
            f for f in os.listdir(chemin)
            if not f.startswith('.')
            and os.path.isfile(os.path.join(chemin, f))
        ]
        fichiers_bruts.sort(
            key=lambda f: os.path.getmtime(os.path.join(chemin, f)),
            reverse=True
        )

        resultats = []
        for fichier in fichiers_bruts:
            chemin_f = os.path.join(chemin, fichier)
            resultats.append({
                'nom'    : f"{nom_dossier}/{fichier}",
                'base'   : fichier,
                'taille' : _fmt_taille(os.path.getsize(chemin_f)),
                'date'   : datetime.fromtimestamp(
                    os.path.getmtime(chemin_f)
                ).strftime('%d/%m/%Y %H:%M'),
            })
        return resultats
    except Exception:
        return []







def get_aptags_dossier(nom_dossier: str) -> list:
    # Filtre les fichiers d'un dossier pour ne renvoyer que ceux contenant APTags.
    return [
        f for f in get_fichiers_dossier(nom_dossier)
        if 'APTags' in f['base']
    ]






def get_metadata_all() -> list:
    try:
        resultats = []
        for entree in sorted(os.listdir(UPLOAD_FOLDER)):
            if entree.startswith('.') or entree.startswith('temp_'):
                continue
            chemin_entree = os.path.join(UPLOAD_FOLDER, entree)
            if os.path.isfile(chemin_entree):
                resultats.append(get_metadonnees(entree))
            elif os.path.isdir(chemin_entree):
                for fichier in sorted(os.listdir(chemin_entree)):
                    if fichier.startswith('.'):
                        continue
                    chemin_f = os.path.join(chemin_entree, fichier)
                    if not os.path.isfile(chemin_f):
                        continue
                    resultats.append({
                        'nom'    : f"{entree}/{fichier}",
                        'base'   : fichier,
                        'taille' : _fmt_taille(os.path.getsize(chemin_f)),
                        'date'   : datetime.fromtimestamp(
                            os.path.getmtime(chemin_f)
                        ).strftime('%d/%m/%Y %H:%M'),
                        'dossier': entree,
                    })
        return resultats
    except Exception:
        return []
    




def valider_fichier(file):
    # Valide que le fichier est au format CSV et que sa taille est inférieure à 10 Mo.
    if not file.filename.endswith('.csv'):
        return False, "Format invalide : seuls les fichiers .csv sont acceptes."
    file.seek(0, os.SEEK_END)
    taille = file.tell()
    file.seek(0)
    if taille > TAILLE_MAX_OCTETS:
        return False, "Fichier trop volumineux : 10 MB max."
    return True, None





def sauvegarder_fichier(file):
    # Sauvegarde le fichier dans data/ si le nom n'est pas déjà utilisé.
    chemin = os.path.join(UPLOAD_FOLDER, file.filename)
    if os.path.exists(chemin):
        return False, f"Le fichier existe deja : {file.filename}"
    file.save(chemin)
    return True, None
