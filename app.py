"""
app.py
------
Point d'entrée de l'application Flask RFID Analytics.

Ce fichier contient les routes Flask principales pour :
- l'upload et la gestion des datasets,
- l'analyse descriptive des fichiers RFID,
- l'analyse prédictive par batch,
- la consultation et l'export de l'historique.

Il gère aussi un petit store de progression pour envoyer l'état
aux clients via SSE lors des traitements longs.
"""

from flask import (
    Flask,
    render_template,
    request,
    session,
    redirect,
    jsonify,
    Response,
    stream_with_context
)

import pandas as pd
import os
import glob
import json
import re
import threading
import tempfile
from datetime import datetime
from modeles import construire_models_config
from pretraitement import preparer_et_entrainer


import pretraitement
import modeles as modeles_module
import data

from rf import (
    entrainer_rf,
    calculer_resultats_rf
)

from ml import (
    graphique_erreur_vs_k,
    graphique_confusion,
    graphique_roc,
    graphique_scores,
    evaluer_predictions,
    entrainer_modele,
    gridsearch_svm,
    crossval_svm,
    _split_train_test_epc,
)

from pretraitement import (
    lancer_traitement,
    preparer_donnees
)

from modeles import (
    detail_modele as get_detail_modele
)


from utils import (
    valider_fichier,
    get_dossiers,
    get_fichiers_dossier,
    get_aptags_dossier,
    get_metadata_all,
    get_metadonnees,
    _fmt_taille,
    UPLOAD_FOLDER,
)


# INITIALISATION

app = Flask(__name__)
app.secret_key = 'rfid_secret_key'

# Initialisation de la base de données SQLite locale pour l'historique.
data.init_db()

# CONFIGURATION


BATCH_TEMP_DIR = os.path.join(tempfile.gettempdir(), 'projet_s8_batches')
TAILLE_MAX_OCTETS = 10 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(BATCH_TEMP_DIR, exist_ok=True)


def _batch_temp_path(batch_id):
    # Chemin local dans lequel on sauvegarde l'état d'un batch en cours.
    return os.path.join(BATCH_TEMP_DIR, f"{batch_id}.json")


def _save_batch_state(batch_id, payload):
    try:
        with open(_batch_temp_path(batch_id), 'w', encoding='utf-8') as f:
            json.dump(payload, f)
    except Exception:
        pass


def _load_batch_state(batch_id):
    try:
        with open(_batch_temp_path(batch_id), 'r', encoding='utf-8') as f:
            payload = json.load(f)
        experiences = payload.get('experiences', {})
        return {
            'experiences': {int(k): v for k, v in experiences.items()},
            'resultats': payload.get('resultats', []),
        }
    except Exception:
        return None

# MODELS_CONFIG — Format attendu par le JavaScript


MODELS_CONFIG = construire_models_config()

# STORE DE PROGRESSION (Server-Sent Events)

_progression_store = {}
_progression_lock = threading.Lock()


def _push_evt(sid, etape, pct, statut='en_cours'):
    # Ajoute un événement de progression pour le client SSE.
    with _progression_lock:
        if sid not in _progression_store:
            _progression_store[sid] = []
        _progression_store[sid].append({
            'etape': etape,
            'pct'  : pct,
            'statut': statut
        })


def _pop_evts(sid):
    with _progression_lock:
        evts = _progression_store.pop(sid, [])
    return evts



# ROUTES — PAGES PRINCIPALES
# -------------------------
# Dans cette section, on regroupe les routes publiques de l'application.
# Les routes servent les pages d'accueil, les uploads, la gestion
# des datasets, l'analyse descriptive et les pages de modèle/historique.

# Flux SSE pour envoyer les étapes de progression côté client.
@app.route('/progression/<sid>')
def progression_stream(sid):
    import time
    def generate():
        deadline = time.time() + 300
        while time.time() < deadline:
            evts = _pop_evts(sid)
            for evt in evts:
                yield f"data: {json.dumps(evt)}\n\n"
                if evt['statut'] in ('termine', 'erreur'):
                    return
            time.sleep(0.4)
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


# Page d'accueil du site, affiche les dossiers de données disponibles.
@app.route('/')
def index():
    return render_template('index.html', dossiers=get_dossiers())


# Route pour uploader un seul fichier CSV vers un dossier existant.
@app.route('/upload', methods=['POST'])
def upload():
    file         = request.files.get('file')
    dossier_dest = request.form.get('dossier_dest', '').strip()

    if not file or file.filename == '':
        return render_template('index.html', dossiers=get_dossiers(),
                               erreur="Aucun fichier sélectionné.")

    if not dossier_dest:
        return render_template('index.html', dossiers=get_dossiers(),
                               erreur="Sélectionnez un dossier de destination avant d'importer.")

    valide, message = valider_fichier(file)
    if not valide:
        return render_template('index.html', dossiers=get_dossiers(), erreur=message)

    chemin_dossier = os.path.join(UPLOAD_FOLDER, dossier_dest)
    if not os.path.isdir(chemin_dossier):
        return render_template('index.html', dossiers=get_dossiers(),
                               erreur=f"Dossier introuvable : {dossier_dest}")

    chemin = os.path.join(chemin_dossier, file.filename)
    if os.path.exists(chemin):
        return render_template('index.html', dossiers=get_dossiers(),
                               erreur=f"Ce fichier existe déjà dans « {dossier_dest} ».")

    file.save(chemin)
    return render_template(
        'index.html',
        dossiers=get_dossiers(),
        succes=f"'{file.filename}' ajouté dans « {dossier_dest} » avec succès."
    )


# Route pour uploader plusieurs fichiers en une seule fois comme un dossier.
@app.route('/upload-dossier', methods=['POST'])
def upload_dossier():
    fichiers = request.files.getlist('dossier')

    if not fichiers or all(f.filename == '' for f in fichiers):
        return render_template('index.html', erreur="Aucun fichier reçu dans le dossier.")

    premier_chemin = fichiers[0].filename
    nom_dossier    = premier_chemin.split('/')[0] if '/' in premier_chemin else 'import'

    horodatage   = datetime.now().strftime('%Y%m%d_%H%M%S')
    nom_final    = f"{nom_dossier}_{horodatage}"
    dossier_dest = os.path.join(UPLOAD_FOLDER, nom_final)
    os.makedirs(dossier_dest, exist_ok=True)

    extensions_ok = {'.csv', '.olpn'}
    nb_ok         = 0
    nb_ignore     = 0
    erreurs_save  = []

    for f in fichiers:
        if not f.filename:
            continue

        nom_fichier = os.path.basename(f.filename)
        ext         = os.path.splitext(nom_fichier)[1].lower()

        if ext not in extensions_ok:
            nb_ignore += 1
            continue

        dest = os.path.join(dossier_dest, nom_fichier)

        try:
            f.save(dest)
            nb_ok += 1
        except Exception as e:
            erreurs_save.append(f"{nom_fichier} : {str(e)}")

    if nb_ok == 0:
        import shutil
        shutil.rmtree(dossier_dest, ignore_errors=True)
        return render_template(
            'index.html',
            erreur="Aucun fichier valide (.csv ou .olpn) trouvé dans le dossier importé."
        )

    msg = f"Dossier '{nom_final}' importé : {nb_ok} fichier(s)."
    if nb_ignore:
        msg += f" ({nb_ignore} ignoré(s) — extension non supportée)"
    if erreurs_save:
        msg += f" Erreurs : {', '.join(erreurs_save)}"

    return render_template('index.html', succes=msg)



# ROUTES — DATASETS


# Affiche la page de gestion des datasets et des dossiers importés.
@app.route('/datasets')
def list_files():
    return render_template('datasets.html', dossiers=get_dossiers())


@app.route('/datasets/dossier/<nom_dossier>')
def voir_dossier(nom_dossier):
    chemin = os.path.join(UPLOAD_FOLDER, nom_dossier)
    if not os.path.isdir(chemin):
        return render_template('datasets.html', dossiers=get_dossiers(),
                               erreur=f"Dossier introuvable : {nom_dossier}")
    fichiers = get_fichiers_dossier(nom_dossier)
    return render_template('datasets_dossier.html',
                           nom_dossier=nom_dossier,
                           fichiers=fichiers)


# Affiche le contenu d'un fichier CSV ou OLPN sélectionné.
@app.route('/datasets/<path:filename>')
def show_file(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(path):
        return render_template(
            'datasets.html',
            dossiers=get_dossiers(),
            erreur="Fichier non trouvé."
        )

    ext = os.path.splitext(path)[1].lower()

    try:
        if ext == '.csv':
            df = pd.read_csv(path)
            table = df.head(50).to_html(classes='data', index=False, border=0)
            return render_template('file.html', table=table, filename=filename)

        elif ext == '.olpn':
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lignes = [l.strip() for l in f.readlines() if l.strip()]
            df = pd.DataFrame({'EPC': lignes})
            table = df.to_html(classes='data', index=False, border=0)
            return render_template('file.html', table=table, filename=filename)

        else:
            return render_template(
                'datasets.html',
                dossiers=get_dossiers(),
                erreur=f"Format non supporté : {ext}"
            )

    except Exception as e:
        return render_template(
            'datasets.html',
            dossiers=get_dossiers(),
            erreur=f"Erreur lecture fichier : {str(e)}"
        )


@app.route('/select', methods=['POST'])
def select_file():
    filename = request.form.get('filename')
    session['fichier_selectionne'] = filename
    return redirect('/datasets')



# ROUTE — TRAITEMENT ANALYTIQUE (analyse descriptive)


# Lance l'analyse descriptive des fichiers ou dossiers sélectionnés.
@app.route('/run', methods=['POST'])
def run():
    nom_dossier   = request.form.get('dossier', '').strip()
    fichiers_form = request.form.getlist('fichiers')
    tolerance_in  = int(request.form.get('tolerance_in',  0))
    tolerance_out = int(request.form.get('tolerance_out', 0))

    if nom_dossier:
        dossier_path = os.path.join(UPLOAD_FOLDER, nom_dossier)
        if not os.path.isdir(dossier_path):
            return render_template('analyse_descriptive.html',
                                   dossiers=get_dossiers(),
                                   erreur=f"Dossier introuvable : {nom_dossier}")

        aptags = [
            f for f in os.listdir(dossier_path)
            if 'APTags' in f and f.endswith('.csv')
        ]
        if not aptags:
            return render_template('analyse_descriptive.html',
                                   dossiers=get_dossiers(),
                                   erreur="Aucun fichier APTags trouvé dans ce dossier.")

        premier_aptag = os.path.join(dossier_path, sorted(aptags)[0])
        try:
            result = lancer_traitement(
                premier_aptag,
                dossier=dossier_path,
                tolerance_in=tolerance_in,
                tolerance_out=tolerance_out
            )
            result['fichier'] = nom_dossier
            return render_template('traitement.html', resultats=[result])
        except Exception as e:
            return render_template('analyse_descriptive.html',
                                   dossiers=get_dossiers(),
                                   erreur=f"Erreur traitement : {str(e)}")

    elif fichiers_form:
        session['fichier_selectionne']   = fichiers_form[0]
        session['fichiers_selectionnes'] = fichiers_form
        try:
            aptag_paths = []
            for f in fichiers_form:
                fp = os.path.join(UPLOAD_FOLDER, f)
                if not os.path.exists(fp):
                    raise FileNotFoundError(f"Introuvable : {f}")
                aptag_paths.append(fp)

            result = lancer_traitement(
                aptag_paths[0],
                dossier=UPLOAD_FOLDER,
                tolerance_in=tolerance_in,
                tolerance_out=tolerance_out,
                aptags_files=aptag_paths
            )
            result['fichier'] = fichiers_form[0] if len(fichiers_form) == 1 else f"{len(fichiers_form)} fichiers"
            return render_template('traitement.html', resultats=[result])
        except Exception as e:
            return render_template('datasets.html', dossiers=get_dossiers(),
                                   erreur=f"Erreur traitement : {str(e)}")

    else:
        return render_template('analyse_descriptive.html',
                               dossiers=get_dossiers(),
                               erreur="Sélectionnez au moins un dossier.")

# ROUTES — MODELES
# Page qui liste les modèles prédictifs disponibles et les métadonnées.
@app.route('/modeles')
def page_modeles():
    return render_template('modeles.html',
                           modeles=modeles_module.get_tous_modeles(),
                           metadata=get_metadata_all())




@app.route('/analyse_predictive')
@app.route('/analyse-predictive')
def analyse_predictive():
    return render_template(
        'analyse_predictive.html',
        dossiers=get_dossiers(),
        MODELS_CONFIG=MODELS_CONFIG
    )


@app.route('/modeles/selectionner', methods=['POST'])
def selectionner_modele():
    modele_id = request.form.get('modele')
    next_page = request.form.get('next') or request.referrer or '/modeles'
    if modele_id:
        session['modele_selectionne'] = modele_id
    return redirect(next_page)


@app.route('/detail_modele/<model_id>')
def afficher_detail_modele(model_id):
    modele = get_detail_modele(model_id)
    if not modele:
        return "Modele introuvable", 404
    return render_template('detail_modele.html', modele=modele)



# ROUTES — HISTORIQUE


# Page de l'historique pour afficher les traitements enregistrés.
@app.route('/historique')
def historique():
    q      = request.args.get('q',      '').strip()
    modele = request.args.get('modele', '').strip()
    tri    = request.args.get('tri',    'recent').strip()
    rows   = data.get_filtered(q=q, modele=modele, tri=tri)
    stats  = data.get_stats()
    return render_template(
        'historique.html',
        rows=rows, stats=stats, q=q, modele=modele, tri=tri
    )

@app.route('/historique/export-selection', methods=['POST'])
def export_selection():
    from export import generer_rapport_multi

    ids = request.form.getlist('ids')

    if not ids:
        return redirect('/historique')

    # Récupérer les données de chaque traitement sélectionné
    traitements = []
    for id_str in ids:
        try:
            id_int = int(id_str)
            row, metriques, parametres, graphiques, descriptif = data.get_one(id_int)
            if row:
                traitements.append((row, metriques, parametres, graphiques, descriptif))
        except Exception:
            continue

    if not traitements:
        return redirect('/historique')

    pdf_bytes = generer_rapport_multi(traitements)

    # Nom du fichier selon le nombre de traitements
    if len(traitements) == 1:
        nom_fichier = f"rapport_{ids[0]}.pdf"
    else:
        nom_fichier = f"rapport_comparatif_{len(traitements)}_analyses.pdf"

    return Response(
        pdf_bytes,
        mimetype='application/pdf',
        headers={
            'Content-Disposition': f'attachment; filename={nom_fichier}'
        }
    )

@app.route('/historique/visualiser-selection', methods=['POST'])
def visualiser_selection():
    ids = request.form.getlist('ids')

    if not ids:
        return redirect('/historique')

    # Récupérer les données de chaque traitement sélectionné
    traitements = []
    for id_str in ids:
        try:
            id_int = int(id_str)
            row, metriques, parametres, graphiques, descriptif = data.get_one(id_int)
            if row:
                traitements.append({
                    'row': row,
                    'metriques': metriques,
                    'parametres': parametres,
                    'graphiques': graphiques,
                    'descriptif': descriptif
                })
        except Exception:
            continue

    if not traitements:
        return redirect('/historique')

    return render_template(
        'visualiser_selection.html',
        traitements=traitements
    )

@app.route('/historique/supprimer-selection', methods=['POST'])
def supprimer_selection():
    ids = request.form.getlist('ids')
    for id_str in ids:
        try:
            data.delete_one(int(id_str))
        except Exception:
            continue
    return redirect('/historique')


@app.route('/historique/<int:id>')
def detail_historique(id):
    row, metriques, parametres, graphiques, descriptif = data.get_one(id)
    if not row:
        return redirect('/historique')
    return render_template('detail_historique.html',
                           row=row,
                           metriques=metriques,
                           parametres=parametres,
                           graphiques=graphiques,
                           descriptif=descriptif)


@app.route('/historique/<int:id>/supprimer', methods=['POST'])
def supprimer_historique(id):
    data.delete_one(id)
    return redirect('/historique')

import shutil


# Supprime un fichier individuel du dataset.
@app.route('/datasets/fichier/<path:filepath>/supprimer', methods=['POST'])
def supprimer_fichier(filepath):
    chemin = os.path.join(UPLOAD_FOLDER, filepath)
    nom_dossier = filepath.split('/')[0]

    if not os.path.isfile(chemin):
        return render_template(
            'datasets_dossier.html',
            nom_dossier=nom_dossier,
            fichiers=get_fichiers_dossier(nom_dossier),
            erreur=f"Fichier introuvable : {filepath}"
        )

    os.remove(chemin)
    return render_template(
        'datasets_dossier.html',
        nom_dossier=nom_dossier,
        fichiers=get_fichiers_dossier(nom_dossier),
        succes=f"Fichier supprimé."
    )


@app.route('/datasets/dossier/<nom_dossier>/supprimer', methods=['POST'])
def supprimer_dossier(nom_dossier):
    chemin = os.path.join(UPLOAD_FOLDER, nom_dossier)

    if not os.path.isdir(chemin):
        return render_template('datasets.html', dossiers=get_dossiers(),
                               erreur=f"Dossier introuvable : {nom_dossier}")

    shutil.rmtree(chemin, ignore_errors=True)
    return render_template('datasets.html', dossiers=get_dossiers(),
                           succes=f"Dossier « {nom_dossier} » supprimé.")




# ROUTE — LANCER BATCH (analyse predictive)
# ---------------------------------------------
# Cette route reçoit le formulaire de batch depuis l'interface.
# Elle reconstruit les expériences, relance éventuellement une expérience
# déjà partiellement exécutée, et lance soit une analyse descriptive,
# soit un entraînement de modèle prédictif pour chaque expérience.

# Route principale qui lance un batch d'analyses prédictives.
@app.route('/modeles/lancer-batch', methods=['POST'])
def lancer_batch():
    try:
        form_data = request.form.to_dict(flat=False)
        batch_id = request.form.get('batch_id')
        relance_exp_num = request.form.get('relance_exp_num')
        relance_idx = int(relance_exp_num) - 1 if relance_exp_num and relance_exp_num.isdigit() else None

        experiences = {}
        pattern = re.compile(r'^exp\[(\d+)\]\[(.+?)\](?:\[\])?$')

        # Lire les champs du formulaire batch et reconstruire les expériences.
        for key, values in form_data.items():
            match = pattern.match(key)
            if not match:
                continue
            idx   = int(match.group(1))
            champ = match.group(2)
            if idx not in experiences:
                experiences[idx] = {}
            if champ == 'fichiers':
                experiences[idx]['fichiers'] = values
            elif champ == 'dossiers':
                experiences[idx]['dossiers'] = values
            else:
                experiences[idx][champ] = values[0] if values else None

        if not experiences:
            return jsonify({'erreur': 'Aucune experience trouvee dans le formulaire.'}), 400

        previous_results = {}
        if batch_id and relance_idx is not None:
            batch_state = _load_batch_state(batch_id)
            if batch_state:
                stored_experiences = batch_state.get('experiences', {})
                if 0 in experiences:
                    stored_experiences[relance_idx] = {
                        **stored_experiences.get(relance_idx, {}),
                        **experiences[0]
                    }
                experiences = stored_experiences
                previous_results = {
                    result['exp_num']: result
                    for result in batch_state.get('resultats', [])
                }

        if not batch_id:
            counter = session.get('batch_counter', 0) + 1
            session['batch_counter'] = counter
            batch_id = f"batch_{counter}"

        tous_les_resultats = []
        erreurs            = []

        for idx in sorted(experiences.keys()):
            exp = experiences[idx]

            modele_id     = exp.get('model') or exp.get('modele')
            noms_dossiers = exp.get('dossiers', [])
            fichiers      = exp.get('fichiers', [])

            # Si on relance une expérience spécifique, on réutilise les résultats précédents.
            if relance_idx is not None and idx != relance_idx and previous_results.get(idx + 1):
                tous_les_resultats.append(previous_results[idx + 1])
                continue

            if not modele_id:
                erreurs.append(f"EXP {idx+1} : aucun modèle sélectionné.")
                continue

            if noms_dossiers:
                aptag_paths = []
                for nd in noms_dossiers:
                    dp = os.path.join(UPLOAD_FOLDER, nd.strip())
                    if not os.path.isdir(dp):
                        erreurs.append(f"EXP {idx+1} : dossier introuvable → {nd}")
                        continue
                    aptags = sorted([
                        f for f in os.listdir(dp)
                        if 'APTags' in f and f.endswith('.csv')
                    ])
                    if not aptags:
                        erreurs.append(f"EXP {idx+1} : aucun APTags dans {nd}")
                        continue
                    aptag_paths.append(os.path.join(dp, aptags[0]))

                if not aptag_paths:
                    erreurs.append(f"EXP {idx+1} : aucun fichier APTags trouvé.")
                    continue

                nom_fichier = ' + '.join(noms_dossiers)

                if len(aptag_paths) == 1:
                    input_path   = aptag_paths[0]
                    dossier_ref  = os.path.dirname(aptag_paths[0])
                    aptags_files = None
                else:
                    dossier_ref  = os.path.dirname(aptag_paths[0])
                    input_path   = aptag_paths[0]
                    aptags_files = aptag_paths

            elif fichiers:
                nom_fichier  = fichiers[0]
                input_path   = os.path.join(UPLOAD_FOLDER, nom_fichier.strip())
                dossier_ref  = os.path.dirname(input_path)
                aptags_files = None
            else:
                erreurs.append(f"EXP {idx+1} : aucun dossier ni fichier sélectionné.")
                continue

            if not os.path.exists(input_path):
                erreurs.append(f"EXP {idx+1} : fichier introuvable → {input_path}")
                continue

            custom_params = {
                k: v for k, v in exp.items()
                if k not in ('model', 'modele', 'fichiers', 'dossiers', 'dossier')
                and v is not None
                and v != ''
            }

            for k, v in custom_params.items():
                try:
                    custom_params[k] = float(v) if '.' in str(v) else int(v)
                except (ValueError, TypeError):
                    pass

            #Chronomètre 
            date_debut = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            t0         = datetime.now()

            # Si le modèle choisi est descriptif, on fait uniquement l'analyse descriptive.
            if modele_id == 'descriptif':
                resultats = lancer_traitement(
                    input_path,
                    dossier=dossier_ref,
                    tolerance_in=custom_params.get('tolerance_in', 0),
                    tolerance_out=custom_params.get('tolerance_out', 0),
                    aptags_files=aptags_files
                )

                date_fin = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                duree    = round((datetime.now() - t0).total_seconds(), 2)

                data.save_traitement({
                    'date_debut'    : date_debut,
                    'date_fin'      : date_fin,
                    'duree'         : duree,
                    'modele'        : 'descriptif',
                    'fichier'       : nom_fichier,
                    'acc_train'     : resultats.get('read_rate', 0),
                    'acc_test'      : resultats.get('accuracy', 0),
                    'nb_epcs'       : resultats.get('nb_epc_attendus', 0),
                    'nb_valides'    : resultats.get('nb_epc_lus', 0),
                    'nb_parasites'  : resultats.get('n_manquants', 0),
                    'descriptif_metrics': resultats,
                    'custom_params' : custom_params,
                })

                tous_les_resultats.append({
                    'exp_num'        : idx + 1,
                    'modele'         : 'descriptif',
                    'fichier'        : nom_fichier,
                    'custom_params'  : custom_params,
                    'read_rate'      : resultats.get('read_rate'),
                    'accuracy'       : resultats.get('accuracy'),
                    'nb_epc_attendus': resultats.get('nb_epc_attendus'),
                    'nb_epc_lus'     : resultats.get('nb_epc_lus'),
                    'nb_lectures'    : resultats.get('nb_lectures'),
                    'nb_runs'        : resultats.get('nb_runs'),
                    'runs_complets'  : resultats.get('runs_complets'),
                    'n_manquants'    : resultats.get('n_manquants'),
                    'marge_moy'      : resultats.get('marge_moy'),
                    'marge_min'      : resultats.get('marge_min'),
                    'marge_max'      : resultats.get('marge_max'),
                    'tags_manquants' : resultats.get('tags_manquants', []),
                    'date_debut'     : date_debut,
                    'date_fin'       : date_fin,
                    'duree'          : duree,
                })
                continue

            # Pour les autres modèles, on prépare les données puis on entraîne le modèle prédictif.
            resultats = preparer_et_entrainer(
                modele_id,
                input_path,
                custom_params,
                dossier_ref=dossier_ref,
                aptags_files=aptags_files
            )

            date_fin = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            duree    = round((datetime.now() - t0).total_seconds(), 2)

            data.save_traitement({
                **resultats,
                "fichier"      : nom_fichier,
                "custom_params": custom_params,
                "date_debut"   : date_debut,
                "date_fin"     : date_fin,
                "duree"        : duree,
            })

            tous_les_resultats.append({
                **resultats,
                'exp_num'      : idx + 1,
                'fichier'      : nom_fichier,
                'custom_params': custom_params,
                'date_debut'   : date_debut,
                'date_fin'     : date_fin,
                'duree'        : duree,
            })

        if batch_id:
            _save_batch_state(batch_id, {
                'experiences': experiences,
                'resultats': tous_les_resultats
            })

        if not tous_les_resultats:
            msg = " | ".join(erreurs) if erreurs else "Aucune expérience n'a pu être lancée."
            return render_template(
                'analyse_predictive.html',
                dossiers=get_dossiers(),
                MODELS_CONFIG=MODELS_CONFIG,
                erreur=msg
            )

        return render_template(
            'ml_p.html',
            resultats=tous_les_resultats,
            MODELS_CONFIG=MODELS_CONFIG,
            erreurs=erreurs,
            batch_id=batch_id
        )

    except Exception as e:
        return render_template(
            'analyse_predictive.html',
            dossiers=get_dossiers(),
            MODELS_CONFIG=MODELS_CONFIG,
            erreur=str(e)
        )
    
@app.route('/historique/<int:id>/export-csv')
def export_csv(id):
    row, metriques, parametres, graphiques, descriptif = data.get_one(id)
    if not row:
        return redirect('/historique')

    # Les données sont déjà dans row, metriques, parametres

    import csv
    import io

    # Export CSV : on écrit d'abord les informations générales puis les métriques.
    output = io.StringIO()
    writer = csv.writer(output)

    # En-tête
    writer.writerow(['Champ', 'Valeur'])

    # Infos générales — indices corrects selon ta table
    writer.writerow(['ID',                row[0]])
    writer.writerow(['Date début',        row[1] or '—'])
    writer.writerow(['Date fin',          row[2] or '—'])
    writer.writerow(['Durée (s)',         row[3] or '—'])
    writer.writerow(['Modèle (id)',       row[4]])
    writer.writerow(['Dataset',           row[5]])
    writer.writerow(['Nom modèle',        row[11]])  # vient de la jointure
    writer.writerow(['Sigle modèle',      row[12]])  # vient de la jointure

    # Métriques ML — directement depuis row
    writer.writerow([])
    writer.writerow(['--- Métriques ML ---', ''])
    writer.writerow(['Accuracy Train (%)',  row[6]])
    writer.writerow(['Accuracy Test (%)',   row[7]])
    writer.writerow(['EPCs analysés',       row[8]])
    writer.writerow(['IN (valides)',         row[9]])
    writer.writerow(['OUT (parasites)',      row[10]])

    # Métriques détaillées — depuis la table metriques
    # metriques = (id, id_traitement, f1, precision, recall, cv_score, oob_score, n_arbres)
    if metriques:
        writer.writerow([])
        writer.writerow(['--- Métriques détaillées ---', ''])
        if metriques[2]:
            writer.writerow(['F1 Score (%)',            round(metriques[2] * 100, 1)])
        if metriques[3]:
            writer.writerow(['Précision (%)',           round(metriques[3] * 100, 1)])
        if metriques[4]:
            writer.writerow(['Rappel (%)',              round(metriques[4] * 100, 1)])
        if metriques[5]:
            writer.writerow(['Validation croisée (%)',  round(metriques[5] * 100, 1)])
        if metriques[6]:
            writer.writerow(['OOB Score (%)',           round(metriques[6] * 100, 1)])
        if metriques[7]:
            writer.writerow(["Nombre d'arbres",         metriques[7]])

    # Paramètres — depuis la table parametres
    if parametres:
        writer.writerow([])
        writer.writerow(['--- Paramètres utilisés ---', ''])
        for k, v in parametres.items():
            writer.writerow([k, v])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={
            'Content-Disposition': f'attachment; filename=analyse_{id}_{row[4]}.csv'
        }
    )

@app.route('/historique/<int:id>/export-graphiques')
def export_graphiques(id):
    row, metriques, parametres, graphiques, descriptif = data.get_one(id)
    if not row:
        return redirect('/historique')

    if not graphiques:
        return redirect(f'/historique/{id}')

    import zipfile
    import base64
    import io

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        noms = {
            'confusion'  : 'matrice_confusion.png',
            'roc'        : 'courbe_roc.png',
            'scores'     : 'distribution_scores.png',
            'erreur_vs_k': 'erreur_vs_k.png',
            'importance' : 'importance_variables.png',
            'metriques'  : 'metriques_comparees.png',
            'rssi'       : 'distribution_rssi.png',
        }
        for cle, nom_fichier in noms.items():
            if graphiques.get(cle):
                try:
                    img_bytes = base64.b64decode(graphiques[cle])
                    zf.writestr(nom_fichier, img_bytes)
                except Exception:
                    pass

    zip_buffer.seek(0)
    return Response(
        zip_buffer.getvalue(),
        mimetype='application/zip',
        headers={
            'Content-Disposition': f'attachment; filename=graphiques_{id}_{row[2]}.zip'
        }
    )



@app.route('/historique/<int:id>/export-pdf')
def export_pdf(id):
    from export import generer_rapport_pdf
    row, metriques, parametres, graphiques, descriptif = data.get_one(id)
    if not row:
        return redirect('/historique')
    pdf_bytes = generer_rapport_pdf(row, metriques, parametres, graphiques, descriptif)
    return Response(
        pdf_bytes,
        mimetype='application/pdf',
        headers={
            'Content-Disposition': f'attachment; filename=rapport_{id}_{row[4]}.pdf'
        }
    )


if __name__ == '__main__':
    app.run(debug=True)