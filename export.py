"""
export.py
---------
Génération du rapport PDF pour une analyse RFID.
Utilise ReportLab pour produire un document structuré et présentable.
"""

import io
import base64
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, Image, PageBreak,
    HRFlowable
)


#Palette de couleurs du projet
BLEU    = colors.HexColor('#00d4ff')
VIOLET  = colors.HexColor('#5d40ff')
ORANGE  = colors.HexColor('#ffab40')
VERT    = colors.HexColor('#00e676')
ROUGE   = colors.HexColor('#ef4444')
GRIS    = colors.HexColor('#7a8090')
NOIR    = colors.HexColor('#141619')
BLANC   = colors.white
GRIS_CL = colors.HexColor('#f0f0f0')


#Styles typographiques
def _styles():
    base = getSampleStyleSheet()

    titre_page = ParagraphStyle(
        'TitrePage',
        fontSize=26,
        fontName='Helvetica-Bold',
        textColor=NOIR,
        alignment=TA_CENTER,
        spaceAfter=20,
    )
    sous_titre = ParagraphStyle(
        'SousTitre',
        fontSize=13,
        fontName='Helvetica',
        textColor=GRIS,
        alignment=TA_CENTER,
        spaceAfter=16,
    )
    titre_section = ParagraphStyle(
        'TitreSection',
        fontSize=14,
        fontName='Helvetica-Bold',
        textColor=VIOLET,
        spaceBefore=18,
        spaceAfter=8,
        borderPadding=(0, 0, 4, 0),
    )
    corps = ParagraphStyle(
        'Corps',
        fontSize=10,
        fontName='Helvetica',
        textColor=colors.HexColor('#333333'),
        spaceAfter=6,
        leading=16,
    )
    label_metrique = ParagraphStyle(
        'LabelMetrique',
        fontSize=9,
        fontName='Helvetica',
        textColor=GRIS,
        alignment=TA_CENTER,
    )
    valeur_metrique = ParagraphStyle(
        'ValeurMetrique',
        fontSize=18,
        fontName='Helvetica-Bold',
        textColor=BLEU,
        alignment=TA_CENTER,
    )

    return {
        'titre_page'     : titre_page,
        'sous_titre'     : sous_titre,
        'titre_section'  : titre_section,
        'corps'          : corps,
        'label_metrique' : label_metrique,
        'valeur_metrique': valeur_metrique,
    }


#Convertir une image base64 en objet Image ReportLab
def _base64_vers_image(b64_str, largeur=14*cm, hauteur=9*cm):
    # Transforme une image encodée en base64 en objet Image pour ReportLab.
    """
    Convertit une chaîne base64 en Image ReportLab.
    Retourne None si la conversion échoue.
    """
    try:
        img_bytes = base64.b64decode(b64_str)
        buf = io.BytesIO(img_bytes)
        return Image(buf, width=largeur, height=hauteur)
    except Exception:
        return None


#Tableau de métriques stylisé
def _tableau_metriques(donnees, styles):
    """
    donnees : liste de tuples (label, valeur)
    Retourne un objet Table ReportLab mis en forme.
    """
    # Crée un tableau stylisé qui met en valeur les métriques ou paramètres.
    data_table = [['Indicateur', 'Valeur']]
    for label, valeur in donnees:
        data_table.append([label, str(valeur)])

    t = Table(data_table, colWidths=[10*cm, 6*cm])
    t.setStyle(TableStyle([
        # En-tête
        ('BACKGROUND',   (0, 0), (-1, 0),  VIOLET),
        ('TEXTCOLOR',    (0, 0), (-1, 0),  BLANC),
        ('FONTNAME',     (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',     (0, 0), (-1, 0),  10),
        ('ALIGN',        (0, 0), (-1, 0),  'CENTER'),
        ('BOTTOMPADDING',(0, 0), (-1, 0),  8),
        ('TOPPADDING',   (0, 0), (-1, 0),  8),
        # Corps
        ('FONTNAME',     (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',     (0, 1), (-1, -1), 10),
        ('ALIGN',        (1, 1), (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [GRIS_CL, BLANC]),
        ('BOTTOMPADDING',(0, 1), (-1, -1), 6),
        ('TOPPADDING',   (0, 1), (-1, -1), 6),
        # Bordures
        ('GRID',         (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('ROUNDEDCORNERS', [4]),
    ]))
    return t


# Fonction principale
# Cette fonction construit le PDF final en combinant les sections de résumé,
# les graphiques et les détails techniques pour une analyse individuelle.
def generer_rapport_pdf(row, metriques, parametres, graphiques, descriptif=None):
    """
    Génère un rapport PDF complet pour une analyse RFID.

    Paramètres (exactement ce que retourne data.get_one()) :
        row        : tuple avec les infos du traitement
        metriques  : tuple avec les métriques ML détaillées
        parametres : dict {nom: valeur} des paramètres utilisés
        graphiques : dict {nom: base64} des graphiques

    Retourne : bytes du PDF généré
    """
    # Construit un PDF structuré avec titre, résumé, graphiques et détails techniques.

    buffer = io.BytesIO()
    styles = _styles()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
        title=f"Rapport Analyse RFID #{row[0]}",
        author="RFID Analytics — ESIGELEC",
    )

    contenu = []  # liste des éléments à mettre dans le PDF

    
    # PAGE DE TITRE
    

    contenu.append(Spacer(1, 3*cm))

    contenu.append(Paragraph("Rapport d'Analyse RFID", styles['titre_page']))
    contenu.append(Spacer(1, 0.8*cm))
    contenu.append(Paragraph("Machine Learning — Classification des lectures RFID", styles['sous_titre']))
    contenu.append(Spacer(1, 0.8*cm))
    contenu.append(HRFlowable(width="100%", thickness=2, color=BLEU))
    contenu.append(Spacer(1, 0.5*cm))

    # Infos de base en tableau centré
    nom_modele = row[11] if row[11] else row[4]
    date_export = datetime.now().strftime('%d/%m/%Y à %H:%M')

    infos_titre = [
        ['Projet',        'RFID Analytics — ESIGELEC'],
        ['Analyse n°',    str(row[0])],
        ['Modèle',        nom_modele],
        ['Dataset',       str(row[5])],
        ['Généré le',     date_export],
    ]

    t_titre = Table(infos_titre, colWidths=[5*cm, 11*cm])
    t_titre.setStyle(TableStyle([
        ('FONTNAME',  (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME',  (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE',  (0, 0), (-1, -1), 11),
        ('TEXTCOLOR', (0, 0), (0, -1), VIOLET),
        ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#333333')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING',    (0, 0), (-1, -1), 7),
        ('ALIGN',     (0, 0), (-1, -1), 'LEFT'),
    ]))
    contenu.append(t_titre)
    contenu.append(PageBreak())

    # SECTION 1 — RÉSUMÉ
    

    contenu.append(Paragraph("1. Résumé de l'analyse", styles['titre_section']))
    contenu.append(HRFlowable(width="100%", thickness=1, color=BLEU))
    contenu.append(Spacer(1, 0.4*cm))

    # Texte de résumé
    acc_test  = row[7] or 0
    acc_train = row[6] or 0
    duree     = row[3] or '—'
    is_descriptif = row[4] == 'descriptif'

    if is_descriptif:
        resume_texte = (
            f"Cette analyse descriptive a été réalisée sur le dataset <b>{row[5]}</b>. "
            f"Le traitement a duré <b>{duree} secondes</b>. "
            f"Le taux de lecture est de <b>{descriptif['read_rate'] if descriptif and descriptif.get('read_rate') is not None else acc_train}%</b> "
            f"et l'accuracy de <b>{descriptif['accuracy'] if descriptif and descriptif.get('accuracy') is not None else acc_test}%</b>."
        )
    else:
        resume_texte = (
            f"Cette analyse a été réalisée avec le modèle <b>{nom_modele}</b> "
            f"sur le dataset <b>{row[5]}</b>. "
            f"Le traitement a duré <b>{duree} secondes</b>. "
            f"Le modèle obtient une accuracy de <b>{acc_test}%</b> sur les données de test "
            f"et de <b>{acc_train}%</b> sur les données d'entraînement."
        )

    contenu.append(Paragraph(resume_texte, styles['corps']))
    contenu.append(Spacer(1, 0.4*cm))

    # Métriques clés en résumé
    if is_descriptif:
        metriques_resume = [
            ('Taux de lecture', f"{descriptif['read_rate'] if descriptif and descriptif.get('read_rate') is not None else acc_train}%"),
            ('Accuracy',        f"{descriptif['accuracy'] if descriptif and descriptif.get('accuracy') is not None else acc_test}%"),
            ('EPCs attendus',   str(descriptif['nb_epc_attendus'] if descriptif and descriptif.get('nb_epc_attendus') is not None else (row[8] or '—'))),
            ('EPCs lus',        str(descriptif['nb_epc_lus'] if descriptif and descriptif.get('nb_epc_lus') is not None else (row[9] or '—'))),
            ('Tags manquants',  str(descriptif['n_manquants'] if descriptif and descriptif.get('n_manquants') is not None else '—')),
        ]
    else:
        metriques_resume = [
            ('Accuracy Train', f"{acc_train}%"),
            ('Accuracy Test',  f"{acc_test}%"),
            ('EPCs analysés',  str(row[8] or '—')),
            ('IN (valides)',   str(row[9] or '—')),
            ('OUT (parasites)',str(row[10] or '—')),
        ]
    if metriques:
        if metriques[2]:
            metriques_resume.append(('F1 Score', f"{round(metriques[2]*100, 1)}%"))
        if metriques[4]:
            metriques_resume.append(('Rappel',   f"{round(metriques[4]*100, 1)}%"))
        if metriques[5]:
            metriques_resume.append(('Validation croisée', f"{round(metriques[5]*100, 1)}%"))
        if metriques[6]:
            metriques_resume.append(('OOB Score', f"{round(metriques[6]*100, 1)}%"))
        if metriques[7]:
            metriques_resume.append(("Nombre d'arbres", str(metriques[7])))

    contenu.append(_tableau_metriques(metriques_resume, styles))
    contenu.append(PageBreak())

    
    # SECTION 2 — GRAPHIQUES
    

    if graphiques:
        contenu.append(Paragraph("2. Visualisations", styles['titre_section']))
        contenu.append(HRFlowable(width="100%", thickness=1, color=BLEU))
        contenu.append(Spacer(1, 0.4*cm))

        # Mapping nom technique → titre lisible + description
        infos_graphiques = {
            'confusion' : (
                'Matrice de confusion',
                'Montre combien de tags IN et OUT ont été correctement classifiés.'
            ),
            'roc' : (
                'Courbe ROC (AUC)',
                'Plus la courbe est haute et à gauche, meilleur est le modèle. AUC = 1 est parfait.'
            ),
            'scores' : (
                'Distribution des scores de classification',
                'Montre la séparation entre les tags IN (valides) et OUT (parasites).'
            ),
            'importance' : (
                'Importance des variables',
                'Variables les plus utiles pour la classification (Random Forest uniquement).'
            ),
            'metriques' : (
                'Comparaison des métriques',
                'Vue synthétique de toutes les métriques du modèle.'
            ),
        }

        for cle, (titre_graph, description) in infos_graphiques.items():
            if not graphiques.get(cle):
                continue

            contenu.append(Paragraph(titre_graph, styles['titre_section']))
            contenu.append(Paragraph(description, styles['corps']))
            contenu.append(Spacer(1, 0.3*cm))

            img = _base64_vers_image(graphiques[cle], largeur=15*cm, hauteur=10*cm)
            if img:
                # Centrer l'image dans un tableau à 1 cellule
                t_img = Table([[img]], colWidths=[15*cm])
                t_img.setStyle(TableStyle([
                    ('ALIGN',   (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN',  (0, 0), (-1, -1), 'MIDDLE'),
                    ('BOX',     (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
                    ('ROUNDEDCORNERS', [4]),
                    ('PADDING', (0, 0), (-1, -1), 8),
                ]))
                contenu.append(t_img)

            contenu.append(Spacer(1, 0.5*cm))
            contenu.append(PageBreak())

    
    # SECTION 3 — DÉTAILS TECHNIQUES
    

    contenu.append(Paragraph("3. Détails techniques", styles['titre_section']))
    contenu.append(HRFlowable(width="100%", thickness=1, color=BLEU))
    contenu.append(Spacer(1, 0.4*cm))

    # Informations du traitement
    details_tech = [
        ('ID du traitement',  str(row[0])),
        ('Date de début',     str(row[1] or '—')),
        ('Date de fin',       str(row[2] or '—')),
        ('Durée',             f"{row[3]} secondes" if row[3] else '—'),
        ('Modèle utilisé',    nom_modele),
        ('Sigle modèle',      str(row[12] or '—')),
        ('Dataset',           str(row[5])),
    ]
    contenu.append(_tableau_metriques(details_tech, styles))
    contenu.append(Spacer(1, 0.5*cm))

    # Paramètres utilisés
    if parametres:
        contenu.append(Paragraph("Paramètres du modèle", styles['titre_section']))
        contenu.append(Spacer(1, 0.2*cm))

        params_data = [['Paramètre', 'Valeur']]
        for k, v in parametres.items():
            params_data.append([str(k), str(v)])

        t_params = Table(params_data, colWidths=[10*cm, 6*cm])
        t_params.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, 0),  ORANGE),
            ('TEXTCOLOR',     (0, 0), (-1, 0),  BLANC),
            ('FONTNAME',      (0, 0), (-1, 0),  'Helvetica-Bold'),
            ('FONTSIZE',      (0, 0), (-1, -1), 10),
            ('ALIGN',         (1, 0), (-1, -1), 'CENTER'),
            ('ROWBACKGROUNDS',(0, 1), (-1, -1), [GRIS_CL, BLANC]),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING',    (0, 0), (-1, -1), 6),
            ('GRID',          (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ]))
        contenu.append(t_params)

    
    # PIED DE PAGE FINAL
    

    contenu.append(Spacer(1, 1*cm))
    contenu.append(HRFlowable(width="100%", thickness=1, color=GRIS))
    contenu.append(Spacer(1, 0.3*cm))
    contenu.append(Paragraph(
        f"Document généré automatiquement par RFID Analytics — ESIGELEC — {date_export}",
        ParagraphStyle('pied', fontSize=8, textColor=GRIS, alignment=TA_CENTER)
    ))

    #Construction du PDF 
    doc.build(contenu)
    buffer.seek(0)
    return buffer.getvalue()


def generer_rapport_multi(traitements):
    """
    Génère un rapport PDF pour plusieurs traitements.

    Paramètre :
        traitements : liste de tuples (row, metriques, parametres, graphiques, descriptif)
                      exactement ce que retourne data.get_one() pour chaque ID

    Retourne : bytes du PDF généré
    """
    # Produit un rapport comparatif entre plusieurs analyses enregistrées.

    buffer = io.BytesIO()
    styles = _styles()
    date_export = datetime.now().strftime('%d/%m/%Y à %H:%M')

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
        title=f"Rapport Comparatif — {len(traitements)} analyses RFID",
        author="RFID Analytics — ESIGELEC",
    )

    contenu = []

    
    # PAGE DE TITRE GLOBALE
    

    contenu.append(Spacer(1, 3*cm))
    contenu.append(Paragraph("Rapport Comparatif", styles['titre_page']))
    contenu.append(Spacer(1, 0.8*cm))
    contenu.append(Paragraph(
        f"Comparaison de {len(traitements)} analyse(s) RFID",
        styles['sous_titre']
    ))
    contenu.append(Spacer(1, 0.8*cm))
    contenu.append(HRFlowable(width="100%", thickness=2, color=BLEU))
    contenu.append(Spacer(1, 0.5*cm))

    infos_titre = [
        ['Projet',       'RFID Analytics — ESIGELEC'],
        ['Généré le',    date_export],
        ['Nb analyses',  str(len(traitements))],
    ]
    t_titre = Table(infos_titre, colWidths=[5*cm, 11*cm])
    t_titre.setStyle(TableStyle([
        ('FONTNAME',      (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME',      (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE',      (0, 0), (-1, -1), 11),
        ('TEXTCOLOR',     (0, 0), (0, -1), VIOLET),
        ('TEXTCOLOR',     (1, 0), (1, -1), colors.HexColor('#333333')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING',    (0, 0), (-1, -1), 7),
        ('ALIGN',         (0, 0), (-1, -1), 'LEFT'),
    ]))
    contenu.append(t_titre)
    contenu.append(PageBreak())


    # TABLEAU COMPARATIF GLOBAL
    

    contenu.append(Paragraph("Vue d'ensemble — Comparaison des analyses", styles['titre_section']))
    contenu.append(HRFlowable(width="100%", thickness=1, color=BLEU))
    contenu.append(Spacer(1, 0.4*cm))

    # En-tête du tableau comparatif
    data_comparatif = [[
        'ID', 'Modèle', 'Dataset',
        'Train', 'Test', 'F1', 'Durée'
    ]]

    for row, metriques, parametres, graphiques, descriptif in traitements:
        f1 = f"{round(metriques[2]*100, 1)}%" if metriques and metriques[2] else '—'
        data_comparatif.append([
            f"#{row[0]}",
            str(row[11] or row[4]),
            str(row[5])[:22],           # tronquer si trop long
            f"{row[6]}%",
            f"{row[7]}%",
            f1,
            f"{row[3]}s" if row[3] else '—',
        ])

    t_comp = Table(data_comparatif, colWidths=[1.5*cm, 3*cm, 4.5*cm, 1.8*cm, 1.8*cm, 1.8*cm, 2*cm])
    t_comp.setStyle(TableStyle([
        # En-tête
        ('BACKGROUND',    (0, 0), (-1, 0),  VIOLET),
        ('TEXTCOLOR',     (0, 0), (-1, 0),  BLANC),
        ('FONTNAME',      (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',       (0, 0), (-1, 0),  9),
        ('ALIGN',         (0, 0), (-1, 0),  'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, 0),  8),
        ('TOPPADDING',    (0, 0), (-1, 0),  8),
        # Corps
        ('FONTNAME',      (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',      (0, 1), (-1, -1), 9),
        ('ALIGN',         (0, 1), (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [GRIS_CL, BLANC]),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('TOPPADDING',    (0, 1), (-1, -1), 6),
        ('GRID',          (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
    ]))
    contenu.append(t_comp)
    contenu.append(PageBreak())

    
    # DÉTAIL DE CHAQUE TRAITEMENT
    

    for numero, (row, metriques, parametres, graphiques, descriptif) in enumerate(traitements, start=1):
        nom_modele = row[11] if row[11] else row[4]

        # Titre du traitement
        contenu.append(Paragraph(
            f"Analyse {numero} sur {len(traitements)} — #{row[0]} — {nom_modele}",
            styles['titre_section']
        ))
        contenu.append(HRFlowable(width="100%", thickness=1, color=ORANGE))
        contenu.append(Spacer(1, 0.4*cm))

        #Résumé en texte
        acc_test  = row[7] or 0
        acc_train = row[6] or 0
        resume = (
            f"Modèle <b>{nom_modele}</b> — Dataset : <b>{row[5]}</b> — "
            f"Accuracy test : <b>{acc_test}%</b> — "
            f"Durée : <b>{row[3] or '—'} secondes</b>"
        )
        contenu.append(Paragraph(resume, styles['corps']))
        contenu.append(Spacer(1, 0.3*cm))

        #Métriques
        metriques_liste = [
            ('Accuracy Train', f"{acc_train}%"),
            ('Accuracy Test',  f"{acc_test}%"),
            ('EPCs analysés',  str(row[8] or '—')),
            ('IN (valides)',   str(row[9] or '—')),
            ('OUT (parasites)',str(row[10] or '—')),
        ]
        if metriques:
            if metriques[2]:
                metriques_liste.append(('F1 Score',    f"{round(metriques[2]*100,1)}%"))
            if metriques[4]:
                metriques_liste.append(('Rappel',      f"{round(metriques[4]*100,1)}%"))
            if metriques[5]:
                metriques_liste.append(('Validation croisée', f"{round(metriques[5]*100,1)}%"))
            if metriques[6]:
                metriques_liste.append(('OOB Score',   f"{round(metriques[6]*100,1)}%"))
            if metriques[7]:
                metriques_liste.append(("Nb d'arbres", str(metriques[7])))

        contenu.append(_tableau_metriques(metriques_liste, styles))
        contenu.append(Spacer(1, 0.5*cm))

        #Paramètres
        if parametres:
            contenu.append(Paragraph("Paramètres utilisés", styles['titre_section']))
            contenu.append(Spacer(1, 0.2*cm))
            params_data = [['Paramètre', 'Valeur']]
            for k, v in parametres.items():
                params_data.append([str(k), str(v)])
            t_params = Table(params_data, colWidths=[10*cm, 6*cm])
            t_params.setStyle(TableStyle([
                ('BACKGROUND',    (0, 0), (-1, 0),  ORANGE),
                ('TEXTCOLOR',     (0, 0), (-1, 0),  BLANC),
                ('FONTNAME',      (0, 0), (-1, 0),  'Helvetica-Bold'),
                ('FONTSIZE',      (0, 0), (-1, -1), 9),
                ('ALIGN',         (1, 0), (-1, -1), 'CENTER'),
                ('ROWBACKGROUNDS',(0, 1), (-1, -1), [GRIS_CL, BLANC]),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING',    (0, 0), (-1, -1), 5),
                ('GRID',          (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
            ]))
            contenu.append(t_params)
            contenu.append(Spacer(1, 0.5*cm))

        #Graphiques
        if graphiques:
            infos_graphiques = {
                'confusion' : 'Matrice de confusion',
                'roc'       : 'Courbe ROC',
                'scores'    : 'Distribution des scores',
                'importance': 'Importance des variables',
                'metriques' : 'Métriques comparées',
            }
            for cle, titre_graph in infos_graphiques.items():
                if not graphiques.get(cle):
                    continue
                contenu.append(Paragraph(titre_graph, styles['titre_section']))
                img = _base64_vers_image(graphiques[cle], largeur=14*cm, hauteur=9*cm)
                if img:
                    t_img = Table([[img]], colWidths=[14*cm])
                    t_img.setStyle(TableStyle([
                        ('ALIGN',  (0, 0), (-1, -1), 'CENTER'),
                        ('BOX',    (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
                        ('PADDING',(0, 0), (-1, -1), 6),
                    ]))
                    contenu.append(t_img)
                contenu.append(Spacer(1, 0.4*cm))

        # Séparation entre traitements
        contenu.append(PageBreak())

    #Pied de page final
    contenu.append(HRFlowable(width="100%", thickness=1, color=GRIS))
    contenu.append(Spacer(1, 0.3*cm))
    contenu.append(Paragraph(
        f"Document généré automatiquement par RFID Analytics — ESIGELEC — {date_export}",
        ParagraphStyle('pied', fontSize=8, textColor=GRIS, alignment=TA_CENTER)
    ))

    doc.build(contenu)
    buffer.seek(0)
    return buffer.getvalue()