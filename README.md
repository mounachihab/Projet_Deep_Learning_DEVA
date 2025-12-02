**README — Multimodal Emotion Valence Prediction on MOSI**

## Objectif du projet
L’objectif de ce projet est de concevoir un modèle multimodal capable de prédire la valence émotionnelle entre -3 et 3, exprimée par un locuteur dans des vidéos.\
Pour cela, nous fusionnons trois modalités complémentaires :

- Texte 
- Audio 
- Vidéo 

Nous utilisons un modèle inspiré de DEVA (Decoupled Emotional Video-Audio), nommé DEVANet, intégrant des mécanismes avancés de fusion multimodale.

## Problématique
Comment fusionner efficacement les informations textuelles, audio et visuelles afin de prédire avec précision la valence émotionnelle exprimée par un locuteur dans les vidéos du dataset MOSI, grâce à l’architecture DEVA?

## Dataset 
Choix des données à partir du dossier MOSI RAW :

Nous sommes partis des données issues du lien suivant : https://www.kaggle.com/datasets/mathurinache/cmu-mosi

Vidéo **— raw/Video/Full/**

- Fichiers conservés :
  - **  .mp4
- Pourquoi on utilise ces données :
  - `  `Vidéos complètes, une par ID
  - `  `Alignées parfaitement avec les transcriptions complètes
  - `  `Nécessaires pour l’extraction des expressions faciales (OpenFace)
- Pourquoi on n’utilise pas les autres dossiers :
  - `  `Video/Segmented/ contient 2199 segments mais pas de transcription segmentée correspondante → impossible d’aligner texte/audio/vidéo

Audio **— raw/Audio/WAV\_16000/Full/**

- Fichiers conservés :
  -   .wav (16 kHz)
- Pourquoi on utilise ces données :
  - `  `Audio brut complet, aligné 1:1 avec les vidéos complètes
  - `  `Qualité élevée, indispensable pour MFCC + jitter + shimmer + pitch (AED)
- Pourquoi on n’utilise pas les autres dossiers :
  - `  `Audio/Segmented/ : 2199 segments sans transcripts alignés

Transcript **— raw/Transcript/Full/**

- Fichiers conservés :
  -   .textonly
- Pourquoi on utilise ces données :
  - `  `Transcriptions complètes, propres et alignées avec les vidéos/audio complets
  - `  `Parfaites pour l’encodage BERT (TextEncoder)
- Pourquoi on n’utilise pas les autres dossiers :
  - `  `Transcript/Segmented/ contient des .annotprocessed mais :
    - ils ne couvrent pas tous les segments
    - donc ils ne correspondent pas aux segments audio/vidéo
    - donc impossible de reconstituer proprement un échantillon multimodal

- `  `Seules 93 vidéos ont simultanément :
  - un fichier vidéo complet (.mp4)
  - un fichier audio complet (.wav)
  - une transcription complète (.textonly)

- `  `Ce sont les seules données parfaitement alignées dans les trois modalités.
- `  `En multimodal, chaque échantillon doit contenir texte + audio + vidéo, sans quoi il est inutilisable.

### Labels
- `  `Les labels MOSI (valence continue -3 → +3) n’existant pas dans le dataset initial, ont été récupérés :
  - d’une part à partir de chatgpt, 
  - d’autre part grâce au lien suivant : [CMU-MOSI](https://www.kaggle.com/datasets/reganw/cmu-mosi)

- ` `(Une comparaison des métriques a été effectuée pour la sélection des meilleurs labels)


## Structure du projet :

- extract\_donnees.ipynb – Extraction initiale des IDs utilisables (transcript/audio/vidéo)
- mosi\_prepared\_dataset.py – Dataset multimodal final : texte, MFCC, vidéo réduite, AED, VED

- filter\_valid\_ids.py – Filtrage automatique des IDs valides (présence transcript/audio/vidéo/AED/VED)

- mosi\_audio\_and\_AED.ipynb – Extraction features audio + génération AED → embedding D\_a
- mosi\_openface\_and\_VED.ipynb – Extraction OpenFace + AU → description visuelle → embedding D\_v
- mosi\_reduce\_video\_features.ipynb – Réduction des features vidéo (PCA ou sélection)

- devanet\_model.py – Contient les briques essentielles utilisées par DEVA pour effectuer la fusion (CrossModalAttention, MFU, projecteurs, etc.).
- devanet\_modules.py – Modules : MFU, projecteurs audio/vidéo, alignement multimodal
- deva.py – Implémente l’architecture multimodale complète :TextEncoder, AED/VED, TPF, CrossTransformer, tête finale.
- bert.py – Encapsulation BERT/RoBERTa : extraction d’embeddings 768 dims
  Lien BERT : (https://huggingface.co/google-bert/bert-base-uncased)


- tpf\_layer.py – Implémentation complète du TPF (Token Pivot Fusion) : Transformers, cross-attention, pré-normalisation
- collate\_utils.py – Collate personnalisé pour BERT + batch multimodal

- train\_devanet.ipynb – Notebook d’entraînement complet (sécurisé, NaN-safe)
- metrics.py – Calcul des métriques MOSI (MAE, MSE, Pearson, CCC)

- MOSI\_full/ – Données multimodales alignées (texte, audio, vidéo, AED, VED)


## Pipeline complet
### Extraction audio
**1) Extraction des features audio .pt**

- À partir des fichiers .wav, on calcule des MFCC (13 coefficients sur 50 frames).
- Ces MFCC sont directement stockés dans des fichiers .pt.
- Ces fichiers .pt servent de features audio numériques dans DEVANet.

**2) Extraction AED (Audio Emotional Description)**

- On calcule 4 indicateurs émotionnels : pitch, loudness, jitter, shimmer.
- On génère une phrase décrivant la manière de parler (ex : *“The speaker uses a soft voice with low pitch…”*).
- Cette phrase est encodée par BERT → produit D\_a (embedding 128d).

### Extraction vidéo
**1) Extraction des CSV OpenFace**

Pour chaque vidéo .mp4 :

- OpenFace produit un fichier CSV contenant :
  - landmarks 2D 
  - landmarks 3D
  - pose de la tête
  - regard (gaze)
  - parfois Action Units (AU)

Ces CSV sont la matière brute pour l’analyse vidéo.

**2) Réduction des CSV vidéo → CSV final réduit**

- On récupère pour chaque vidéo un gros CSV OpenFace (landmarks, pose, gaze, AUs…).
- On essaie d’abord de garder uniquement les colonnes AU si elles existent (meilleure info émotionnelle).
- Sinon, on garde les colonnes de pose et de regard (pose\_Tx, gaze\_0\_x…).
- Sinon, on garde les 20 premières colonnes numériques du CSV.
- On limite le CSV final à 50 frames maximum.
- On sauvegarde ce CSV réduit, qui sera ensuite transformé en video\_feat.

à Ces CSV réduits sont ensuite convertis en tenseurs video\_feat dans mosi\_prepared\_dataset.

**3) Extraction VED (Visual Emotional Description)**

- À partir des landmarks du CSV, on simules certaines Action Units :\
  AU01, AU02, AU07, AU12, AU25, AU26.
- on calcule une phrase décrivant l’expression (ex : *“raised inner brows and parted lips”*).
- Cette phrase est encodée via BERT → produit D\_v (embedding 128d).

## Préparation du dataset
Chaque exemple MOSI contient :

- text : tokens BERT
- audio\_feat : MFCC
- video\_feat : features vidéo réduites
- D\_a, D\_v : embeddings émotionnels
- label : valence réelle

**DEVANet**

Le modèle suit les étapes suivantes :

1. TextEncoder → extrait X\_t
1. ajout des tokens D\_a et D\_v → X\_t\_aug
1. projection audio & vidéo → X\_a, X\_v
1. fusion multimodale via MFU
1. pooling
1. régression de la valence

## Entraînement
L’entraînement est effectué via train\_devanet.ipynb :

- DataLoader multimodal (collate personnalisé)
- gestion des NaN
- gradient clipping
- optimiser : Adam, lr = 5e-5
- loss : MSE (régression)

### Évaluation
Les métriques utilisées pour MOSI sont :

- **MAE** – Mean Absolute Error
- **MSE** – Mean Squared Error
- **Pearson r** – corrélation texte↔prédiction
- **CCC** – Concordance Correlation Coefficient (référence dans la littérature)

### Comparaison labels (ChatGPT) vs labels (kaggle)

  Dans notre projet, nous avons évalué deux jeux de labels (car non existant dans le dataset initial) :
- Labels ChatGPT — générés automatiquement à partir des transcriptions
- Labels Prof/Kaggle — fournis dans les datasets officiels MOSI/MOSEI
  
Comparaison directe des labels :

L’analyse statistique montre un écart massif entre les deux sources :
    •	Taux d’hallucination : 96.77 %
Dans 96.77 % des segments, les labels ChatGPT ne correspondent pas du tout aux vraies annotations humaines.
    •	Écart moyen : 1.6 points sur une échelle [-3, +3]
    
Cela représente 27 % d’erreur relative, ce qui est énorme pour un problème de régression fine.

 Impact sur le modèle multimodal :
 
Lorsque l’on entraîne notre DEVANet sur les labels ChatGPT, les résultats semblent bons :

Données ChatGPT — Test
    •	Acc-2 : 0.7895
    •	F1 : 0.7823
    •	MAE : 0.9205
    •	Corr : 0.7548
    
Mais avec les labels officiels, les performances chutent :

Données Prof/Kaggle — Test

    •	Acc-2 : 0.6842
    •	F1 : 0.6842
    •	MAE : 1.2762
    •	Corr : 0.2941
    
Interprétation :

Le modèle paraît meilleur avec les labels ChatGPT mais uniquement parce que
ChatGPT génère des labels plus faciles, plus réguliers et moins nuancés que les annotations humaines.
Le modèle apprend des tendances artificielles, pas des émotions réelles.
Conclusion : quel jeu de labels choisir ?

-	Les labels ChatGPT ne peuvent pas servir de référence, car ils ne capturent pas la complexité émotionnelle réelle et introduisent un fort biais.
-	Les labels Kaggle/Prof sont indispensables : ce sont les annotations humaines utilisées dans tous les benchmarks MOSI/MOSEI.




