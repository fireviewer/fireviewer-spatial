# Contrats Omniverse FireViewer V1

Ce dossier définit quatre contrats distincts et liés par empreinte.

1. `fireviewer.omniverse-map-upload-contract.v1` décrit une carte SimReady
   destinée à l'upload. La scène contient uniquement le terrain réel,
   l'ensemble de la végétation détectée, les bâtiments et les routes. Elle ne
   contient ni périmètre, ni caméra, ni timeline, ni vérité feu, ni Flow.
2. `fireviewer.omniverse-progressive-perimeter-layer-contract.v1` décrit un
   package de périmètres datés, produit séparément et attachable comme calque
   tracé et coloré sans muter le package carte.
3. `fireviewer.omniverse-simulated-case-production-contract.v1` décrit un cas
   simulé dérivé d'une révision de carte acceptée. Il ajoute une timeline, la
   révision exacte des périmètres, la destruction de la végétation, les effets
   Flow, le parc de caméras, les modalités et les barrières dataset.
4. `fireviewer.omniverse-reproducible-download-bundle-contract.v1` décrit le
   package autonome téléchargeable d'un cas simulé ou d'une reproduction. Ce
   contrat ne couvre ni l'upload de carte, ni l'implémentation du compte.

Les deux contrats sont candidats tant que la reproduction visible de Die n'a
pas reçu une acceptation humaine liée par SHA-256 à la scène examinée. Les
tests de schéma, l'ouverture Kit et un run sans capture ne remplacent pas cette
décision visuelle.

## Fichiers normatifs

- `map-upload-contract.schema.json` : contrat d'une carte sans simulation ;
- `perimeter-layer-contract.schema.json` : contrat des périmètres progressifs ;
- `simulated-case-production-contract.schema.json` : contrat d'un cas simulé ;
- `reproducible-download-bundle-contract.schema.json` : contrat du package
  autonome téléchargeable ;
- `production-profiles.v1.json` : valeurs verrouillées des profils de
  production ;
- `examples/die-map-upload.candidate.json` : projection candidate de la carte
  de Die ;
- `examples/die-progressive-perimeters.candidate.json` : périmètres candidats ;
- `examples/die-retrospective-case.candidate.json` : reproduction candidate de
  l'incendie de Die ;
- `examples/die-reproduction-download.candidate.json` : bundle téléchargeable
  candidat ;
- `validate_contracts.py` : validation JSON Schema et invariants inter-fichiers.

## Cycle de vie

`candidate_pending_die_visual_acceptance` signifie que le contrat peut être
testé, mais qu'il ne peut autoriser ni upload, ni attachement de calque, ni
capture, ni téléchargement, ni utilisation pour l'entraînement. Le passage à
`active` exige :

- un reçu humain `accepted` de la scène visible de Die ;
- une scène de carte autonome sans périmètre, caméra ou prim de simulation ;
- tous les gates de la carte à `passed` ;
- un reçu de simulation visible avant toute capture ;
- les empreintes finales des schémas, manifests et stages.

Le pipeline ne publie rien automatiquement. L'upload d'une carte, la capture
d'un dataset et l'admission à l'entraînement restent trois décisions séparées.

## Profil carte uploadable

Une carte conforme doit notamment fournir :

- `EPSG:2154`, élévations `NGF-IGN69`, unité mètre et axe Z vertical ;
- MNT et MNS co-localisés à 0,50 m au maximum ;
- terrain maillé visible et orthophoto réellement projetée sur ce terrain ;
- tous les arbres détectés par `MNS - MNT`, y compris les forêts denses, sans
  LOD dans ce profil ;
- bâtiments issus des empreintes source, avec toiture et ancrage MNT ;
- routes plates drapées sur le MNT ;
- aucune timeline, aucun feu, aucune fumée et aucune extension Flow active.

Les périmètres ne font jamais partie de l'upload de carte. Ils sont produits
comme 21 révisions append-only dans le cas Die, projetés en `EPSG:2154`, drapés
sur le terrain et composés par sous-couche dans une scène de revue ou dans un
cas. Le package de périmètres ne contient aucune caméra ni simulation.

## Profil cas simulé

Le cas simulé référence exactement une carte acceptée et une révision acceptée
du package de périmètres. C'est uniquement à ce niveau que sont ajoutées les
55 caméras humaines et les 7 caméras aériennes thermiques. La timeline conserve
`1 jour = 60 secondes réelles`. Chaque état sélectionne 20 plans de vue de
façon pseudo-aléatoire déterministe et produit 5 zooms du même plan. Les
caméras positives sont réorientées vers le point de flamme visible le plus
proche ; les vues négatives ne reçoivent jamais de feu fabriqué.

Le profil `synthetic_training_18d_v1` verrouille 18 jours, 10 états par jour,
20 plans par état et 5 zooms, soit 18 000 captures. Le profil
`retrospective_daily_replay_v1` suit les états quotidiens de la source ; le cas
Die actuel comporte 21 états, 420 plans et 2 100 captures potentielles.

Les cinq zooms d'un plan, tous les états d'un incident et toutes les modalités
associées restent dans le même split. Aucun dérivé d'une même situation ne
peut fuir entre entraînement, validation et test.

## Validation locale

```powershell
python omniverse/contracts/v1/validate_contracts.py
python -m pytest -q omniverse/test_omniverse_contracts.py
```

Ces commandes valident le contrat et ses invariants. Elles ne prouvent pas la
qualité visuelle, une capture RTX ou l'acceptation d'un dataset d'entraînement.
Un package téléchargeable n'est libérable qu'après inventaire SHA-256, scan des
chemins relatifs et réouverture Kit depuis une copie isolée.
