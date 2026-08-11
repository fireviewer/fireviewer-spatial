# FireViewer Spatial

Outils source pour préparer, valider et intégrer les packages OpenUSD et
Omniverse de FireViewer. Le runtime Unity et ses scènes de site sont retirés du
périmètre actif.

Ce dépôt ne contient aucune carte, orthophoto, scène générée, archive de
production, dossier d’upload, modèle 3D, atlas, arbre source ou donnée
d’incident. Les packages validés sont publiés séparément dans GitHub Releases
ou un stockage Blob.

## Position dans l’architecture événementielle v2

Ce dépôt fournit les référentiels et contrats géométriques utilisés pour
localiser des phénomènes actifs à partir de contributions documentées. Il ne
déduit pas une localisation depuis un texte et ne publie aucune géométrie.

Une branche spatiale admissible relie explicitement :

- le point de prise de vue privé et son incertitude ;
- le modèle de caméra, l’orientation et le profil de vue ;
- l’ancrage visuel rattaché à une preuve ;
- la révision immuable du terrain et de la banque de rendus ;
- la pose, le raycast ou la triangulation ;
- le résultat géométrique, son incertitude et ses contrôles.

En l’absence d’orientation, de pose, d’intersection terrain ou de précision
suffisante, le résultat reste un secteur ou une abstention. Le point de prise
de vue n’est jamais assimilé au point actif. Les matchers cross-view restent en
benchmark ou en shadow tant qu’un benchmark événementiel indépendant ne
justifie pas leur promotion.

## Contenu

- `blender/` : génération et contrôles géométriques ;
- `omniverse/` : contrats OpenUSD, composition déterministe et outils Kit ;
- `omniverse/contracts/v2/` : frontière définitive du catalogue de 295 assets
  et des vingt compositions reproductibles ;
- `contracts/spatial/` : schémas du package et du catalogue ;
- `tests/` : fixtures synthétiques sans données de production.

## Production

Chaque production utilise un dossier de travail externe et une configuration de
zone explicite. Les entrées LiDAR, orthophotos, vecteurs, modèles et rendus
restent hors du checkout. Les manifestes générés conservent provenance,
licences, tailles et SHA-256.

La validation automatisée ne remplace ni l'ouverture isolée dans Kit, ni la
revue visuelle humaine. Aucun nouveau plan de scène ou pack n'est produit tant
que les 295 assets USD refaits ne sont pas reçus et acceptés.

Le dataset historique issu de la première simulation et son pack autonome de
reproduction complet sont conservés hors Git. Ils servent au replay et à
l'audit, mais ne définissent pas le contrat des futures scènes.

Les packages et banques de rendus sont versionnés par zone et par révision. Les
sorties de localisation conservent les CRS horizontaux, datums verticaux,
transformations, résolutions et empreintes exactes nécessaires au replay.

## Documentation de référence

- `docs/CAMERA_AND_CRS_CONTRACT.md` : repères, caméra, pose et contrôles ;
- `docs/PACKAGE_VERSIONING.md` : immutabilité et dépendances des révisions ;
- `docs/RENDER_BANK_SPEC.md` : contenu et portée des banques de rendus ;
- `docs/UAV_REGISTRATION_BENCHMARK.md` : protocole de comparaison des matchers.

La doctrine produit, les contrats transverses et la matrice d’acceptation sont
maintenus dans le dépôt canonique `fireviewer/Fireviewer_doc`.

## Contrôles

```powershell
python -m unittest discover -s tests -v
python -m pytest -q
python -m ruff check .
python omniverse/contracts/v2/validate_contracts.py
```

Ces contrôles valident le code et les contrats. Ils ne prouvent pas une scène
Omniverse, un rendu RTX, un package cartographique ou une publication distante.

## Licences

Le code est sous AGPL-3.0-or-later et la documentation sous CC BY 4.0. Les
sources IGN, OpenStreetMap et autres données externes conservent leurs licences
propres et ne sont pas redistribuées dans ce dépôt.
