# FireViewer Spatial

Outils de production, validation et scellement des cartes 3D OpenUSD et des
timelines de périmètres observés de FireViewer. Les sorties lourdes vivent hors
Git ; ce dépôt contient uniquement le code, les contrats et de petites fixtures
synthétiques.

## Pipeline actif

Le moteur headless [`simple_production_api.py`](./blender/simple_production_api.py)
expose deux jobs séparés et séquentiels :

1. une carte carrée à partir d'un centre GPS et de la longueur d'un côté ;
2. une timeline supplémentaire de périmètres observés, liée à une carte déjà
   produite.

Une carte est découpée sur la grille Lambert-93 en tuiles de 500 m. Pour chaque
tuile, le pipeline acquiert temporairement MNT, MNS et orthophoto, puis produit :

- un relief fixe `terrain.fvtg` avec trois LOD déterministes ;
- une texture de sol orthophoto bakée, légère et locale ;
- un inventaire d'objets mesuré par `MNS−MNT` et confirmé par le contexte
  géographique disponible ;
- une scène OpenUSD autonome utilisant les assets embarqués et hashés.

Les rasters source sont supprimés seulement après validation de la tuile. Ils
ne figurent jamais dans le ZIP final. Les placements fixes optionnels sont
validés par le schéma
[`fixed_asset_placement_request.v1.schema.json`](./blender/fixed_asset_placement_request.v1.schema.json)
et prennent leur altitude sur le MNT.

La zone finale contient notamment :

```text
zone.usda
zone.blend
zone.done.json
zone-plan.json
zone-context.json
packages/<tile>/
shared/prototypes/
provenance/<tile>/
```

`zone.usda` et `zone.blend` sont les scènes unifiées autonomes. La production
active ne calcule plus de galerie PNG ; elle privilégie la génération du ZIP,
son contrôle dans l'administration et son ouverture indépendante.

## Périmètres observés

[`geographic_perimeter_layer.py`](./blender/geographic_perimeter_layer.py)
normalise un JSON/GeoJSON observé vers `EPSG:2154` et produit :

```text
geographic-perimeters.usda
fire-progression-timeline.json
perimeters.normalized.json
perimeter-layer.manifest.json
preview/perimeter-viewer.manifest.json
preview/frame-*.glb
```

Chaque état correspond à un instant observé ou à une plage temporelle
explicitement fournie. Entre observations, la valeur est `undefined` ; aucune
interpolation ni prédiction n'est inventée. Les GLB sont uniquement des vues
dérivées pour le navigateur. L'USD, le JSON normalisé et la timeline restent
les données de référence.

## Upload, simulation et datasets

[`portable_scene_package.py`](./blender/portable_scene_package.py) inventorie et
rehash chaque octet avant l'archive. Les contrats actifs sont :

- `fireviewer.simple-measured-map-package.v2` et
  `fireviewer.simple-measured-map-upload-contract.v2` ;
- `fireviewer.observed-perimeter-package.v1` et
  `fireviewer.observed-perimeter-upload-contract.v1`.

Le backend et le frontend importent le dossier extrait du ZIP, contrôlent ces
contrats, chaque hash, la scène Blender autonome et toutes les vues de timeline.
Ils ne convertissent et ne reconstruisent aucune donnée.

Une simulation, un dataset ou un replay lie ensuite les artefacts immuables via
[`scene-consumer-input.schema.json`](./contracts/spatial/v1/scene-consumer-input.schema.json).
Une carte publiée ou `technical_unpublished` est admissible pour un travail
interne. Le consommateur n'a jamais le droit de reconstruire le terrain ou les
périmètres et l'exécution de la simulation reste hors de ce dépôt.

Lors de la publication, le ZIP original de la carte reste un téléchargement de
la fiche incident. Les futurs packs de simulation sont des archives séparées,
liées au build de carte et à la timeline consommée ; ils ne remplacent jamais
le package de base.

## API et image Docker

Le contrat HTTP est décrit par
[`simple_production_api_contract.v1.json`](./blender/simple_production_api_contract.v1.json)
et le déploiement par
[`SIMPLE_PRODUCTION_POD.md`](./docs/SIMPLE_PRODUCTION_POD.md). L'API FastAPI
reste sans base de données et n'autorise qu'une production à la fois. Les
secrets sont fournis au démarrage ; ils ne sont jamais intégrés à l'image, aux
reçus ou aux archives.

## Référentiels spatiaux

- coordonnées d'entrée : `EPSG:4326`, ordre longitude/latitude dans les JSON ;
- production : `EPSG:2154`, grille de 500 m ;
- altitude : `NGF-IGN69` ;
- unités OpenUSD : mètres, axe vertical Z.

Les contrats événementiels et les outils de recalage restent distincts de la
production de carte. Ils peuvent référencer une révision spatiale ; ils ne la
modifient pas.

## Contrôles

En développement Windows, les temporaires du pipeline doivent être dirigés
vers `D:` :

```powershell
$env:TEMP = 'D:\Dev\project\fireviewer-repositories\fireviewer-work\temp'
$env:TMP = $env:TEMP
$env:PYTHONPYCACHEPREFIX = 'D:\Dev\project\fireviewer-repositories\fireviewer-work\cache\pycache'

python -m pytest -q
python -m ruff check .
```

Dans l'image Docker et sur le pod, l'espace de travail et les temporaires sont
placés sous `/work`, conformément au guide de déploiement.

Une suite locale verte ne prouve ni une acquisition IGN live, ni un rendu
Blender/Omniverse, ni un pod. Ces gates sont rapportés séparément.

## Données et licences

Ce dépôt Git n'embarque aucune carte, orthophoto, scène générée, archive, asset
3D, modèle, token ou dataset de production. Les sorties de l'ancien pipeline
ont été retirées localement ; `ground-context` et les ressources du pipeline
définitif restent des dépendances externes hashées. Le code est sous
AGPL-3.0-or-later ; la documentation est sous CC BY 4.0. Les données IGN et les
autres sources externes conservent leurs licences et attributions propres.
