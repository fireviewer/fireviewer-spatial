# Export spatial FireViewer pour Unity

Ce dossier produit un catalogue distant indépendant de la plateforme et des
ressources immuables. Il ne publie rien sur Internet : le dossier de sortie
peut ensuite être copié tel quel vers un stockage objet ou un CDN.

## Contrat produit

- Le catalogue conserve `EPSG:2154`, les mètres et les axes
  `X=east,Y=north,Z=up`.
- Le terrain détaillé est la grille réelle de chaque paquet de 500 m. Seules
  les altitudes sont quantifiées en `UInt16`; le pas, l'intervalle et l'erreur
  maximale observée sont inscrits dans l'en-tête.
- Tous les arbres acceptés par le paquet source sont exportés. L'exporteur ne
  fait aucun échantillonnage supplémentaire. Leur position et leur altitude
  restent millimétriques. L'encodage courant conserve les dimensions au
  centimètre afin de représenter aussi les mesures LiDAR dépassant 65,535 m,
  sans les plafonner ni les supprimer; le lecteur Unity accepte également les
  tuiles v1 antérieures dont les dimensions étaient millimétriques.
- Les bâtiments, routes et eaux sont recalés par `detail_vector_lod.py`, puis
  convertis en maillages triangulés compacts. Les coordonnées vectorielles
  gardent une erreur horizontale bornée à environ 3,82 mm sur une tuile de
  500 m; l'erreur exacte observée est inscrite par section.
- Chaque `.fwtile` possède cinq sections obligatoires : `terrain`, `trees`,
  `buildings`, `roads`, `water`. Chaque section est compressée et possède ses
  propres tailles et SHA-256 brut/stocké.
- L'imagerie proche choisit la version IGN 0,20 m lorsqu'elle est marquée
  `ready`; sinon elle exige la version 0,50 m. Une ressource manquante ou dont
  le SHA diverge arrête l'export.
- Le LOD lointain reste toujours disponible comme repli. Dans les artefacts
  actuels, l'imagerie globale est à 2 m mais le MNT global réel est à 5 m. Le
  catalogue conserve ces deux résolutions sans présenter le MNT comme un 2 m.
- Le budget global est de 16 tuiles détaillées résidentes, avec préchargement
  à 750 m et publication atomique à 600 m.

## Format binaire v1

Le préfixe little-endian est `8s magic + UInt16 major + UInt16 minor + UInt32
headerLength`, suivi d'un en-tête JSON canonique puis des sections. Les URLs du
catalogue sont relatives et contiennent le SHA-256 du contenu. Le lecteur doit
vérifier le SHA du fichier, puis les SHA stocké et brut de chaque section avant
de créer les objets Unity.

Mapping Unity recommandé en mètres :

```text
Unity X = east  - origin east
Unity Y = up    - origin up
Unity Z = north - origin north
```

Le facteur historique FireViewer de 100 unités par mètre est décrit dans le
catalogue, mais n'est pas appliqué aux données binaires.

## Commandes

Canary d'une tuile (le mode à utiliser avant une production complète) :

```powershell
python unity/export_remote_catalog.py `
  --artifact-root .artifacts/spatial-lidar-surface/synthetic-zone-fire-2026-v1 `
  --tile-id x888000_y6400000_s500
```

Production robuste de toutes les tuiles prêtes :

```powershell
python unity/run_export_batches.py `
  --artifact-root .artifacts/spatial-lidar-surface/synthetic-zone-fire-2026-v1 `
  --output-root .artifacts/spatial-lidar-surface/synthetic-zone-fire-2026-v1/unity-remote-catalog `
  --workers 1 `
  --batch-size 8
```

La production est reprise à partir des reçus atomiques sous `receipts/`. Un
reçu n'est réutilisé que si tous les SHA d'entrée et de sortie restent valides.
Le catalogue mutable `catalog.json` ne référence que des ressources immuables.
Le worker unique est intentionnel : le modèle vectoriel global occupe environ
3,5 Gio une fois décodé. Les processus courts bornent les allocations natives
GDAL/GEOS tout en amortissant ce chargement sur plusieurs tuiles.

Audit complet avant publication :

```powershell
python unity/validate_remote_catalog.py `
  --artifact-root .artifacts/spatial-lidar-surface/synthetic-zone-fire-2026-v1 `
  --output-root .artifacts/spatial-lidar-surface/synthetic-zone-fire-2026-v1/unity-remote-catalog
```

Cet audit recoupe les identifiants avec le manifeste, relit et décompresse les
cinq sections de chaque tuile, vérifie chaque taille/SHA-256, les reçus, le LOD
lointain, le budget résident et calcule le volume de déploiement référencé.

Vérifications ciblées :

```powershell
python -m pytest unity/test_fwtile.py -q
ruff check unity
ruff format --check unity
```
