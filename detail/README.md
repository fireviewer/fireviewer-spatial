# Détail LiDAR contrôlé — Montmaur

Ce sous-pipeline prépare uniquement une zone locale Montmaur à partir de
fichiers déjà présents sur la machine. Il n'effectue aucun téléchargement.

## Outils observés sur la machine

- disponibles : Python 3.11, `laspy 2.7`, `lazrs`, `numpy`, `scipy`,
  `scikit-learn`, `rasterio`, `shapely`, `pyproj` ;
- absents du `PATH` : PDAL, `gdalinfo`, `ogr2ogr`, `lasinfo` ;
- non requis par ce traitement : GeoPandas, Fiona et Open3D.

La production lit les COPC locaux avec `laspy.CopcReader`, sans convertir le
nuage en maillage. Les fichiers LAS ordinaires ne sont acceptés que par les
tests synthétiques ; la CLI de production exige un vrai COPC.

## Entrées

- AOI Montmaur GeoJSON, WGS84 ou EPSG:2154 ;
- un ou plusieurs COPC IGN LiDAR HD classifiés ;
- MNT et MNS alignés, idéalement à 0,5 m ;
- empreintes de bâtiments BD TOPO ;
- axes de haies BD TOPO.

Exemple, lorsque les fichiers locaux sont disponibles :

```powershell
python detail/prepare_montmaur_detail.py `
  --aoi C:\data\montmaur-aoi.l93.geojson `
  --mnt C:\data\montmaur-mnt-tile-1-0m50.tif `
  --mnt C:\data\montmaur-mnt-tile-2-0m50.tif `
  --mns C:\data\montmaur-mns-tile-1-0m50.tif `
  --mns C:\data\montmaur-mns-tile-2-0m50.tif `
  --copc C:\data\LHD_FXX_tile-1.copc.laz `
  --copc C:\data\LHD_FXX_tile-2.copc.laz `
  --buildings C:\data\montmaur-buildings.geojson `
  --hedges C:\data\montmaur-hedges.geojson `
  --output-dir .artifacts\spatial-hybrid-zone\montmaur-detail-v1
```

Le dossier de sortie doit être absent. Cette règle évite d'écraser une
révision déjà contrôlée. Les options `--mnt`, `--mns` et `--copc` sont à
répéter pour chaque tuile intersectant le carré. Les rasters doivent partager
la même grille native à 0,5 m ; le mosaïquage ne décale ni ne rééchantillonne
les altitudes.

Avant une requête exhaustive, le contrôle d'en-têtes et d'emprises s'exécute
avec `dry_run_montmaur.py`. Il ne décompresse aucun point COPC et marque ses
estimations de volume comme `INFÉRÉ` :

```powershell
python detail/dry_run_montmaur.py `
  --zone-contract detail_zones.v1.json `
  --source-root .artifacts/spatial-lidar-surface/synthetic-zone-fire-2026-v1/detail/montmaur/sources `
  --buildings .artifacts/spatial-lidar-surface/synthetic-zone-fire-2026-v1/vectors/buildings.l93.geojson `
  --hedges C:\tmp\synthetic-zone-bdtopo-haies.geojson `
  --output .artifacts/spatial-lidar-surface/synthetic-zone-fire-2026-v1/detail/montmaur/dry-run-inputs.json
```

## Méthodes

### Bâtiments

L'empreinte reste celle de BD TOPO, découpée à l'AOI. La base est la médiane
des pixels MNT inclus. Le toit est d'abord le percentile 95 des retours COPC
de classe 6 dans l'empreinte ; en l'absence d'assez de retours classifiés, le
percentile 75 du MNS sert de repli. Si le sol ou le toit observé est
insuffisant, la hauteur reste `null`. Aucune hauteur par défaut n'est ajoutée.

### Haies

Chaque axe BD TOPO demeure une entité distincte. Les retours de végétation à
moins du rayon d'association sont affectés à l'axe le plus proche. La hauteur
est le percentile 75 au-dessus du MNT et la largeur vaut deux fois le
percentile 90 de la distance latérale à l'axe. Les points ainsi affectés sont
retirés des candidats arbres. Une haie insuffisamment observée conserve des
hauteurs et largeurs `null`.

### Arbres détectés

Le traitement :

1. normalise les retours des classes 3, 4 et 5 par le MNT ;
2. retient des maxima locaux sur une grille déterministe ;
3. supprime les apex voisins selon un rayon dépendant de la hauteur ;
4. affecte chaque retour de couronne à l'apex admissible le plus proche ;
5. conserve l'apex réellement observé et mesure le diamètre par deux fois le
   percentile 95 des distances radiales ;
6. publie, lorsqu'elle n'est pas dégénérée, l'enveloppe convexe des retours de
   couronne affectés.

Les seuils sont enregistrés dans `detail-manifest.json`. Les identifiants sont
dérivés des coordonnées centimétriques des apex observés et sont stables pour
des entrées et paramètres identiques.

## Sorties

- `buildings.l93.geojson` ;
- `trees-detected.l93.geojson` ;
- `tree-crowns-detected.l93.geojson` ;
- `hedges.l93.geojson` ;
- `detail-manifest.json` avec sources, hashes, paramètres, statistiques et
  limites de validité.

## Limite obligatoire : « chaque arbre »

Le résultat est un catalogue de **couronnes détectées**, pas un inventaire
exhaustif des arbres physiques. Le LiDAR aérien peut ne pas observer les
troncs ou arbres dominés ; deux couronnes jointives peuvent être fusionnées ;
une même couronne complexe peut produire plusieurs maxima ; les classes
source peuvent être erronées ; les arbres proches d'une haie BD TOPO sont
volontairement exclus des candidats arbres.

Il est donc possible de vérifier que chaque entité publiée repose sur des
retours observés, mais impossible d'affirmer que « chaque arbre » réel est
présent sans relevé terrain et protocole de validation indépendant.

## Tests

```powershell
python -m pytest detail/test_prepare_montmaur_detail.py -q
```

Les fixtures couvrent le déterminisme, les couronnes voisines, la séparation
des haies, les observations clairsemées, les hauteurs négatives, le NoData,
les replis bâtiment, la déduplication et le refus d'un LAS ordinaire par le
contrat de production.
