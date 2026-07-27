# Scène de contrôle Blender — zone globale FireViewer

Ce dossier produit une scène Blender de contrôle visuel, isolée du site et de
son backend. La vérité géographique reste en `EPSG:2154`; Blender travaille en
mètres autour d'une origine Lambert-93 enregistrée dans la scène.

Le flux reste séparé en deux étapes :

1. `prepare_control_package.py` lit les rasters et vecteurs avec le Python du
   projet, puis produit un paquet `.json.gz` déterministe ;
2. `build_control_scene.py` construit la scène avec uniquement `bpy`,
   `mathutils` et la bibliothèque standard de Blender.

Le constructeur Blender n'importe donc ni `rasterio`, ni `shapely`, ni
`pyproj`, ni un `numpy` externe. Ce pipeline concerne uniquement la scène
globale ; il ne modifie pas le pipeline `detail/`.

## Géométrie de la scène v2

- `Terrain/TerrainPreview` : terrain validé, conservé à `--terrain-step 4` pour
  la livraison globale ;
- `Buildings/Buildings` : empreintes et extrusion déjà validées, inchangées ;
- `Vegetation/VegetationCanopy` : faces supérieures sur le MNS à 5 m et jupes
  périphériques descendant sommet par sommet jusqu'au MNT ;
- `Roads/Roads` : rubans issus des tronçons routiers BD TOPO, drapés sur le MNT,
  avec largeur de chaussée ou largeur dérivée de l'importance ;
- `Water/WaterCourses` : cours d'eau nommés, drapés sur le MNT ;
- `Water/WaterSegments` : tronçons hydrographiques avec largeur issue de leur
  classe BD TOPO ;
- `Water/WaterSurfaces` : polygones hydrographiques triangulés puis drapés sur
  le MNT ;
- `FirePerimeter` : périmètre source et limite du buffer d'analyse.

La canopée n'utilise plus `base_elevation_m` ou `block_height_m` comme un bloc
plat. Chaque sommet supérieur vaut `max(MNS, MNT)` et chaque sommet inférieur
visible vaut exactement le MNT. Les zones hors polygones de végétation ne
créent aucune face.

## Garde-fous géographiques

Le MNS est obligatoire lorsqu'une couche de végétation est fournie. Il doit
avoir exactement le même CRS Lambert-93, les mêmes dimensions, la même origine,
la même taille de pixel et la même transformation affine que le MNT. Toute
différence est rejetée avant la création de géométrie.

Le MNT et le MNS sources restent inchangés. Le pas 1 reste disponible pour un
diagnostic ponctuel du terrain, mais la livraison globale conserve
explicitement `--terrain-step 4`. La canopée utilise toujours la grille source
5 m, indépendamment du pas d'aperçu du terrain.

Les routes et couches hydro sont des entrées locales : le script ne télécharge
rien. Les options `--roads`, `--water-courses`, `--water-segments` et
`--water-surfaces` sont répétables pour accepter plusieurs pages GeoJSON.

## Portabilité

Le paquet et les propriétés de scène ne contiennent aucun chemin local absolu.
Ils stockent uniquement :

- le nom de chaque fichier source ;
- son SHA-256 ;
- le CRS, l'origine locale et les statistiques de traitement.

La scène stocke `preview_package_file_name` et `preview_package_sha256`, jamais
le chemin de travail ayant servi à ouvrir le paquet.

## Préparation finale recommandée

Les dépendances système sont listées dans `requirements.txt`.

```powershell
python -m pip install -r blender/requirements.txt

python blender/prepare_control_package.py `
  --mnt C:\tmp\synthetic-zone-mnt-5m.tif `
  --mns C:\tmp\synthetic-zone-mns-5m.tif `
  --perimeter .artifacts\spatial-lidar-surface\synthetic-zone-fire-2026-v1\vectors\fire-perimeter.l93.geojson `
  --perimeter-crs EPSG:2154 `
  --buildings .artifacts\spatial-lidar-surface\synthetic-zone-fire-2026-v1\vectors\buildings.l93.geojson `
  --buildings-crs EPSG:2154 `
  --vegetation .artifacts\spatial-lidar-surface\synthetic-zone-fire-2026-v1\vectors\vegetation.l93.geojson `
  --vegetation-crs EPSG:2154 `
  --roads C:\tmp\synthetic-zone-bdtopo-routes.l93.geojson `
  --roads C:\tmp\synthetic-zone-bdtopo-routes-page2.l93.geojson `
  --roads-crs EPSG:2154 `
  --water-courses C:\tmp\synthetic-zone-bdtopo-cours-eau.l93.geojson `
  --water-segments C:\tmp\synthetic-zone-bdtopo-troncons-hydro.l93.geojson `
  --water-surfaces C:\tmp\synthetic-zone-bdtopo-surfaces-hydro.l93.geojson `
  --water-crs EPSG:2154 `
  --buffer-m 1500 `
  --terrain-step 4 `
  --output .artifacts\spatial-lidar-surface\synthetic-zone-fire-2026-v1\blender\synthetic-zone-global-control-v2.json.gz
```

Remplacer `--output ...` par `--validate-only` contrôle les sources et construit
les statistiques sans écrire de paquet.

## Construction Blender

Le constructeur remplace le contenu de la scène active. Lancer cette étape
dans une session dédiée.

```powershell
& "C:\chemin\vers\blender.exe" --background `
  --python blender/build_control_scene.py -- `
  --package .artifacts\spatial-lidar-surface\synthetic-zone-fire-2026-v1\blender\synthetic-zone-global-control-v2.json.gz `
  --output .artifacts\spatial-lidar-surface\synthetic-zone-fire-2026-v1\blender\synthetic-zone-global-control-v2.blend
```

Depuis la console Python de Blender :

```python
FIREVIEWER_PACKAGE = r"D:\chemin\synthetic-zone-global-control-v2.json.gz"
FIREVIEWER_OUTPUT = r"D:\chemin\synthetic-zone-global-control-v2.blend"
exec(open(r"D:\chemin\vers\fireviewer\tools\spatial-hybrid-zone\blender\build_control_scene.py", encoding="utf-8").read())
```

## Détail global tuilé à 0,5 m

`prepare_global_05m.py` produit l'index portable
`fireviewer.global-05m-production-manifest.v1`. Le constructeur Blender ignore
les tuiles `pending` ou `incomplete` et crée une collection racine par tuile
`ready` :

```text
GlobalTiles
└── GlobalTile_x886500_y6400500_s500
    ├── Terrain_x886500_y6400500_s500
    └── Vegetation_x886500_y6400500_s500
```

Le terrain de chaque tuile conserve les altitudes MNT, reçoit sa propre
orthophoto IGN 0,5 m par UV Lambert-93, et garde l'image en fichier externe.
La végétation reste un maillage de points avec `Instance on Points` et des
prototypes partagés dans la tuile ; aucun nœud `Realize Instances` n'est admis.

Le mode par défaut `visible` évite de reconstruire un monolithe : l'index et
les collections de contrôle sont importés, mais seules les tuiles explicitement
sélectionnées, marquées visibles, ou intersectant le rayon de travail sont
matérialisées. `all_ready` doit rester un choix explicite.

```powershell
& "C:\chemin\vers\blender.exe" --background `
  --python blender/build_control_scene.py -- `
  --package .artifacts/spatial-lidar-surface/synthetic-zone-fire-2026-v1/blender/synthetic-zone-global-control-v2.json.gz `
  --orthophoto-source .artifacts/spatial-lidar-surface/synthetic-zone-fire-2026-v1/blender/synthetic-zone-ign-orthophoto-2m.source.json `
  --tile-index .artifacts/spatial-lidar-surface/synthetic-zone-fire-2026-v1/global-05m/production-manifest.json `
  --tile-focus-l93 888468.37 6400707.38 `
  --tile-visible-radius-m 750 `
  --output .artifacts/spatial-lidar-surface/synthetic-zone-fire-2026-v1/blender/synthetic-zone-global-tiled.blend
```

Une zone d'attention précise peut aussi être chargée avec un ou plusieurs
`--tile-id`. Le contrat terrain est exclusif : hors vue proche, seul
`TerrainPreview` (ortho globale 2 m) est rendu et aucune végétation 3D globale
n'est affichée. En vue proche, les tuiles MNT 0,5 m et tous leurs arbres ne sont
activés que si l'empreinte de vue complète est couverte, résidente et sous le
budget de 16 tuiles ; le terrain global est alors masqué et les tuiles restent
à un offset Z de `0 m`. Si une seule tuile manque, le système revient au terrain
global complet, sans îlot d'arbres ni superposition de deux sols.

Chaque maillage détaillé atteint les quatre limites exactes de son coeur 500 m.
Les altitudes de bord sont échantillonnées sur la phase native du raster IGN,
aux mêmes coordonnées Lambert-93 dans les deux tuiles adjacentes. Le chargeur
rejette un ancien paquet dépourvu de ce contrat de couture.

Pour reconstruire les paquets MNT/végétation et leurs reçus sans télécharger
ni modifier les orthophotos 0,5 m ou 0,2 m :

```powershell
python blender/prepare_global_05m.py `
  --output-root .artifacts/spatial-lidar-surface/synthetic-zone-fire-2026-v1/global-05m `
  --aoi .artifacts/spatial-lidar-surface/synthetic-zone-fire-2026-v1/vectors/area-of-interest.l93.geojson `
  --execute --phase tiles --rebuild-mid-packages `
  --package-workers 2 --memory-budget-gib 4 --minimum-free-gib 20 `
  --continue-on-error
```

Dans une session Blender déjà ouverte, la fonction
`apply_tiled_collection_visibility(bpy, (x_l93, y_l93), rayon_m)` actualise le
culling des collections déjà chargées. Elle ne télécharge ni ne matérialise une
tuile absente ; pour cela, relancer `load_global_tiles_into_scene` avec le même
index et les identifiants voulus.

## Coût prévu

Mesure effectuée sur les masques locaux actuels :

- terrain à pas 4 : environ `268 364` sommets et `267 087` faces ;
- végétation couverte : `83,564075 km²` ;
- canopée : `3 342 563` faces supérieures, environ `3 444 840` sommets
  supérieurs et `208 110` faces de jupe, soit `3 550 673` faces avant les
  sommets inférieurs de bord ;
- routes : `108 380` sommets et `50 352` faces ;
- cours d'eau : `18 078` sommets et `8 845` faces ;
- tronçons hydro : `80 462` sommets et `37 282` faces ;
- surfaces hydro : `9 777` sommets et `14 208` triangles.

**INFÉRÉ**, tant que le paquet complet v2 n'a pas été généré et mesuré : le
JSON gzip devrait peser plusieurs dizaines à quelques centaines de mégaoctets,
et le pic mémoire du prétraitement/chargement peut atteindre plusieurs
gigaoctets. Prévoir au minimum 16 Go de RAM disponible pour la scène complète.
Le coût vient presque entièrement de la canopée à 5 m ; les routes et l'hydro
restent secondaires.

## Tests hors Blender

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'

python -m py_compile `
  blender/spatial_data.py `
  blender/prepare_control_package.py `
  blender/build_control_scene.py `
  blender/test_spatial_data.py `
  blender/test_build_control_scene.py

python -m unittest discover `
  -s blender `
  -p "test_*.py"
```

Les tests synthétiques vérifient notamment la canopée sur pente, le contact des
jupes avec le MNT, le dessus au MNS, l'alignement/rejet des rasters, le drapage
des deux bords routiers, la triangulation hydro et la portabilité des
métadonnées.
