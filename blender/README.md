# Terrain adaptatif FireViewer
Ce dossier contient le pipeline terrain actif, indépendant du site, du backend
et des futures simulations. La vérité géographique reste en `EPSG:2154` et les
sorties lourdes sont écrites hors Git, exclusivement sur `D:`.

## Composants actifs et compatibilité

| Composant | Rôle |
| --- | --- |
| `terrain_source_acquisition.py` | acquisition/reprise MNT/MNS 2 m, contrôle CRS et extraction canonique avec halo |
| `orthophoto_build_acquisition.py` | source RGB temporaire WMS/WMTS à 1 m par fenêtre de 500 m + halo ; jamais une texture runtime |
| [`orthophoto_surface_correspondence.py`](./orthophoto_surface_correspondence.py) | correspondance déterministe multiscale vers les 72 profils PBR stables et quatre cartes 500×500 |
| `adaptive_terrain_quadtree.py` | arbre maître LOD0, effondrement LOD1/LOD2 et codec FVTQ v1 |
| `compact_hag.py` | réservation `MNS−MNT` en centimètres dans `hag-max-2m.tif` |
| `compile_tile_composition.py` | lecteur/qualification de la composition de sol v2 ; aucune nouvelle production |
| `ground_material_contract.py` | `ground-material-contract.v2`, quatre atlas PBR propres partagés et shader runtime fail-closed |
| `tile_package.py` | nouvelle production `tile-package.v3.json`/`tile.done.v3.json` et lecture seule des packages v2 |
| `frustum_streaming.py` | résidence LOD pilotée par frustum et transitions atomiques |
| `build_adaptive_terrain_fixture.py` | qualification synthétique contiguë 2×2 |
| `validate_adaptive_terrain_usd.py` | import Blender headless, composition PBR de référence `world_xy`/`world_triplanar` et contrôle de l'AOV LOD |
| `validate_adaptive_terrain_zone.py` | vues de zone complètes, 3×3, coutures et obliques, puis gate humain séparé |
| [`prepare_adaptive_zone_specs.py`](./prepare_adaptive_zone_specs.py) | génération hors ligne d'un unique `zone-spec.v1` 2 m |

L'orchestration mono-zone et son backend concret se trouvent à la racine dans
`terrainctl.py` et `terrain_production_backend.py`. L'export
OpenUSD portable est dans `omniverse/adaptive_terrain_usd.py`. Les contrats
publics sont sous `contracts/terrain/v1`.

## Contrat géométrique

Une tuile coeur mesure 500 m et sa préparation utilise un halo de 10 m. Le MNT
et le MNS de travail sont co-enregistrés à 2 m. Les hauteurs sont quantifiées au
millimètre avant toute décision de subdivision.

| LOD | Usage | Erreur verticale | Cellule plane maximale | Raffinement local |
| --- | --- | ---: | ---: | ---: |
| LOD0 | pixels de la caméra principale | 0,50 m | 31,25 m | 3,906 m, 1,953 m sur rupture |
| LOD1 | périphérie et préchauffage | 2 m | 125 m | 15,625 m |
| LOD2 | horizon et reste du carré | 8 m | 500 m | 62,5 m |

LOD1 et LOD2 sont des effondrements du même arbre maître LOD0. Ils ne sont pas
recalculés séparément. Les sommets sont ordonnés par Morton et les hauteurs sont
stockées en millimètres relatifs au datum de la tuile. Les frontières partagent
leurs sommets, hauteurs et signatures ; l'équilibrage 2:1 et la triangulation
des transitions empêchent les trous.

Le package v3 produit par les fixtures et attendu pour toute nouvelle tuile
contient :

```text
terrain-lod0.fvtq
terrain-lod1.fvtq
terrain-lod2.fvtq
hag-max-2m.tif
ground-profile-ids.png
ground-profile-weights.png
ground-confidence.png
ground-orientation.png
surface-correspondence.json
source/mnt-provenance.v1.json
source/mns-provenance.v1.json
tile-package.v3.json
tile.done.v3.json
terrain-lod0.usda
terrain-lod1.usda
terrain-lod2.usda
terrain-tile.usda
terrain-usd-package.v1.json
```

Les FVTQ, le HAG, la provenance compacte et leurs reçus constituent la vérité
reproductible. Les GeoTIFF MNT/MNS sources et caches ne sont pas nécessaires
pour régénérer Blender ou OpenUSD après acceptation.

Les anciens `surface-overlays.json.gz`, `tile-composition.json.gz`,
`tile-package.v2.json` et `tile.done.v2.json` appartiennent au chemin de
compatibilité v2. Ils restent lisibles pour les tests, le replay et l'audit,
mais ne sont plus une sortie admissible pour une nouvelle production. Le
package v3 contient quatre cartes de coeur de 500 × 500 pixels : IDs, weights,
confidence et orientation. Leur production et leur scellement sont implémentés
par
[`orthophoto_surface_correspondence.py`](./orthophoto_surface_correspondence.py)
et `tile_package.py` ; leurs contrats verrouillent les noms, canaux et hashes.

Les reçus canoniques exposent aussi, pour chaque LOD, les seize variantes de
raccord FVTQ et leurs coûts/hashes. Aux jointures, la QA compare les signatures
effectives des sept couples de LOD compatibles avec l'équilibrage 2:1. Le HAG
est dérivé directement des millimètres entiers des deux halos canoniques.

## Sols 2D cibles

Une fenêtre orthophoto RGB temporaire couvre le coeur 500 m et son halo de
10 m, soit 520 × 520 pixels à 1 m en `EPSG:2154`. Le compilateur implémenté
classifie son coeur vers `ground-profile-ids.png` et
`ground-profile-weights.png` en RGBA8, puis `ground-confidence.png` et
`ground-orientation.png` en L8. `surface-correspondence.json` hash-locke les
entrées et les quatre cartes sans conserver de chemin ou pixel orthophoto. Les
réponses WMS/WMTS, `.part` et le RGB canonique ne sont supprimés qu'après
l'écriture de `tile-package.v3.json` et la validation de `tile.done.v3.json`
contre toutes les sorties dépendantes, avant l'acquisition de la fenêtre
suivante.

Les cartes ne peuvent référencer que 72 textures PBR propres empaquetées dans
quatre atlas (`basecolor`, `normal`, `height`, `ORM`). Cette bibliothèque est
encore à produire et à accepter. L'atlas v3 actuel est rejeté ; aucun matériau
procédural, profil `procedural_only`, fallback orthophoto ou image source
rapprochée n'est importable dans la scène.

Le `ground-material-contract.v2` lie la bibliothèque, les quatre atlas, les 72
profils et leurs projections. Blender constitue l'implémentation de référence
pour la QA terrain : il échantillonne réellement les atlas en `world_xy` ou en
`world_triplanar` à partir du LOD0 FVTQ. La qualification technique synthétique
est passée avec Blender 4.5.3 en 512 px, zéro pixel LOD/couverture invalide et
44 544 pixels triplanaires. Son reçu reste
`textured_technical_probe_pending_surface_library` : aucune bibliothèque réelle
ni revue humaine n'est donc implicitement acceptée.

La couche USD portable ne prétend pas réaliser ce mélange PBR : son
`UsdPreviewSurface` magenta est uniquement diagnostique. Le binding runtime
reste `pending_dedicated_mdl_validation`, avec `usd_runtime_gate=false`, tant
qu'un shader MDL dédié hash-locké n'a pas été validé. Cette attente ne bloque
pas l'acceptation terrain texturée dans Blender ; elle bloque la qualification
du runtime texturé OpenUSD/Omniverse.

Les parcelles, l'occupation du sol, les transports, l'hydrographie et la
géologie servent seulement de priors et de corrections de la classification.
L'ancienne grille de fond 5 m et ses overlays vectoriels v2 sont dépréciés en
écriture et restent disponibles en lecture seule pour les anciens packages.
Voir
[`ORTHOPHOTO_TEXTURE_CORRESPONDENCE.md`](../docs/ORTHOPHOTO_TEXTURE_CORRESPONDENCE.md)
pour le chemin cible et les gates encore ouverts, et
[`ORTHOPHOTO_SURFACE_CORRESPONDENCE.md`](./ORTHOPHOTO_SURFACE_CORRESPONDENCE.md)
pour l'API et les invariants du matcher.

## Streaming caméra

La sélection radiale et le plafond de seize tuiles sont retirés du chemin
actif. Le planificateur teste les AABB 3D contre le frustum réel et construit :

- `visible_lod0` pour les intersections exactes ;
- `guard_lod0` pour la couronne de 500 m ;
- `staging_lod0` pour la caméra suivante ou prédite ;
- `resident_lod1` pour deux couronnes supplémentaires ;
- `resident_lod2` pour le reste de la zone.

La caméra principale ne publie que si tout son ensemble visible LOD0 est
résident et validé. LOD1 ou LOD2 ne servent jamais de fallback visible. Une
transition charge le staging en double buffer, vérifie hashes et budget, puis
publie caméra et LOD0 atomiquement. Trois échecs mettent un payload en
quarantaine.

Le budget couvre le LOD0 actif, de garde et de staging, la couronne LOD1, tout
le LOD2 et 25 % de réserve. Un dépassement échoue avant le rendu ; aucun far
clip, cadrage ou LOD visible n'est modifié silencieusement.

## Qualification locale

Tous les temporaires de ces commandes doivent être dirigés vers `D:` :

```powershell
$env:TEMP = 'D:\Dev\project\fireviewer-repositories\fireviewer-work\temp'
$env:TMP = $env:TEMP
$env:PYTHONPYCACHEPREFIX = 'D:\Dev\project\fireviewer-repositories\fireviewer-work\cache\pycache'

python blender/build_adaptive_terrain_fixture.py `
  --output D:\Dev\project\fireviewer-repositories\fireviewer-work\qa\adaptive-fixture-2x2-v1
```

La fixture produit quatre tuiles continues, douze FVTQ, les HAG, les quatre
cartes synthétiques v3, les payloads USD, un catalogue de coûts, les contrôles
des coutures et une seconde compilation bit à bit. Elle conserve la lecture v2
uniquement dans des tests de compatibilité dédiés.

Cette fixture est une qualification synthétique. Elle ne vaut pas acceptation
visuelle de la future bibliothèque PBR et aucune des six zones de production
n'est produite ou acceptée par ce dépôt au 9 août 2026. Le gate atlas exhaustif
v3 est actuellement `failed_technical_render` : 24 profils sont sombres ou
plats dans la bande micro et quatre profils `road_surface` sont encore
`procedural_only`. Aucun reçu
`pending` ou `accepted_blender_visual` n'a été émis. Les diagnostics légers sont
conservés sous `fireviewer-work/qa/ground-atlas-v3-acceptance-probe/` et
`fireviewer-work/qa/adaptive-atlas-v3-probe/` sur `D:`.

L'acquisition orthophoto temporaire possède uniquement des tests synthétiques :
aucune orthophoto réelle n'a été téléchargée et aucune carte de correspondance
500 × 500 n'a encore été produite pour les six zones. Le matcher et son contrat
sont néanmoins implémentés : ses six tests synthétiques couvrent les six
classes de référence, les restrictions/corrections et l'identité bit à bit
entre compilation groupée et séparée. Cela ne débloque ni la bibliothèque PBR
réelle, ni sa revue visuelle par le gate Blender, ni l'état `0/6`.

Le contrôle Blender utilise le Blender LTS déjà installé sur `D:` :

```powershell
& $blender --background --factory-startup --disable-autoexec --offline-mode `
  --python-exit-code 1 `
  --python blender/validate_adaptive_terrain_usd.py -- `
  --package D:\...\tiles\x700000_y6300000 `
  --report D:\...\accepted_blender_textured_visual.json `
  --render-exr D:\...\terrain-lod-aov.exr `
  --coverage-exr D:\...\terrain-coverage-aov.exr `
  --beauty-topdown D:\...\terrain-topdown.png `
  --beauty-oblique D:\...\terrain-oblique.png `
  --atlas-acceptance-receipt D:\...\atlas.accepted_blender_visual.v1.json `
  --resolution 512
```

Le reçu terrain Blender exige notamment l'import USD avec les comptes FVTQ
attendus, tous les pixels terrain de l'AOV `fireviewer:terrain_lod` à zéro, la
composition PBR de référence réellement texturée et l'acceptation visuelle de
la bibliothèque. Il est indépendant du gate du futur shader runtime USD/MDL.

`terrainctl.py` redirige également `TEMP`, `TMP`, `PYTHONPYCACHEPREFIX`, le
profil, les scripts et les extensions Blender vers le run sur `D:`. Le chemin
et le SHA-256 de l'exécutable Blender sont verrouillés dans la dépendance
`toolchain` ; aucun téléchargement de Blender n'est effectué par le pipeline.

## Pipeline historique déprécié

Les scripts `prepare_incident_terrains.py`, `prepare_global_05m.py`, `tile_streaming.py`,
`blender_tile_streaming_runtime.py` et les branches correspondantes de
`build_control_scene.py` restent uniquement pour rejouer les anciennes preuves
et préserver les tests de compatibilité. Leur grille uniforme, leur sélection
radiale, leur fallback global et leurs orthophotos ne sont pas admissibles dans
une nouvelle production.

La composition de sol v2 (`compile_tile_composition.py`, grille 5 m et
`surface-overlays.json.gz`) suit la même règle : lecture, replay et audit
uniquement. Les données SIG ne reviendront dans le nouveau compilateur que
comme priors ou corrections hashées.

Le dataset de la première simulation et son pack complet restent une archive
autonome hors Git. Die ne réutilise aucun ancien terrain, sol ou placement.

La phase terrain ne produit ni bâtiment, ni route 3D, ni rail métallique, ni
petit asset, ni végétation, ni incendie, ni simulation.
