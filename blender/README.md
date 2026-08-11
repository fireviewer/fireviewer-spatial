# Production simple des cartes FireViewer

Ce dossier contient le pipeline actif de production mesurée. Il transforme une
emprise GPS en carte OpenUSD autonome sans stocker les rasters source dans le
livrable.

## Chaîne active

| Module | Rôle |
| --- | --- |
| `simple_production_api.py` | API FastAPI des jobs carte et périmètres, état fichier, reprise et téléchargement |
| `simple_production_gradio.py` | planification de zone et orchestration historique conservée comme bibliothèque Python, sans UI Gradio active |
| `prepare_simple_measured_zone_context.py` | contexte géographique hashé de la zone |
| `prepare_simple_measured_tile_sources.py` | acquisition temporaire MNT/MNS/orthophoto d'une tuile |
| `fixed_terrain_grid.py` | relief FVTG fixe, trois LOD et codec déterministe |
| `orthophoto_ground_texture.py` | texture de sol bakée légère, alignée sur la grille |
| `mns_mnt_placement_inventory.py` | candidats mesurés par MNS−MNT, confirmations et exclusions contextuelles |
| `fixed_asset_placement.py` | placements GPS explicites vers une tuile propriétaire et une altitude MNT |
| `produce_simple_measured_tile.py` | scellement atomique d'une tuile complète |
| `build_reference_usd_asset_library.py` | catalogue complet d'identités et substitution déterministe par un USD réel compatible |
| `render_simple_zone_gallery.py` | `zone.blend` et 20 captures de contrôle |
| `portable_scene_package.py` | inventaire byte-for-byte, contrats d'upload et ZIP déterministe |
| `geographic_perimeter_layer.py` | calques USD et timeline observée sans interpolation |
| `geographic_perimeter_viewer.py` | GLB dérivés de contrôle sur une carte validée |

Les exporteurs OpenUSD sont
[`fixed_terrain_usd.py`](../omniverse/fixed_terrain_usd.py) et
[`build_measured_scene_usd.py`](../omniverse/build_measured_scene_usd.py).

## Invariants carte

- entrée : centre `EPSG:4326` et côté du carré en kilomètres ;
- grille : `EPSG:2154`, tuiles coeur de 500 m ;
- altitude : `NGF-IGN69` ;
- relief : MNT, FVTG et trois LOD déterministes ;
- objets : positions et hauteurs issues de MNS−MNT, confirmées autant que
  possible par le contexte ;
- sol : orthophoto bakée par tuile, jamais la mosaïque source brute ;
- assets : USD et textures rehashés, bundle autonome, aucun cube noir ;
- sortie : `zone.usda`, `zone.blend`, packages de tuiles, prototypes utilisés,
  reçus et 20 captures ;
- stockage : travail temporaire sur `D:`, aucun artefact FireViewer durable sur
  `C:`.

Un asset explicitement placé garde son identité de catalogue, son point GPS et
son yaw. Son altitude vient du terrain. Un asset manquant utilise un donneur
USD réel compatible et déterministe ; le reçu conserve l'identité demandée et
celle du donneur. L'ajout ultérieur du vrai USD sous le même identifiant remplace
automatiquement le donneur sans changer la demande de placement.

## Invariants périmètres

- seuls des instants observés ou des plages explicites sont admis ;
- affected et active restent deux catégories distinctes ;
- `between_observations=undefined`, `prediction=none` ;
- chaque package cible exactement le package, la révision, la zone, le build et
  le contrat de sa carte ;
- les GLB de contrôle sont non autoritatifs et re-générables depuis la timeline ;
- produire une timeline supplémentaire ne reconstruit jamais la carte.

## Validation

Le package carte n'est scellé qu'après :

1. validation de chaque tuile et de ses dépendances ;
2. composition de la scène unifiée ;
3. vérification des assets utilisés ;
4. génération et hash des 20 captures ;
5. suppression des rasters source ;
6. inventaire final et relecture complète du contrat portable.

Le package périmètre est scellé après validation du JSON/GeoJSON normalisé, de
l'USD, de la timeline, du lien à la carte et de chaque GLB de contrôle.

Les tests synthétiques n'acceptent pas humainement une scène et ne remplacent
pas un smoke Blender, une acquisition live ou un pod.

## Interface locale

L'interface n'est pas embarquée dans l'image de production. Elle est montée
uniquement dans l'espace authentifié `fireviewer-frontend` à la route
`/admin/production` et appelle l'API FastAPI sans modifier le moteur ni ses
contrats.

Voir [`SIMPLE_PRODUCTION_POD.md`](../docs/SIMPLE_PRODUCTION_POD.md) pour le
build, le démarrage, les sorties et la reprise.
