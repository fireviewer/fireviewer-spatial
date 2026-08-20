# Déploiement de la production spatiale FireViewer

Ce document décrit l’état actuel du déploiement du builder de cartes sans exposer de secrets, d’identifiants d’infrastructure ou de valeurs temporaires de test.

## Principe

La production de cartes reste un calcul **batch et éphémère** derrière une frontière de job pilotée par le backend FireViewer.

```text
administration FireViewer
        ↓
backend / état du job
        ↓
worker de production spatiale
        ↓
dossier spatial scellé
        ├── preuves de validation
        └── viewer public complet
```

Le navigateur ne reçoit jamais les identifiants du provider, les jetons Hugging Face ni les secrets de callback.

## Chemins actuellement conservés

Deux générations de worker coexistent volontairement.

### Chemin stable

Le worker Lightning historique reste disponible comme fallback tant que la nouvelle voie n’a pas passé sa validation représentative.

Fichiers principaux :

- `deploy/Dockerfile.lightning-map-production`;
- `blender/lightning_map_production.py`.

### Voie de comparaison / factual-v2

Une voie parallèle est maintenant disponible pour comparer le comportement historique avec un profil de placement plus strict.

Fichiers principaux :

- `deploy/Dockerfile.lightning-map-production-compare`;
- `deploy/Dockerfile.runpod-map-production-v2`;
- `blender/map_validation_job.py`;
- `blender/map_validation_folder.py`;
- `blender/compare_validation_folders.py`;
- `blender/mns_mnt_placement_inventory_v2.py`;
- `blender/produce_simple_measured_tile_v2.py`.

Cette voie est **implémentée mais pas encore promue comme chemin de production canonique**. La décision dépend d’une comparaison live contrôlée.

## Validation bornée

La première comparaison est volontairement limitée à **exactement 9 tuiles de 500 m**.

Le helper `blender/plan_9_tile_validation.py` fabrique une requête qui résout vers une grille 3 × 3. La même requête doit être utilisée sur les deux providers afin que la comparaison porte sur les mêmes tuiles.

Le job refuse de démarrer la campagne de validation si le plan ne contient pas exactement 9 tuiles.

## Publication folder-native

Les nouveaux chemins de validation **ne produisent pas d’archive ZIP**.

Les preuves de comparaison sont publiées comme fichiers ordinaires dans le dataset public FireViewer :

```text
validation/<zone_id>/<build_id>/<provider>/
```

Elles contiennent uniquement les éléments nécessaires à la comparaison : plan, reçu de zone, inventaires de placement, reçus de source et reçu du viewer lorsqu’il existe.

Ce dossier de validation n’est pas une map supplémentaire et ne doit jamais être présenté comme telle.

## Viewer public

Le viewer navigateur est publié séparément sous :

```text
maps/<zone_id>/<build_id>/runtime/
  viewer.glb
  viewer-scene.v1.json
```

Le viewer est la **représentation complète de la map**, pas une variante simplifiée.

Le job échoue si les invariants suivants ne sont pas respectés :

- couverture mesh complète ;
- politique `fail_closed_exact_visual_scene` ;
- mêmes quantités logiques de bâtiments, arbres et context assets que le build canonique ;
- aucun placeholder ;
- SHA-256 et taille cohérents avec le GLB exporté.

L’optimisation par instancing, partage de meshes/textures et organisation du GLB est permise tant qu’elle ne retire aucun objet logique et ne modifie pas son placement factuel.

## Profil factual-v2

Le profil v2 corrige plusieurs ambiguïtés du chemin historique sans modifier le fallback stable.

### Bâtiments

- footprint BD TOPO comme autorité XY pour les bâtiments instanciés ;
- MNT comme autorité de sol ;
- MNS−MNT comme autorité de hauteur ;
- aucun bâtiment issu uniquement d’une morphologie HAG sans footprint confirmé ;
- représentation discrète par assets USD réels et revus ;
- primitives de remplacement interdites.

### Arbres

Pour garder la première comparaison contrôlée, le nombre/statut des candidats reste basé sur le détecteur historique 1 m. La position et la hauteur peuvent toutefois être raffinées depuis le MNT/MNS natif 0,5 m à l’intérieur de la même cellule de pic.

Ce nombre reste une estimation de couronnes individuelles et non un comptage certifié de troncs.

### Context assets

Une route, une voie ferrée ou une géométrie hydrographique ne crée plus automatiquement un objet d’équipement générique. Les objets ponctuels doivent être explicitement justifiés et utiliser un asset réel du catalogue.

Les géométries continues de route, rail ou hydro peuvent rester construites à partir des données géographiques source : elles ne sont pas assimilées à des équipements ponctuels.

## Données et publication

Le dataset public de référence pour les productions spatiales est :

`fireviewer/simple-measured-scenes-v1`

Le backend conserve l’identité immuable des viewers publiés (repository, révision, chemin, hash, taille et reçu de complétude). La simple présence d’un `viewer.glb` sur Hugging Face ne le rend pas automatiquement actif sur un incident : le rattachement/publication reste une action versionnée du backend.

## Ce qui n’est pas encore validé

À ce stade, il ne faut pas présenter comme acquis :

- que RunPod remplace Lightning en production ;
- qu’un GPU accélère significativement ce pipeline ;
- que factual-v2 améliore la précision terrain sur toutes les zones ;
- que le nouveau chemin a déjà passé une campagne live représentative ;
- que la publication scientifique/source complète du futur provider final est qualifiée.

Ces points restent des gates de validation, pas des caractéristiques marketing.

## Après la comparaison

Si factual-v2 est retenu, les travaux suivants restent distincts : callbacks backend provider-neutral, publication complète du dossier scientifique/source, récupération/annulation, métriques coût/runtime, validation du matching dimensionnel des assets et complétion des géométries continues route/rail/hydro.

Le statut de ces éléments doit être reflété dans la documentation canonique `fireviewer/Fireviewer_doc` sans promouvoir un niveau de maturité supérieur aux preuves réellement obtenues.
