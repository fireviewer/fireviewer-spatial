# Versionnement des cartes et périmètres

## Principe

Une carte ou une timeline de périmètres est immuable après scellement. Toute
modification des sources, de l'emprise, du code, des assets ou des observations
crée un nouveau package, une nouvelle révision et un nouveau `build_id`.

## Package carte

Le package `fireviewer.simple-measured-map-package.v1` inventorie notamment :

- la demande GPS, l'emprise alignée et le plan de tuiles de 500 m ;
- `zone.usda`, `zone.blend` et `zone.done.json` ;
- chaque relief FVTG, texture de sol bakée et scène de tuile ;
- les assets USD et textures réellement utilisés ;
- les placements MNS−MNT et les placements GPS explicites ;
- les petits reçus de provenance conservés après suppression des rasters ;
- quatre vues générales et seize vues de détail ;
- le SHA-256 et la taille de chaque fichier portable.

Le ZIP original reste autonome après extraction. Il ne contient ni MNT, ni MNS,
ni orthophoto brute, ni cache, ni chemin absolu de la machine de production.

## Package de périmètres

Le package `fireviewer.observed-perimeter-package.v1` verrouille :

- le package, la révision, la zone et le build exact de la carte de base ;
- les observations normalisées ;
- `geographic-perimeters.usda` ;
- `fire-progression-timeline.json` ;
- le manifeste et les vues GLB dérivées de contrôle.

Une nouvelle observation produit un package supplémentaire. Elle ne modifie et
ne reconstruit jamais la carte. Entre observations, la progression reste
`undefined` et aucune prédiction n'est ajoutée.

## Compatibilité et consommateurs

Les simulations, datasets et replays utilisent
`fireviewer.scene-consumer-input.v1`. Le contrat référence les identités et
hashes immuables ; le consommateur ne peut recalculer ni terrain ni périmètres.

Les anciens packages adaptatifs, atlas de sol et catalogues Unity ne sont plus
des sorties admissibles et leur code a été retiré. Une archive externe de ces
formats ne peut être importée comme carte active.

## Publication et retrait

L'import technique, la validation humaine et la publication sont trois gates
distincts. Le retrait d'un package public empêche son nouvel usage public sans
effacer sa provenance ni les liens d'audit déjà produits.
