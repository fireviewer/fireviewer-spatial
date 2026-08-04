# Spécification de la banque de rendus

## Portée

La banque de rendus est locale à une zone ou un package d’incident. Elle n’indexe pas automatiquement tout le territoire.

## Sources

- MNT ;
- MNS/DSM ;
- LiDAR ;
- orthophoto ;
- scène ou package 3D ;
- vecteurs utiles ;
- profils de caméra.

## Artefacts par rendu

- RGB ;
- profondeur ;
- masque de ciel ;
- masque de surface valide ;
- identifiants de surfaces ;
- pose ;
- intrinsics ;
- FOV ;
- révision du package ;
- empreinte.

## Sélection des poses

Les poses sont priorisées autour :

- routes ;
- villages ;
- hameaux ;
- points hauts ;
- interfaces habitat-forêt ;
- chemins ;
- zones accessibles ;
- positions déclarées lorsqu’elles sont disponibles et autorisées.

La granularité est décidée à partir du rappel du retrieval et du coût de stockage, pas fixée arbitrairement.

## Versionnement

Une banque est régénérée lorsque sa dépendance spatiale change. Les anciennes révisions restent identifiables pour le replay.

## Index

Le pilote peut utiliser un index local ou une recherche exhaustive sur des descripteurs pré-calculés. Une infrastructure distribuée n’est pas une dépendance du MVP.
