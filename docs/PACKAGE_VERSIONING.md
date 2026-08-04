# Versionnement des packages spatiaux

## Principe

Un package spatial est immuable après publication. Toute modification crée une nouvelle révision.

## Contenu du manifeste

- identifiant de zone ;
- révision ;
- emprise ;
- CRS ;
- datum vertical ;
- MNT/MNS ;
- orthophoto ;
- GLB ou assets ;
- catalogue de rendus ;
- tailles ;
- empreintes ;
- licences ;
- date des sources ;
- reçu de validation.

## Dépendances

Les artefacts de recalage enregistrent la révision exacte du package et de la banque de rendus.

## Compatibilité

Une nouvelle révision ne remplace pas silencieusement les références des anciens runs.

## Publication

La validation automatisée ne remplace pas la revue Unity ou la validation humaine requise par le contrat du package.

## Retrait

Le retrait d’un package public masque son usage futur sans effacer les liens d’audit nécessaires aux anciens runs.
