# Versionnement des packages spatiaux

## Principe

Un package spatial est immuable après publication. Toute modification crée une nouvelle révision.

## Contenu du manifeste

- identifiant de zone ;
- révision ;
- emprise ;
- CRS ;
- datum vertical ;
- contrat et révision des sources MNT/MNS 2 m ;
- FVTQ LOD0, LOD1 et LOD2 ;
- réservation HAG compacte ;
- quatre cartes de sol 500 × 500 : IDs, poids, confiance et orientation ;
- manifeste hash-locké `surface-correspondence.json` ;
- `ground-material-contract.v2` et identité des quatre atlas partagés ;
- manifeste `tile-package.v3.json` et reçu `tile.done.v3.json` ;
- payloads OpenUSD dérivés ;
- catalogue de rendus ;
- tailles ;
- empreintes ;
- licences ;
- date des sources ;
- reçu de validation.

Une orthophoto, une image source rapprochée ou un ancien terrain n'est pas une
dépendance admissible d'un package terrain adaptatif. Le RGB orthophoto
temporaire n'est supprimé qu'après validation de `tile.done.v3.json`, puis ne
subsiste ni dans le package ni dans le runtime.

## Dépendances

Les artefacts de recalage enregistrent la révision exacte du package et de la banque de rendus.

## Compatibilité

Une nouvelle révision ne remplace pas silencieusement les références des anciens runs.

Les packages terrain v2 restent lisibles uniquement pour le replay et l'audit.
Toute nouvelle production écrit exclusivement `tile-package.v3.json`,
`tile.done.v3.json` et `ground-material-contract.v2`.

## Publication

La qualification terrain exige le reçu Blender headless texturé, la
composition `world_xy`/`world_triplanar` sur le LOD0 FVTQ et son AOV LOD0. La
fixture synthétique passe ce gate technique en 512 px avec zéro pixel
LOD/couverture invalide et 44 544 pixels triplanaires. Le statut de production
reste en attente de la bibliothèque PBR réelle et de sa revue humaine.

Le `UsdPreviewSurface` magenta du package est diagnostique. Le shader runtime
reste `pending_dedicated_mdl_validation` et `usd_runtime_gate=false` tant qu'un
MDL dédié n'a pas été validé. Ce gate USD séparé ne bloque pas l'acceptation du
terrain dans Blender ; il bloque la qualification de rendu texturé
OpenUSD/Omniverse et ne remplace pas la revue humaine finale.

## Retrait

Le retrait d’un package public masque son usage futur sans effacer les liens d’audit nécessaires aux anciens runs.
