# Contrats du pipeline spatial actif

Le pipeline actif produit des cartes mesurées autonomes et des packages de
périmètres observés. Ses contrats applicatifs sont maintenus au plus près des
modules sous `blender/` et `omniverse/` ; les schémas partagés restent dans ce
dossier et sous `contracts/spatial/v1`.

## Assets

- `asset-library.v1.schema.json` : catalogue mesuré des assets approuvés ;
- `reference-asset-library.v1.schema.json` : catalogue complet, sélection du
  meilleur USD disponible et donneur réel déterministe lorsqu'une identité est
  encore absente ;
- les USD et textures sont rehashés lors du build et du démarrage ; aucun cube
  ou placeholder noir n'est admis dans une production livrée.

## Carte mesurée

Les contrats JSON adjacents aux modules verrouillent :

- la grille FVTG fixe et ses trois LOD ;
- la texture orthophoto bakée ;
- l'inventaire MNS−MNT et les confirmations contextuelles ;
- les placements GPS explicites ;
- la scène OpenUSD autonome ;
- la validation Blender, l'acceptation humaine et les vingt captures ;
- l'inventaire byte-for-byte du ZIP.

Les rasters MNT, MNS et orthophoto sont des sources temporaires. Seuls leurs
petits reçus et hashes restent après validation de la tuile.

## Périmètres et consommateurs

`contracts/spatial/v1/scene-consumer-input.schema.json` lie une simulation, un
dataset ou un replay à une carte immuable et, si nécessaire, à sa timeline de
périmètres. Les GLB servent au contrôle web et ne remplacent jamais la timeline
JSON ni l'USD.

## Compatibilité

Les anciens schémas terrainctl, FVTQ, PBR et streaming ne sont pas des contrats
d'écriture du pipeline actif. Ils ne doivent être invoqués que pour l'audit
d'archives historiques jusqu'à leur retrait approuvé.
