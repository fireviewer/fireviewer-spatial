# Contrats de composition Omniverse V2

Ces contrats remplacent les contrats de scènes et de simulation V1. Leur couche
de composition reste `blocked_pending_usd_assets` jusqu'à réception des 295
assets USD refaits. Ce blocage ne s'applique pas à la production préalable des
six terrains 3D et sols 2D sans orthophoto. Le relief provient du MNT/MNS ; les
sols utilisent en plus des couches de contexte approuvées pour distinguer les
surfaces naturelles, champs, routes, chemins, berges et plateformes
ferroviaires.

Les images rapprochées ne sont jamais utilisées comme tuiles territoriales :
21 sources de micro-détail hors ligne sont empaquetées dans quatre atlases PBR
runtime. Soixante-douze profils procéduraux couvrent les variations naturelles,
brûlées, agricoles, routières, chemins, cours d’eau, plateformes ferroviaires
et parois rocheuses. Leurs variations méso et macro sont déterministes et ne
créent aucune texture importée supplémentaire.

## Frontière définitive

- un catalogue unique de exactement 295 assets USD, triés par identifiant
  stable et liés par SHA-256 ;
- aucune primitive, aucun placeholder et aucun fallback géométrique simplifié ;
- quatre scènes de base et cinq compositions par base, soit vingt scènes ;
- reconstruction des positions, reliefs et couches géographiques selon le
  nouveau contrat, sans réutiliser les placements Die dépréciés ;
- variété déterministe guidée par les contextes compatibles de chaque asset ;
- couverture des 295 assets au moins une fois dans le portfolio ;
- scènes, simulation, reproduction et carte restent des couches distinctes ;
- aucune publication ni génération de pack tant que les gates USD, Kit et
  visuels ne sont pas passés.

Le catalogue partiel peut être contrôlé, mais l'algorithme refuse de produire
un plan avant que les 295 entrées soient `accepted`, hashées et ouvertes par
Kit. Le plan répartit ensuite chaque asset comme minimum obligatoire dans une
scène compatible et fournit, par scène, un ordre complet déterministe des
assets compatibles. Les géométries et placements d'assets seront authorés
uniquement après réception du catalogue final. Les terrains et sols 2D sont
produits avant cette réception et deviennent leurs références géographiques
immuables.

## Fichiers

- `asset-catalog-contract.schema.json` : contrat d'entrée des 295 USD ;
- `scene-composition-contract.schema.json` : règles immuables du portfolio ;
- `examples/*.pending.json` : état d'attente actuel, sans faux hash ni asset ;
- `../../scene_composition.py` : validation et planification déterministe ;
- `validate_contracts.py` : validation JSON Schema et invariants métier.

## Validation locale

```powershell
python omniverse/contracts/v2/validate_contracts.py
python -m pytest -q omniverse/test_scene_composition_v2.py
```

Ces validations ne prouvent ni l'ouverture des futurs USD, ni le rendu RTX,
ni la qualité visuelle. Ces preuves ne pourront être produites qu'après
livraison des assets refaits.
