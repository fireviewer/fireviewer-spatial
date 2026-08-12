# Contrats de composition Omniverse V2

Ces contrats décrivent uniquement la composition aval des assets Omniverse.
Ils ne produisent pas de carte. Toute scène ou simulation doit désormais
référencer une carte FVTG/OpenUSD immuable et, si nécessaire, une timeline de
périmètres via `fireviewer.scene-consumer-input.v1`.

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
