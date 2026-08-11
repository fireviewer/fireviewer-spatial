# Contrats de composition Omniverse V2

Ces contrats remplacent les contrats de scènes et de simulation V1. Ils ne
relancent aucune scène : leur état courant est
`blocked_pending_usd_assets` jusqu'à réception des 295 assets USD refaits.

## Frontière définitive

- un catalogue unique de exactement 295 assets USD, triés par identifiant
  stable et liés par SHA-256 ;
- aucune primitive, aucun placeholder et aucun fallback géométrique simplifié ;
- quatre scènes de base et cinq compositions par base, soit vingt scènes ;
- conservation des positions géospatiales, identifiants et reliefs validés ;
- variété déterministe guidée par les contextes compatibles de chaque asset ;
- couverture des 295 assets au moins une fois dans le portfolio ;
- scènes, simulation, reproduction et carte restent des couches distinctes ;
- aucune publication ni génération de pack tant que les gates USD, Kit et
  visuels ne sont pas passés.

Le catalogue partiel peut être contrôlé, mais l'algorithme refuse de produire
un plan avant que les 295 entrées soient `accepted`, hashées et ouvertes par
Kit. Le plan répartit ensuite chaque asset comme minimum obligatoire dans une
scène compatible et fournit, par scène, un ordre complet déterministe des
assets compatibles. La géométrie et les placements détaillés seront authorés
uniquement après réception du catalogue final.

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
