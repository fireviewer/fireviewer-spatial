# Générateur mono-zone adaptatif

[`prepare_adaptive_zone_specs.py`](./prepare_adaptive_zone_specs.py) prépare un
unique `zone-spec.v1` pour [`terrainctl.py`](../terrainctl.py). Il ne contacte
aucun service et ne télécharge aucune donnée.
Chaque cœur de tuile de 500 m produit exactement une paire de requêtes HTTPS
MNT/MNS à 2 m ; le halo de 10 m est ajouté plus tard par l’acquisition.

L’ancien générateur `prepare_incident_terrains.py` fondé sur `global-05m` et la
représentation uniforme à 0,5 m sont dépréciés pour toute nouvelle production.
Le
[`adaptive_terrain_zone_catalog.v1.json`](./adaptive_terrain_zone_catalog.v1.json)
réutilise uniquement les six identités et emprises carrées EPSG:2154 déjà
approuvées. Il ne réutilise aucun ancien terrain, sol, asset ou placement,
notamment pour Die. Sa signature canonique, ses comptes dérivés et sa
provenance sont validés à chaque chargement.

## Entrée compacte

Le JSON d’entrée `fireviewer.adaptive-zone-spec-input.v1` décrit une seule
zone : identifiant, révision, carré Lambert-93, endpoints et couches MNT/MNS,
révision fournisseur, huit chemins de dépendances réelles sur D:, racines de
travail/export séparées sur D: et estimation du pic disque. Les scores pilote
et la tuile de régression sont optionnels.

Les hashes des huit dépendances sont calculés localement. Un reçu visuel atlas
encore `pending` peut donc être verrouillé dans le plan ; il ne devient
bloquant qu’au `preflight`, qui exige `accepted_blender_visual`.
Le contrat produit est vérifié contre l’implémentation active et le
[`zone-spec.schema.json`](../contracts/terrain/v1/zone-spec.schema.json) avant
son écriture atomique.

```powershell
python blender\prepare_adaptive_zone_specs.py `
  --input D:\fireviewer-work\zones\FR-30-00001\adaptive-zone-input.v1.json `
  --output D:\fireviewer-work\zones\FR-30-00001\zone-spec.v1.json

python terrainctl.py `
  --zone D:\fireviewer-work\zones\FR-30-00001\zone-spec.v1.json `
  --phase plan `
  --dry-run
```

Il n’existe volontairement ni option `--all`, ni entrée multi-zone. L’ordre
verrouillé est Lédenon, Oupia, Die, Taradeau, Fontainebleau, Trévillach ; la
zone suivante reste interdite tant que la précédente n’est pas acceptée puis
nettoyée.
