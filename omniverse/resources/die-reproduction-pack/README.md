---
pretty_name: FireViewer Die 2026 Omniverse Reproduction Pack
license: other
tags:
  - fireviewer
  - omniverse
  - openusd
  - wildfire
  - simulation
  - geospatial
---

# FireViewer Die 2026 Omniverse Reproduction Pack

Pack autonome et reproductible de simulation historique FireViewer pour Die et
Pontaix. L’archive contient la scène OpenUSD, le terrain, l’orthophoto, les
bâtiments, les routes, la végétation, les périmètres temporels, les caméras, le
scénario, la configuration Flow et les contrats d’exécution nécessaires à une
réouverture dans le runtime Omniverse/Kit verrouillé par le pack.

## Archive publiée

- fichier : `fireviewer-die-2026-reproduction-download-r1.zip`
- taille : `217296677` octets (`207.23 MiB`)
- SHA-256 : `1990504c41ce3da672ce4a25f8d345b67ad751c318bf769008d175f541040db0`
- scène d’entrée : `fireviewer-die-2026-reproduction-download-r1/dataset.usda`
- contrat : `fireviewer.omniverse-reproducible-download-bundle-contract.v1`
- statut de release : `active_pilot_capture_authorized`

Le fichier `download-bundle-contract.json` décrit les dépendances, les hashes,
le runtime et les garanties de portabilité. `bundle-acceptance.json` contient le
reçu d’acceptation correspondant.

## Portée

Ce pack sert à la reproduction technique et à la simulation documentée. Il ne
constitue ni une observation active, ni une prévision, ni une consigne de
sécurité. Il n’autorise pas automatiquement la production ou la publication
d’un dataset d’entraînement.

Les données et ressources tierces conservent leurs conditions d’utilisation et
leurs attributions, enregistrées dans les manifestes et fichiers de provenance
de l’archive.

## Vérification après téléchargement

```powershell
Get-FileHash -Algorithm SHA256 .\fireviewer-die-2026-reproduction-download-r1.zip
```

La valeur obtenue doit correspondre exactement au SHA-256 publié ci-dessus.
