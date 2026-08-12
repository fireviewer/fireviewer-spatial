# Contrats des consommateurs spatiaux

Ce dossier relie une simulation, un dataset ou un replay aux packages immuables
produits par le pipeline actif. Il ne contient aucune carte ni observation réelle.

| Artefact | Rôle |
| --- | --- |
| `scene-consumer-input.schema.json` | liaison immuable d'une simulation, d'un dataset ou d'un replay vers une carte et, si nécessaire, sa timeline observée |
| `fixtures/scene-consumer-input.json` | exemple synthétique sans donnée de production |

## Conventions verrouillées

- Un consommateur ne reconstruit ni le terrain ni les périmètres. Il référence
  leurs package IDs, révisions, build IDs, contrats et archives hashés.
- Une carte `technical_unpublished` peut alimenter un travail interne ; sa
  publication publique reste une décision distincte.
- La timeline observée conserve `between_observations=undefined` et
  `prediction=none`. Les GLB de contrôle ne remplacent jamais son JSON.

Le [contrat caméra et CRS](../../../docs/CAMERA_AND_CRS_CONTRACT.md) constitue
la référence normative. La fixture reste synthétique et ne localise aucun événement.
