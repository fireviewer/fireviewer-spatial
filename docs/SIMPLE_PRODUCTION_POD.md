# Déploiement Modal CPU de la production simple

La production de cartes est un job Modal CPU Serverless asynchrone. Aucun
conteneur cartographique ne reste actif entre deux demandes et aucune requête
HTTP n'attend la fin de Blender.

L'administration appelle uniquement le backend FireViewer :

```text
POST /api/v1/admin/map-jobs
GET  /api/v1/admin/map-jobs/{job_id}
GET  /api/v1/admin/map-jobs/{job_id}/download
```

Le backend transmet la demande au petit endpoint Modal. La fonction lourde est
créée à la demande avec huit CPU, produit les tuiles, `zone.usda`, `zone.blend`
et le ZIP autonome, puis redescend à zéro conteneur. Le résultat est publié dans
la dataset Hugging Face privée configurée. Le navigateur ne reçoit ni le jeton
Modal ni le jeton Hugging Face.

## Déploiement

Le module déployable est
[`modal_map_production.py`](../blender/modal_map_production.py). Il réutilise
l'image runtime validée contenant Blender 4.5.3, OpenUSD et les assets, puis
superpose le code Python courant au build Modal. Il ne télécharge aucune donnée
géographique avant le démarrage d'un job.

Ressources privées requises :

- volume `fireviewer-map-production-work-v1` ;
- dictionnaire `fireviewer-map-production-jobs-v1` ;
- secret `fireviewer-map-production-secrets` contenant `HF_TOKEN` et
  `FIREVIEWER_API_TOKEN`.

Commande :

```powershell
python -m modal deploy fireviewer-spatial/blender/modal_map_production.py
```

Le backend Vercel reçoit exclusivement :

```text
FV_MAP_PRODUCTION_PROVIDER=modal
FV_MAP_MODAL_BASE_URL=https://<workspace>--fireviewer-map-production-api.modal.run
FV_MAP_MODAL_API_TOKEN=<secret serveur>
```

## Contrat de sortie

La production active rend zéro capture. Le ZIP contient au minimum :

- `zone.usda`, scène OpenUSD unifiée ;
- `zone.blend`, scène Blender autonome avec textures emballées ;
- `packages/<tile>/`, chaque terrain de 500 m ;
- `shared/prototypes/`, les assets réellement utilisés ;
- `provenance/<tile>/`, les reçus source compacts ;
- `zone-context.json`, `zone-plan.json` et `zone.done.json`.

Les MNT, MNS et orthophotos bruts sont supprimés après validation des tuiles et
n'entrent jamais dans l'archive. Les contrats actifs sont
`fireviewer.simple-measured-map-package.v2` et
`fireviewer.simple-measured-map-upload-contract.v2`. Ils lient directement
`zone.blend` et interdisent une galerie héritée dans un nouveau package.

Les anciens packages v1 avec vingt captures restent lisibles pour compatibilité,
mais ne sont plus produits.

## Publication

Le worker publie atomiquement dans la dataset privée
`fireviewer/simple-measured-scenes-v1` : le ZIP, `zone.done.json` et
`dataset-entry.json`. La réponse admin expose un lien temporaire vers le ZIP.
La mise en ligne publique sur une fiche incident reste une décision explicite
de l'administrateur après import et contrôle ; le job ne publie jamais seul une
carte au public.

Les périmètres observés restent un flux séparé. Ils peuvent référencer le ZIP de
carte et produire leurs GLB de timeline sans reconstruire la carte.
