# Déploiement Lightning Batch Job de la production simple

La production de cartes est un Batch Job Lightning asynchrone. Elle ne crée ni
Deployment, ni replica, ni worker permanent. Le backend lance une machine CPU
8 cœurs / 32 Go uniquement pour la durée du calcul et l'arrête explicitement
sur demande de l'administrateur. Le premier contrôle réel utilise une machine
non interruptible.

À l'intérieur du job, le moteur traite au maximum quatre tuiles de 500 m en
parallèle. Les 294 assets immuables sont présents dans l'image et validés une
fois au démarrage. Les tuiles utilisent un bundle partagé versionné : elles ne
recopient et ne re-hashent pas la bibliothèque entière.

L'administration appelle uniquement le backend FireViewer :

```text
POST /api/v1/admin/map-jobs
GET  /api/v1/admin/map-jobs/{job_id}
POST /api/v1/admin/map-jobs/{job_id}/cancel
GET  /api/v1/admin/map-jobs/{job_id}/captures
GET  /api/v1/admin/map-jobs/{job_id}/download
```

Le backend crée un nom de job déterministe à partir de la requête et de la clé
d'idempotence, puis lance le Batch Job via `lightning-sdk`. La requête, les
petits événements de progression et le résultat final sont stockés dans le
Blob privé déjà utilisé par FireViewer. Un jeton HMAC propre au job autorise
uniquement ses callbacks. Le navigateur ne reçoit ni les identifiants
Lightning, ni le jeton Hugging Face, ni le jeton de callback.

## Image et stockage

Le module déployable est
[`lightning_map_production.py`](../blender/lightning_map_production.py), embarqué
par
[`Dockerfile.lightning-map-production`](../deploy/Dockerfile.lightning-map-production).
L'image runtime contient Blender 4.5.3, OpenUSD, le générateur et les assets
validés. Elle ne télécharge aucune donnée géographique pendant sa construction.

Le job utilise :

- l'image immuable `pilot-v1-20260814-r24-lightning` ;
- `Machine.CPU_X_8`, non interruptible, avec une durée maximale bornée ;
- une Data Connection Lightning montée sous `/lightning-work` pour les
  checkpoints reprenables des tuiles ;
- le disque local rapide `/lightning-scratch/fireviewer-map-production` pour
  l'assemblage, `zone.blend` et le ZIP ;
- le jeton `HF_TOKEN` injecté uniquement au job ;
- quatre workers de tuiles et huit workers légers de prototypes.

Le backend Vercel reçoit exclusivement des variables serveur :

```text
FV_MAP_PRODUCTION_PROVIDER=lightning
FV_MAP_LIGHTNING_USER_ID=<identifiant programme Lightning>
FV_MAP_LIGHTNING_API_KEY=<clé programme Lightning>
FV_MAP_LIGHTNING_TEAMSPACE=<teamspace>
FV_MAP_LIGHTNING_IMAGE=charlibillabert/fireviewer-simple-production-ui:pilot-v1-20260814-r24-lightning
FV_MAP_LIGHTNING_CHECKPOINT_CONNECTION=<data connection>
FV_MAP_LIGHTNING_MAX_RUNTIME_SECONDS=86400
FV_MAP_CALLBACK_BASE_URL=https://fireviewer-api.vercel.app
FV_MAP_CALLBACK_SIGNING_SECRET=<secret serveur aléatoire>
FV_MAP_HF_DATASET_ID=fireviewer/simple-measured-scenes-v1
FV_MAP_HF_TOKEN=<secret serveur>
```

## Contrat de sortie

La production active rend zéro capture et expose toujours `captures: []`. Le
ZIP contient au minimum :

- `zone.usda`, scène OpenUSD unifiée ;
- `zone.blend`, scène Blender autonome avec textures emballées ;
- `packages/<tile>/`, chaque terrain de 500 m ;
- `shared/prototype-bundles/v1-<sha256>/`, les assets réellement utilisés,
  incorporés une seule fois dans le ZIP ;
- `provenance/<tile>/`, les reçus source compacts ;
- `zone-context.json`, `zone-plan.json` et `zone.done.json`.

Les MNT, MNS et orthophotos bruts sont supprimés après validation des tuiles et
n'entrent jamais dans l'archive. Les contrats actifs sont
`fireviewer.simple-measured-map-package.v2` et
`fireviewer.simple-measured-map-upload-contract.v2`.

## Publication

Le job publie dans la dataset privée `fireviewer/simple-measured-scenes-v1`
uniquement le ZIP final et ses petits reçus. Le backend délivre ensuite une URL
Hugging Face signée à l'administrateur authentifié. La publication sur une
fiche incident reste une action admin séparée.

Les périmètres observés restent un flux distinct. Ils peuvent référencer le ZIP
de carte et produire leur timeline sans reconstruire la carte.
