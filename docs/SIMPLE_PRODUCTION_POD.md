# Déploiement Lightning Batch Job de la production simple

La production de cartes est un Batch Job Lightning asynchrone. Elle ne crée ni
Deployment, ni replica, ni worker permanent. Le backend lance une machine CPU
8 cœurs / 32 Go uniquement pour la durée du calcul et l'arrête explicitement
sur demande de l'administrateur. Le premier contrôle réel utilise une machine
non interruptible.

À l'intérieur du job, le moteur maintient au maximum douze acquisitions de
sources et six compilations de tuiles de 500 m en parallèle. Les 294 assets
immuables sont présents dans l'image et validés une fois au démarrage. Les
tuiles utilisent un bundle partagé versionné : elles ne recopient et ne
re-hashent pas la bibliothèque entière.

Les acquisitions sont regroupées par blocs de 4 × 4 tuiles. Une zone de
25 × 25 tuiles utilise ainsi 49 blocs et 147 requêtes WMS au lieu de 1 875.
Chaque bloc est découpé localement en tuiles logiques de 500 m, avec les mêmes
résolutions et halos qu'avant.

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

- l'image candidate `pilot-v1-20260815-r34-lightning`, qui embarque
  `fireviewer.mns-mnt-placement-algorithm.v2` ;
- `Machine.CPU_X_8`, non interruptible, avec une durée maximale bornée ;
- `/lightning-work/fireviewer-map-production` pour les checkpoints compressés
  de reprise et les petits reçus ;
- le SSD éphémère `/lightning-scratch/fireviewer-map-production` pour les
  métatuiles sources, les packages complets, l'assemblage, `zone.blend` et le
  ZIP ;
- le jeton `HF_TOKEN` injecté uniquement au job ;
- douze workers d'acquisition de sources, six workers de compilation de
  tuiles et huit workers légers de prototypes.

Les gros packages ne sont jamais assemblés sur un montage réseau. Chaque tuile
est validée, compressée et hashée une fois sur le SSD. Sa publication vers le
volume persistant est ensuite sérialisée, atomique et bornée à 180 secondes.
Les téléchargements raster et l'attente d'une métatuile sont également bornés ;
une tuile sans activité pendant 8 minutes fait échouer le job au lieu de le
laisser facturer indéfiniment. Une reprise sur le même volume re-hashe et
restaure uniquement les checkpoints déjà publiés.
Sans volume persistant configuré, les checkpoints restent limités à la durée du
Batch Job ; la requête et la progression restent néanmoins persistées côté
backend. Seuls le ZIP final et ses petits reçus sont publiés sur Hugging Face.

Le backend Vercel reçoit exclusivement des variables serveur :

```text
FV_MAP_PRODUCTION_PROVIDER=lightning
FV_MAP_LIGHTNING_USER_ID=<identifiant programme Lightning>
FV_MAP_LIGHTNING_API_KEY=<clé programme Lightning>
FV_MAP_LIGHTNING_TEAMSPACE=<teamspace>
FV_MAP_LIGHTNING_IMAGE=charlibillabert/fireviewer-simple-production-ui:pilot-v1-20260815-r34-lightning
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
