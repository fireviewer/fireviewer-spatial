# Déploiement RunPod Serverless de la production simple

La production de cartes est un job RunPod Serverless asynchrone. L'endpoint
Flex garde `workersMin=0` et `workersMax=1` : aucun worker cartographique ne
reste actif entre deux demandes et aucune requête HTTP n'attend la fin du job.
À l'intérieur d'un job, le moteur traite au maximum quatre tuiles en parallèle.
Les 294 assets immuables sont validés une seule fois au démarrage du processus
worker. Les tuiles ne les recopient et ne les re-hashent plus : leur bundle
partagé est un index de liens vers les assets embarqués. Les quatre tuiles
simultanées utilisent ainsi les neuf vCPU sans multiplier les gros fichiers sur
le volume réseau.

L'administration appelle uniquement le backend FireViewer :

```text
POST /api/v1/admin/map-jobs
GET  /api/v1/admin/map-jobs/{job_id}
GET  /api/v1/admin/map-jobs/{job_id}/download
```

Le backend transmet la demande à la file native RunPod (`/run`), puis lit son
état réel (`/status/{id}`). Le worker produit les tuiles, `zone.usda`,
`zone.blend` et le ZIP autonome, puis redescend à zéro. Le résultat est publié
dans la dataset Hugging Face privée configurée. Le navigateur ne reçoit ni le
jeton RunPod ni le jeton Hugging Face.

## Déploiement

Le module déployable est
[`runpod_map_production.py`](../blender/runpod_map_production.py), embarqué par
[`Dockerfile.runpod-map-production`](../deploy/Dockerfile.runpod-map-production).
L'image runtime contient Blender 4.5.3, OpenUSD et les assets validés. Elle ne
télécharge aucune donnée géographique avant le démarrage d'un job.

Ressources privées requises :

- endpoint RunPod queue-based Flex avec zéro worker minimum et un maximum ;
- volume réseau monté sous `/runpod-volume` pour les checkpoints de tuiles et
  les petits reçus reprenables ;
- disque éphémère local `/tmp/fireviewer-map-production` pour assembler
  `zone.blend`, matérialiser le ZIP autonome et le compresser ;
- secret runtime `HF_TOKEN` et variable `FIREVIEWER_HF_DATASET_ID` ;
- image immuable `pilot-v1-20260814-r22-runpod`.

Le backend Vercel reçoit exclusivement :

```text
FV_MAP_PRODUCTION_PROVIDER=runpod
FV_MAP_RUNPOD_ENDPOINT_ID=<endpoint queue RunPod>
FV_MAP_RUNPOD_API_KEY=<secret serveur>
FV_MAP_HF_TOKEN=<secret serveur>
```

## Contrat de sortie

La production active rend zéro capture et le contrat RunPod expose toujours
`captures: []`. Le ZIP contient au minimum :

- `zone.usda`, scène OpenUSD unifiée ;
- `zone.blend`, scène Blender autonome avec textures emballées ;
- `packages/<tile>/`, chaque terrain de 500 m ;
- `shared/prototype-bundles/v1-<sha256>/`, le lot causal actif des assets
  réellement utilisés, chacun incorporé une seule fois comme fichier normal
  dans le ZIP ;
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
`fireviewer/simple-measured-scenes-v1` uniquement le ZIP final,
`zone.done.json` et `dataset-entry.json`. Le staging local est ensuite supprimé.
La réponse admin expose un lien temporaire vers le ZIP.
La mise en ligne publique sur une fiche incident reste une décision explicite
de l'administrateur après import et contrôle ; le job ne publie jamais seul une
carte au public.

Les périmètres observés restent un flux séparé. Ils peuvent référencer le ZIP de
carte et produire leurs GLB de timeline sans reconstruire la carte.
