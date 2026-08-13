# API de production simple FireViewer

Le conteneur actif est un worker RunPod Serverless headless, sans Gradio,
frontend ni base de données. L'écran React vit séparément dans l'espace authentifié de
`fireviewer-frontend`, à `/admin/production`; son composant est
`src/features/simple-production/SimpleProductionWorkspace.tsx`. Aucune route
publique ne donne accès à la production.

Le premier flux produit une zone complète et demande uniquement :

- la latitude du centre ;
- la longitude du centre ;
- la taille d'un côté du carré en kilomètres.

Il accepte aussi, de façon optionnelle, des placements fixes. Dans l'interface
locale, on indique la latitude et la longitude, puis on choisit l'asset USD exact
dans le catalogue exposé par l'API. L'altitude n'est jamais saisie : elle est
échantillonnée sur le MNT de la tuile propriétaire. L'échelle reste celle du
catalogue.

La même liste peut être importée avec le modèle
`blender/fixed_asset_placement_template.v1.json`. Le JSON fermé est :

```json
{
  "schema": "fireviewer.fixed-asset-placement-request.v1",
  "crs": "EPSG:4326",
  "placements": [
    {
      "placement_id": "church-main",
      "asset_id": "identifiant_exact_du_catalogue",
      "latitude": 43.9,
      "longitude": 4.5,
      "yaw_deg": 0
    }
  ]
}
```

Aucun autre champ n'est accepté. Les identifiants de placement doivent être
uniques, chaque `asset_id` doit exister dans le catalogue embarqué et chaque
point doit rester dans le carré demandé. La projection Lambert-93, l'attribution
à une seule tuile de 500 m et le hash de la liste sont déterministes.

Le second flux accepte un fichier JSON/GeoJSON de périmètres observés et, pour
le contrôle visuel, le ZIP autonome d'une carte FireViewer. Il produit les zones
touchées et actives comme calques USD fixes, ainsi qu'une timeline JSON fondée
sur les heures réelles. Une entrée peut être un instant ou une plage explicite
`time_window.start` / `time_window.end`; ces bornes sont conservées sans être
devinées. Un curseur charge chaque instant ou plage sur le terrain dans un
viewer 3D. Aucune progression n'est interpolée ou prédite entre deux entrées.
Une seule production s'exécute par worker. L'état, les reçus et les résultats
sont des fichiers sous `/runpod-volume/fireviewer-map-production`.

## Contrat HTTP

Le navigateur appelle uniquement les routes authentifiées du backend
`/api/v1/admin/map-production/*` et `/api/v1/admin/map-jobs/*`. Le backend
soumet le travail à l'opération asynchrone RunPod `/run`, interroge `/status`
et ne transmet jamais les secrets RunPod ou Hugging Face au navigateur. Les
contrats de l'Admin et les ZIP produits restent inchangés.

## Construire l'image RunPod

Depuis `fireviewer-repositories` :

```powershell
docker build -f fireviewer-spatial/deploy/Dockerfile.simple-production-base -t fireviewer-map-base:pilot-v1-20260811-r12 .
docker build -f fireviewer-spatial/deploy/Dockerfile.runpod-map-production -t charlibillabert/fireviewer-simple-production-ui:pilot-v1-20260813-r14-runpod .
```

Le build dérive de l'image runtime validée et ne télécharge aucune donnée
géographique ni ne compile de terrain.
Il embarque les dépendances Python, Blender 4.5.3 avec ses bindings OpenUSD, le
générateur complet et le catalogue intégral des assets 3D attendus. Les USD
normalisés validés ont priorité sur les anciennes versions Hunyuan ; les assets
explicitement rejetés par le lot de normalisation ne peuvent pas revenir depuis
une source de priorité inférieure. Les USDZ premium conservent la priorité quand
ils correspondent à la même référence. Une identité sans USD propre
réutilise déterministiquement un véritable USD compatible : catégorie exacte en
priorité, puis famille compatible, similarité sémantique et hash stable. Le reçu
conserve l'identité demandée et celle du donneur. Aucun cube ni point noir n'est
publié ; lorsqu'un USD propre est ajouté sous le même identifiant, il remplace
automatiquement cette substitution sans changer les placements. Les
images de référence, anciennes versions et lots de validation servent uniquement
à construire l'index dans des étapes Docker jetables. L'image finale reçoit
uniquement les USD/textures sélectionnés. L'archive Blender
officielle est vérifiée par SHA-256 avant extraction. Le build puis chaque
démarrage prouvent que `bpy` et `pxr` s'importent réellement et rehashent les
assets avant l'ouverture de l'API. Le pod ne télécharge ensuite que les sources
IGN de la zone demandée.

Le catalogue ne constitue jamais un quota : un même USD réel peut
être instancié autant de fois que des candidats MNS/MNT ou SIG distincts le
justifient. Les compteurs de scène séparent les prototypes uniques des instances
placées. Les catégories sans observation géographique exploitable restent dans
le catalogue, mais ne sont pas dispersées arbitrairement sur le terrain.

## Exécution

Le template RunPod lance directement
`blender/runpod_map_production.py`. Le volume persistant est monté sur
`/runpod-volume` et `FIREVIEWER_TILE_WORKERS` fixe le parallélisme borné par
tuile (8 dans l'image active). Une tuile déjà scellée est revalidée et reprise.

Le worker doit autoriser les requêtes HTTPS sortantes vers la Géoplateforme IGN.
Pour publier les résultats, injecter `HF_TOKEN` comme secret RunPod ; il n'est jamais copié
dans l'image, les reçus ou le ZIP. La cible verrouillée de l'image est la dataset
privée `fireviewer/simple-measured-scenes-v1`.

## Résultat téléchargé

Le bouton de téléchargement apparaît seulement après production. Le ZIP est
autonome après extraction et contient :

- `zone.usda`, point d'entrée de la scène unifiée ;
- `packages/<tile>/`, le pack complet de chaque terrain de 500 m ;
- `shared/prototypes/`, les seuls assets 3D réellement utilisés ;
- `provenance/<tile>/`, les petits reçus source ;
- `zone-context.json`, `zone-plan.json` et `zone.done.json`.

Les MNT, MNS et orthophotos bruts sont supprimés après validation de chaque
tuile et ne sont pas placés dans le ZIP. L'interface ne crée aucune acceptation
humaine automatique : le reçu de zone reste technique.

Si une paire MNS/MNT reste incohérente, la tuile n'arrête plus toute la zone :
elle conserve terrain et orthophoto, utilise le MNT comme MNS de secours et
n'infère aucun objet de hauteur sur cette tuile. Les reçus de tuile et de zone
enregistrent explicitement `degraded_mns_fallback` et le nombre de tuiles
dégradées ; cette information doit rester visible lors de la validation.

Le téléchargement du flux périmètres est un ZIP autonome contenant :

- `geographic-perimeters.usda`, avec un prim fixe par catégorie et instant/plage ;
- `fire-progression-timeline.json`, futur pilote temporel de la simulation ;
- `perimeters.normalized.json`, source WGS84 canonique ;
- `perimeter-layer.manifest.json`, hashes, compteurs et identité de build.
- `preview/perimeter-viewer.manifest.json` et un GLB dérivé hashé par état,
  uniquement pour le contrôle dans un navigateur.

Les temps de simulation conservent l'instant observé ainsi que les secondes de
début et de fin de chaque plage explicite, depuis le premier début réel. Un
instant est représenté avec début = fin = instant observé. La valeur entre deux
entrées reste explicitement indéfinie : le paquet ne fabrique ni vitesse ni
géométrie intermédiaire.

Le viewer local revalide les FVTG du ZIP de carte, reconstruit un LOD léger avec
la texture sol, puis drape séparément chaque observation. Il affiche un GLB dérivé
par date parce que le navigateur ne lit pas directement l'USD. Ces GLB
ne sont jamais des données de simulation : l'USD et la timeline JSON restent
les seules sorties autoritatives.

## Import dans le site et consommateurs aval

Le site n'essaie pas de convertir OpenUSD dans le navigateur. L'administrateur
extrait le ZIP, sélectionne son dossier racine, puis le frontend et le backend
vérifient l'inventaire, chaque SHA-256 et les contrats actifs. La carte est
contrôlée par ses 20 captures liées au package ; `zone.usda` reste téléchargeable
comme scène de référence. La timeline est contrôlée par ses GLB dérivés, tandis
que `fire-progression-timeline.json` et les géométries normalisées restent la
vérité temporelle.

Une simulation, un dataset ou un replay consomme une carte publiée ou non et,
si nécessaire, une timeline déjà produite via
`contracts/spatial/v1/scene-consumer-input.schema.json`. Ce contrat interdit de
reconstruire terrain ou périmètres. L'exécution de la simulation n'appartient
pas à cette API de production.

## Reprise

Un job est identifié par ses coordonnées, sa taille, sa grille Lambert-93 et,
si elle n'est pas vide, la liste canonique hashée des placements fixes.
Une tuile déjà scellée est revalidée et réutilisée sans retéléchargement. Un ZIP
complet et cohérent est renvoyé directement si la même demande est relancée.

## Développement de l'interface Admin

Depuis `fireviewer-frontend` :

```powershell
npm run dev
```

Ouvrir `http://127.0.0.1:5173/admin/production`, activer la session
administrateur locale, puis renseigner l'URL de l'API. La production passe par
le même garde d'authentification que le reste de l'Admin. Le composant affiche
les phases réellement exécutées : contrôle du moteur
embarqué, contexte IGN, téléchargement séparé MNT/MNS/orthophoto avec volumes,
décodage, FVTG, texture sol, inventaire MNS-MNT avec comptes, exports OpenUSD,
rehash du package, suppression des rasters bruts, scène unifiée et ZIP. Une
phase n'avance qu'après le retour de l'opération correspondante.

Après validation du ZIP, le worker publie dans la dataset privée 23 artefacts
sous `zones/<zone_id>/<build_id>/` : le ZIP autonome, `zone.done.json`,
`dataset-entry.json` et les 20 captures. L'upload est idempotent et échoue fermé si la cible n'est
pas privée ou si le jeton est absent. La dataset publique historique n'est pas
modifiée par ce parcours.

Modal n'est plus un fournisseur de cartes. Son workspace reste séparé et peut
être réutilisé plus tard par l'agent de collecte périodique ; aucune source de
collecte n'est inventée dans le worker cartographique.
