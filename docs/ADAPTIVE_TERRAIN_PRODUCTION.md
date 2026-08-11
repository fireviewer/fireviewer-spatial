# Production des terrains adaptatifs

Ce document est le chemin de production cible des terrains FireViewer. Les
anciens pipelines de grille uniforme, d'orthophoto conservée comme texture et
de sélection radiale ne sont pas compatibles avec ce contrat. Une orthophoto
RGB à 1 m peut désormais intervenir uniquement comme source temporaire de
reconnaissance au build ; elle n'entre jamais dans le package ni dans le
runtime.

## État réel au 9 août 2026

Le système, ses contrats et ses fixtures synthétiques sont en qualification
locale. `0/6` zone réelle est produite ou acceptée et ce dépôt ne contient aucun
téléchargement MNT/MNS de production. Les reçus Blender produits par les tests
synthétiques prouvent uniquement le chemin technique testé ; ils ne valident ni
l'atlas réel des 72 profils, ni un terrain réel.

Le gate exhaustif atlas v3 a échoué en mode fail-closed. La bibliothèque cible
des 72 textures PBR propres et ses quatre atlas n'est donc pas disponible. Le
reçu léger
`ground-atlas-v3-acceptance-probe/atlas-render.failed-technical.v1.json`
identifie 24 cellules micro sombres ou plates. Le diagnostic
`adaptive-atlas-v3-probe/atlas-v3-contract-gap.json` identifie en plus quatre
profils `road_surface` `procedural_only`, sans échantillonnage atlas. Aucun reçu
de rendu `pending` et aucune acceptation visuelle n'ont été émis. Le prochain
gate est donc de corriger/refaire l'atlas, relancer les 216 cellules et 13 vues
de contrôle, puis obtenir la revue humaine exhaustive. Le compilateur et le
contrat de correspondance décrits dans
[`ORTHOPHOTO_TEXTURE_CORRESPONDENCE.md`](./ORTHOPHOTO_TEXTURE_CORRESPONDENCE.md)
sont désormais implémentés et leurs six tests synthétiques sont verts. Leur
intégration au backend mono-zone, à `tile-package.v3` et à `tile.done.v3` est
réalisée. Sans une bibliothèque PBR réelle acceptée, le `preflight` de Lédenon
échoue volontairement et aucun téléchargement de sa zone ne doit commencer. Le
gate Blender texturé utilisant le vrai relief FVTQ et la projection triplanaire
est techniquement qualifié sur fixture synthétique : Blender 4.5.3, deux vues
512 px, zéro pixel LOD/couverture invalide et 44 544 pixels triplanaires. Son
statut reste toutefois `pending_surface_library`, sans acceptation de texture
réelle ni revue humaine.

## Invariants

- La vérité géographique est exprimée en mètres dans `EPSG:2154`.
- Une zone est un carré explicite composé de tuiles coeur de 500 m.
- Le MNT et le MNS de travail sont co-enregistrés sur une grille globale de
  2 m. Un halo de 10 m est acquis autour des tuiles dépendantes.
- Les hauteurs de décision sont quantifiées au millimètre avant subdivision.
- Le résultat canonique est un quadtree adaptatif et déterministe, pas le
  raster de travail.
- L'orthophoto RGB de reconnaissance est acquise à 1 m par fenêtre de 500 m
  avec halo de 10 m. Elle n'est supprimée qu'après le scellement de
  `tile-package.v3.json` et la validation de `tile.done.v3.json` contre toutes
  ses sorties. Elle n'est jamais une dépendance du package.
- Aucun matériau procédural ou profil `procedural_only` n'est admissible dans
  une nouvelle production.
- Une seule zone peut être planifiée ou produite par invocation.
- Aucune phase n'est franchie sans le reçu valide de la phase précédente.
- Aucun LOD autre que LOD0 n'est publiable dans les pixels terrain d'une
  caméra principale RGB ou thermique.

## Représentation canonique

Chaque tuile acceptée contient au minimum :

```text
tiles/<tile_id>/
├── source/
│   ├── mnt-provenance.v1.json
│   └── mns-provenance.v1.json
├── terrain-lod0.fvtq
├── terrain-lod1.fvtq
├── terrain-lod2.fvtq
├── hag-max-2m.tif
├── ground-profile-ids.png
├── ground-profile-weights.png
├── ground-confidence.png
├── ground-orientation.png
├── surface-correspondence.json
├── tile-package.v3.json
├── tile.done.v3.json
├── terrain-lod0.usda
├── terrain-lod1.usda
├── terrain-lod2.usda
├── terrain-tile.usda
└── terrain-usd-package.v1.json
```

Le format, les noms des quatre cartes et le manifeste sont verrouillés par
[`orthophoto_surface_correspondence_contract.v1.json`](../blender/orthophoto_surface_correspondence_contract.v1.json).
Le package de zone cible contient en plus un seul
bundle partagé `shared/ground-material/` : 72 profils PBR propres, quatre atlas
runtime (`basecolor`, `normal`, `height`, `ORM`),
`ground-material-contract.v2.json` et couche USD. Cette bibliothèque n'existe
pas encore sous une forme acceptée. Les tuiles v3 enregistrent son identité
sans dupliquer les textures.

Les trois fichiers FVTQ sont des vues imbriquées d'un même arbre. Ils utilisent
le format little-endian `fireviewer.terrain-quadtree.v1`, l'ordre Morton et des
hauteurs entières relatives au datum local. LOD2 est un sous-ensemble de LOD1,
lui-même sous-ensemble de LOD0. Une recompilation à partir des mêmes entrées
doit produire les mêmes octets, indépendamment du nombre de workers, du chemin
de travail ou d'une reprise.

| Niveau | Erreur verticale | Cellule plane maximale | Raffinement local |
| --- | ---: | ---: | ---: |
| LOD0 | 0,50 m | 31,25 m | 3,906 m, 1,953 m sur rupture contrainte |
| LOD1 | 2 m | 125 m | 15,625 m |
| LOD2 | 8 m | 500 m | 62,5 m |

Les frontières sont résolues à l'échelle de la zone : sommets et hauteurs de
bord identiques, arbres équilibrés 2:1 et signature déterministe des raccords.
Une tuile n'est pas acceptée si une des seize configurations de bord produit
un trou ou une discontinuité.

`tile-package.v3.json` et `tile.done.v3.json` enregistrent les seize variantes
de raccord de chaque LOD : masque, coût en triangles, hash de la liste d'indices,
erreur maximale et quatre signatures de bord. La QA rejoue sur chaque jointure
les sept couples de LOD admissibles (`0/0`, `0/1`, `1/0`, `1/1`, `1/2`, `2/1`,
`2/2`) et refuse toute signature différente.

Les packages `tile-package.v2.json`/`tile.done.v2.json` et leurs compositions
restent pris en charge en lecture seule pour rejouer et auditer l'historique.
Aucune nouvelle production ne les écrit.

Le HAG compact est calculé directement depuis les grilles canoniques MNT/MNS
en millimètres entiers, puis arrondi en centimètres entiers. Aucun aller-retour
en flottants n'intervient dans son identité canonique.

## Correspondance du sol

Chaque coeur de 500 m reçoit quatre cartes de 500 × 500 pixels : IDs de profils
parmi les 72, poids, confiance et orientation. Elles sont produites depuis une
fenêtre orthophoto RGB temporaire de 520 × 520 pixels couvrant le coeur et son
halo de 10 m. La fenêtre est hashée et classifiée, ses quatre cartes et son
manifeste sont incorporés au package v3, puis la source RGB n'est supprimée
qu'après validation de `tile.done.v3.json`.

Les parcelles, l'occupation du sol, le transport, l'hydrographie et la géologie
servent uniquement de priors et de corrections sémantiques. Ils ne peignent
plus le sol comme source principale. Une confiance insuffisante bloque la
fenêtre ou exige une correction explicite ; elle ne déclenche ni matériau
procédural, ni fallback orthophoto.

L'ancienne grille de fond à 5 m et ses `surface-overlays.json.gz` v2 sont
dépréciées en écriture pour toute nouvelle production. Les lecteurs v2 restent
disponibles uniquement pour le replay et l'audit des packages historiques.
Cette dépréciation est limitée à la composition de sol v2 et ne vise pas tous
les autres contrats portant un numéro v2.

Les quatre atlas partagés cibles apporteront le détail PBR. Aucune image source
ni orthophoto ne sera importée individuellement dans une scène. Le contrat
détaillé et son état d'implémentation sont suivis dans
[`ORTHOPHOTO_TEXTURE_CORRESPONDENCE.md`](./ORTHOPHOTO_TEXTURE_CORRESPONDENCE.md).
L'API exécutable et ses invariants de sérialisation sont documentés dans
[`ORTHOPHOTO_SURFACE_CORRESPONDENCE.md`](../blender/ORTHOPHOTO_SURFACE_CORRESPONDENCE.md).

## Gates de rendu terrain et USD

Le gate terrain est Blender headless. Son matériau de référence lit les quatre
atlas, les IDs, poids, confiance et orientation, puis compose les projections
métriques `world_xy` et `world_triplanar` sur le LOD0 FVTQ. Ce chemin technique
est vert sur la fixture synthétique avec ses AOV hashés. Le gate de production
reste fermé jusqu'à la bibliothèque PBR réelle et à la revue humaine liée aux
hashes ; la preuve synthétique ne les remplace pas.

La couche OpenUSD portable garde volontairement un `UsdPreviewSurface` magenta
de diagnostic. Il ne constitue ni un rendu PBR de référence, ni un fallback de
production. Le shader runtime reste
`pending_dedicated_mdl_validation` et l'acceptation de zone conserve
`usd_runtime_gate=false` jusqu'à validation d'un MDL dédié hash-locké. Ce gate
runtime séparé ne bloque pas l'acceptation du terrain texturé dans Blender ; il
bloque seulement la qualification du rendu texturé OpenUSD/Omniverse.

## Résidence pilotée par la caméra

Le planificateur teste les AABB 3D des tuiles contre les six plans du frustum
réel. Il construit les ensembles suivants :

- `visible_lod0` : intersection exacte avec le frustum ;
- `guard_lod0` : couronne de 500 m autour du frustum ;
- `staging_lod0` : caméra suivante ou frusta prédits ;
- `resident_lod1` : deux couronnes supplémentaires, soit 1 km ;
- `resident_lod2` : reste du carré.

Le budget préalable couvre l'ensemble LOD0 actif, de garde et de staging, la
couronne LOD1, la totalité de LOD2, puis 25 % de réserve. Il n'existe aucune
limite arbitraire à seize tuiles. Un cadrage dépassant le budget échoue avant
le rendu ; il ne modifie jamais silencieusement le champ, le far clip ou le
LOD visible.

Une transition planifiée charge la caméra suivante dans un buffer caché,
vérifie ses hashes, son budget et sa couverture, puis publie sa caméra et son
ensemble LOD0 en une opération atomique. La vue précédente peut rester trois
secondes avant démotion. En interactif, l'union des frusta prédits sur deux
secondes, échantillonnés toutes les 250 ms, est préchargée. Une rotation ou une
téléportation conserve la dernière vue complète jusqu'à résidence intégrale de
la nouvelle vue.

## CLI et phases

Une nouvelle zone est décrite hors ligne, sans téléchargement, par le
générateur mono-zone
[`prepare_adaptive_zone_specs.py`](../blender/prepare_adaptive_zone_specs.py).
Son contrat et le catalogue signé des six emprises sont documentés dans
[`ADAPTIVE_ZONE_SPEC_GENERATOR.md`](../blender/ADAPTIVE_ZONE_SPEC_GENERATOR.md).
L'ancien `prepare_incident_terrains.py` en 0,5 m est historique et ne peut pas
alimenter ce chemin actif.

```powershell
python terrainctl.py `
  --zone D:\fireviewer-work\zones\FR-30-00001\zone-spec.v1.json `
  --phase plan `
  --dry-run
```

`terrainctl.py` accepte exactement une phase parmi `plan`, `preflight`,
`pilot`, `produce`, `qa`, `accept` et `cleanup`, puis exactement un mode parmi
`--dry-run`, `--execute` et `--resume`. L'option sentinelle `--all` est
explicitement refusée et plusieurs `--zone` provoquent une erreur.

Le CLI installe par défaut le backend mono-zone concret : acquisition MNT/MNS
2 m et orthophoto temporaire 1 m, extraction canonique 253 x 253 avec halo de
normales, double compilation FVTQ, HAG, correspondance des sols, scellement
`tile-package.v3`/`tile.done.v3`, OpenUSD et QA Blender. Une dépendance, une
source ou une preuve absente arrête la phase. La composition de sol v2 reste
une capacité de lecture/qualification historique ; elle n'est jamais écrite
par ce chemin. Un marqueur et un verrou globaux sur `D:` empêchent deux zones
d'avancer même si leurs dossiers de travail sont distincts.

Les sorties lourdes restent hors Git, sous un répertoire de production sur
`D:`. Les temporaires et caches de production sont également redirigés vers
`D:`. Le préflight exige le pic disque mesuré plus 20 Gio libres.

Pour une source HTTPS encore jamais téléchargée, `source_revision_id` est
obligatoire mais `expected_sha256` et `expected_byte_count` restent optionnels.
Le préflight émet alors un `recipe_build_id` provisoire sans prétendre connaître
les octets distants. Les acquisitions inscrivent leurs tailles et hashes
observés dans les provenances ; la QA calcule un arbre de Merkle couvrant toutes
les sources et tous les reçus de tuile. `zone.acceptance.v1` expose à la fois le
`recipe_build_id` et le `build_id` final. Une source locale `file:` conserve
l'obligation de taille et SHA-256 avant exécution.

| Phase | Entrée bloquante | Sortie attendue |
| --- | --- | --- |
| `plan` | zone-spec et contrats valides | plan immuable et estimation |
| `preflight` | reçu plan + bibliothèque PBR, atlas et correspondance acceptés | sources, contrats, CRS, espace et hashes contrôlés |
| `pilot` | reçu preflight | fixture 3 x 3 et mesures réelles |
| `produce` | pilote accepté | packages canoniques de toutes les tuiles |
| `qa` | production complète | QA structurelle + rendu de zone `rendered_pending_zone_visual_review` + modèle de revue |
| `accept` | QA complète + revue humaine exhaustive et hash-liée | `zone-visual.accepted_blender_visual.v2.json` puis `zone.acceptance.v1.json`, avec gate USD runtime séparé à `false` |
| `cleanup` | acceptation valide | sources et caches de travail supprimables |

Chaque fenêtre orthophoto suit en plus une boucle strictement séquentielle :
acquisition/reprise à 1 m sur `D:`, classification, corrections SIG, scellement
des quatre cartes 500 × 500, écriture du package v3, validation de
`tile.done.v3.json`, puis suppression du RGB, des réponses et des `.part`
avant la fenêtre suivante. Le mode `cleanup` exécute un plan de
suppression borné et vérifiable après
acceptation. Seuls les répertoires de sources, temporaires et caches du run,
ainsi que ses fichiers `.part`, sont supprimés. L'opération est idempotente et
conserve les FVTQ, cartes de correspondance acceptées, USD, HAG compact,
bundle de matériau, index, reçus et preuves Blender.

## Qualification et ordre de production

La qualification commence par la bibliothèque réelle des 72 textures PBR
propres et ses quatre atlas. Le compilateur de correspondance est implémenté et
ses raccords synthétiques groupés/séparés sont bit-identiques ; son intégration
backend/package v3 est réalisée, tandis qu'une preuve sur source réelle reste
ouverte. Le gate Blender texturé et son chemin `world_triplanar` sont qualifiés
sur une fixture terrain synthétique contiguë 2 x 2. Viennent ensuite la
bibliothèque PBR réelle acceptée, une fenêtre réelle contrôlée, puis
un pilote 3 x 3 de Lédenon. Si `pilot.regression_tile_id` désigne une tuile hors
du bloc sélectionné, elle est ajoutée comme dixième preuve sans casser la
continuité du bloc 3 x 3. Aucun téléchargement général ne commence tant que
tous ces gates ne sont pas acceptés.

L'ordre des zones est verrouillé : Lédenon, Oupia, Die, Taradeau,
Fontainebleau, puis Trévillach. Une zone est produite, contrôlée, acceptée et
nettoyée avant toute acquisition de la suivante. Die ne réutilise ni son ancien
terrain, ni son ancien sol, ni ses placements.

| Ordre | Incident | Carré | Couples MNT/MNS actifs | Tuiles | Jointures |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | FR-30-00001 — Lédenon | 14,5 km | 841 | 841 | 1 624 |
| 2 | FR-34-00001 — Oupia | 16 km | 1 024 | 1 024 | 1 984 |
| 3 | FR-26-00001 — Die | 20 km | 1 600 | 1 600 | 3 120 |
| 4 | FR-83-00001 — Taradeau | 22,5 km | 2 025 | 2 025 | 3 960 |
| 5 | FR-77-00001 — Fontainebleau | 23 km | 2 116 | 2 116 | 4 140 |
| 6 | FR-66-00001 — Trévillach | 27,5 km | 3 025 | 3 025 | 5 940 |

Le chemin actif crée exactement une paire de requêtes MNT/MNS par tuile. Les
`2 782` couples de l'ancien découpage v2 ne sont ni téléchargés, ni réutilisés.

Gate global : `6/6` zones, `10 631/10 631` tuiles et
`20 768/20 768` jointures acceptées, sans source raster brute résiduelle.

Le gate global exige six zones acceptées, tous leurs packages et raccords
validés, aucune source brute résiduelle et une revue visuelle globale. Les
bâtiments, routes 3D, petits assets, végétation et simulations restent bloqués
jusqu'à cette acceptation.

## Preuves bloquantes

- hashes FVTQ identiques sur deux compilations ;
- erreurs LOD mesurées sous les seuils contractuels ;
- hiérarchie stricte LOD2 dans LOD1 dans LOD0 ;
- frontières et raccords sans trou ;
- cartes IDs/weights/confidence/orientation de 500 × 500 alignées et continues ;
- absence d'orthophoto et de matériau procédural dans les packages ;
- `tile.done.v3.json` validé avant toute suppression du RGB temporaire ;
- refus des nodata, CRS ou transforms incohérents ;
- couverture LOD0 complète avant chaque publication ;
- dépassement mémoire et échec de chargement traités en mode fail-closed ;
- AOV `fireviewer:terrain_lod` égal à zéro pour chaque pixel terrain de la
  caméra principale ;
- une vue orthographique du carré, neuf cadrages 3×3, jusqu'à vingt coutures
  bi-tuile et les cadrages obliques relief/réseaux/surfaces, tous avec AOV LOD
  et couverture ;
- aucune acceptation visuelle créée automatiquement : le reçu technique reste
  en attente jusqu'à une revue humaine explicite de chaque capture et de chaque
  hash ;
- réouverture des USD générés et contrôle Blender headless avec le Blender LTS
  déjà installé sur `D:` ;
- composition PBR Blender de référence `world_xy`/`world_triplanar` validée sur
  le LOD0 FVTQ ; le PreviewSurface USD magenta reste explicitement non
  qualifiant tant que `usd_runtime_gate=false`.
