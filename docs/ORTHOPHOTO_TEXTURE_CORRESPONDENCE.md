# Correspondance orthophoto vers textures PBR

Ce document décrit le chemin cible des sols FireViewer. Il remplace, pour les
nouvelles productions, la composition principalement procédurale et les
overlays vectoriels de sol de la révision historique v2.

## État réel au 9 août 2026

- `0/6` terrain réel est produit ou accepté.
- Aucune des 72 textures PBR propres cibles n'est déclarée terminée.
- La bibliothèque propre et ses quatre atlas runtime restent à produire puis à
  qualifier visuellement.
- L'atlas v3 actuel est rejeté : 24 cellules micro sont sombres ou plates et
  quatre profils `road_surface` restent `procedural_only`.
- Le compilateur déterministe
  [`orthophoto_surface_correspondence.py`](../blender/orthophoto_surface_correspondence.py)
  et son contrat public verrouillé sont implémentés.
- Ses six tests synthétiques sont verts : champs, forêt, eau, roche, route et
  chemin, restrictions SIG, correction approuvée, quantification des poids et
  identité bit à bit indépendante du découpage en fenêtres.
- L'acquisition temporaire, le compilateur, le backend mono-zone et le
  scellement `tile-package.v3`/`tile.done.v3` sont intégrés.
- Seule l'acquisition temporaire d'une source RGB synthétique ou distante est
  préparée et testée localement. Aucun téléchargement réel de production n'a
  été lancé.

Une fixture, un test ou une image de contrôle ne constitue ni une texture PBR
propre acceptée, ni un terrain réel produit.

## Décision

L'orthophoto devient une source de reconnaissance temporaire au moment du
build. Elle sert uniquement à déterminer quel profil de sol propre doit être
appliqué à chaque mètre du terrain.

Elle n'est jamais :

- une texture de terrain Blender ou OpenUSD ;
- un fallback visuel ;
- un fichier du package canonique ou téléchargeable ;
- une dépendance du runtime ;
- conservée après la validation du `tile.done.v3.json` qui signe toutes ses
  sorties dépendantes.

Le runtime reçoit seulement des identifiants, poids, confiance et orientation
référençant une bibliothèque PBR propre. Aucun matériau procédural ne peut
remplacer une texture manquante ou une classification incertaine.

## Fenêtre de travail

L'unité d'acquisition est alignée sur une tuile terrain :

| Propriété | Valeur verrouillée |
| --- | ---: |
| CRS | `EPSG:2154` |
| coeur | 500 m × 500 m |
| résolution RGB de reconnaissance | 1 m |
| halo de calcul | 10 m |
| image RGB canonique temporaire | 520 × 520 pixels |
| cartes livrables du coeur | 500 × 500 pixels |

Le mode 0,5 m est exclu. Le halo apporte le contexte nécessaire aux pixels de
bord, puis il est retiré des cartes finales. Deux fenêtres voisines prennent
leurs décisions dans le même repère Lambert-93 et doivent produire des bords
compatibles.

Les requêtes WMS 1.3 ou WMTS sont canoniques et liées à une révision
fournisseur. Les octets reçus, la requête, le RGB canonique et son
géoréférencement sont hashés. Les transferts utilisent des fichiers `.part` et
une reprise HTTP stricte. Tout le travail reste dans un répertoire temporaire
hors Git sur `D:`.

## Bibliothèque PBR cible

La bibliothèque cible contient exactement 72 profils de sol propres. Chaque
profil doit fournir une cellule cohérente dans quatre atlas partagés :

1. `basecolor` ;
2. `normal` ;
3. `height` ;
4. `ORM` pour occlusion, roughness et metallic.

Ces 72 jeux de textures ne sont pas encore disponibles ni acceptés. Le futur
catalogue devra verrouiller pour chaque profil son identifiant, sa sémantique,
ses quatre cellules, son échelle physique, ses gutters et leurs SHA-256.

Une variation d'UV, une orientation ou un mélange déterministe peut composer
des textures propres. Cela ne permet pas de synthétiser un matériau manquant,
d'utiliser un bruit comme surface principale ou de conserver un profil
`procedural_only`. Les 72 profils devront tous échantillonner les atlas
acceptés.

Le bundle partagé est verrouillé par `ground-material-contract.v2`. Blender
est l'implémentation de référence du gate terrain texturé : il compose les
quatre atlas en `world_xy` ou en `world_triplanar` sur la surface LOD0 FVTQ. La
qualification technique synthétique est passée avec deux vues Blender 512 px,
zéro pixel LOD/couverture invalide et 44 544 pixels triplanaires. Le reçu reste
explicitement en attente de la bibliothèque PBR réelle et de la revue humaine.

L'export OpenUSD conserve volontairement un `UsdPreviewSurface` magenta de
diagnostic. Il ne démontre pas le rendu PBR et ne peut pas servir de fallback.
Le shader runtime dédié reste `pending_dedicated_mdl_validation` avec
`usd_runtime_gate=false`. Cette séparation ne bloque pas l'acceptation terrain
dans Blender ; elle réserve la qualification OpenUSD/Omniverse texturée à une
future validation MDL dédiée.

## Cartes de correspondance implémentées

Pour chaque coeur de 500 m, le compilateur scelle quatre cartes alignées de
500 × 500 pixels :

| Fichier | Format | Rôle |
| --- | --- | --- |
| `ground-profile-ids.png` | RGBA8 | quatre indices stables parmi les 72 profils |
| `ground-profile-weights.png` | RGBA8 | poids dont la somme vaut exactement 255 |
| `ground-confidence.png` | L8 | confiance de la décision de correspondance |
| `ground-orientation.png` | L8 | axe non dirigé dans `[0, π[` pour les surfaces orientées |

Le manifeste compact `surface-correspondence.json`, de schéma
`fireviewer.surface-correspondence-tile.v1`, signe les quatre cartes, la source
orthophoto et l'extrait tuile+halo uniquement par hash, la bibliothèque PBR, le
modèle, l'algorithme, le contrat, les priors et les corrections approuvées. Il
ne contient ni chemin, ni pixel, ni couleur issus de l'orthophoto. Les indices
suivent `stable_index` `0..71` du contrat v4, jamais un tri alphabétique.

La spécification d'API, de sérialisation et de performance est détaillée dans
[`ORTHOPHOTO_SURFACE_CORRESPONDENCE.md`](../blender/ORTHOPHOTO_SURFACE_CORRESPONDENCE.md).

Les cartes doivent conserver :

- le hash du RGB temporaire observé et la révision fournisseur ;
- le hash de la bibliothèque des 72 profils et des quatre atlas ;
- le hash de l'algorithme de correspondance ;
- le repère, l'emprise, la résolution et l'ordre des lignes ;
- les corrections SIG appliquées et leur provenance ;
- une preuve de continuité avec les fenêtres adjacentes.

Une confiance insuffisante provoque un rejet ou une correction explicite. Elle
ne déclenche jamais un matériau procédural ou l'import de l'orthophoto.

## Rôle des données SIG

Les parcelles, l'occupation du sol, le transport, l'hydrographie et la géologie
restent des priors et des corrections sémantiques :

- ils restreignent ou corrigent les classes plausibles ;
- ils stabilisent l'identité et l'orientation des surfaces étroites ;
- ils arbitrent les ponts, tunnels, gués et limites ambiguës ;
- leur provenance est enregistrée dans le reçu de correspondance.

Ils ne constituent plus le moteur principal qui peint le sol. Pour la nouvelle
production, l'ancienne composition v2 et ses `surface-overlays.json.gz` sont
dépréciées en écriture. La compatibilité v2 reste disponible uniquement en
lecture pour rejouer et auditer les anciens packages ; aucun nouveau terrain ne
doit être produit avec ces overlays.

Cette dépréciation vise la composition de sol et les overlays v2. Elle ne
déprécie pas par extension tous les autres formats portant un numéro de version
v2 dans le dépôt.

## Production séquentielle et nettoyage

Une seule fenêtre orthophoto est traitée à la fois dans une seule zone :

```text
plan WMS/WMTS 1 m + révision fournisseur
        ↓
contrôle du pic disque et de la marge libre sur D:
        ↓
téléchargement/reprise de la fenêtre 500 m + halo
        ↓
RGB canonique 520 × 520 hashé
        ↓
classification + priors/corrections SIG
        ↓
cartes coeur 500 × 500 contrôlées et scellées
        ↓
tile-package.v3.json écrit + tile.done.v3.json validé
        ↓
suppression du RGB, des réponses et des .part de la fenêtre
        ↓
fenêtre suivante
```

Le téléchargement de la fenêtre suivante est interdit tant que
`tile.done.v3.json` n'a pas validé toutes les sorties dépendantes de la fenêtre
courante et que son répertoire temporaire n'est pas vide. Un rejet conserve
uniquement le lot fautif minimal et son diagnostic, puis la reprise utilise le
même plan hashé.

Aucune orthophoto, mosaïque, réponse WMS/WMTS ou source RGB ne doit subsister
dans le package de tuile, le package de zone, Git ou les packs téléchargeables.

## Gates avant le premier terrain réel

1. produire les 72 jeux PBR propres et les empaqueter dans quatre atlas ;
2. corriger les défauts ayant fait rejeter l'atlas v3 ;
3. valider techniquement puis humainement les 72 profils et les quatre atlas ;
4. rejouer le gate Blender texturé déjà qualifié synthétiquement avec la
   bibliothèque réelle et accepter sa projection `world_triplanar` ;
5. exécuter une fenêtre réelle contrôlée de Lédenon, puis prouver ses raccords
   et l'absence totale d'orthophoto dans le package reconstruit ;
6. seulement alors lancer, séquentiellement, le pilote réel de Lédenon.

Les cinq autres zones restent bloquées derrière l'acceptation et le nettoyage
de Lédenon. Les bâtiments, routes 3D, petits assets, végétation et simulations
restent hors de cette phase.
