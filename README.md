# FireViewer Spatial

Outils source pour préparer, valider et intégrer les packages OpenUSD et
Omniverse de FireViewer. Le runtime Unity et ses scènes de site sont retirés du
périmètre actif.

Ce dépôt ne contient aucune carte, orthophoto, scène générée, archive de
production, dossier d’upload, modèle 3D, atlas, arbre source ou donnée
d’incident. Les packages validés sont publiés séparément dans GitHub Releases
ou un stockage Blob.

## Position dans l’architecture événementielle v2

Ce dépôt fournit les référentiels et contrats géométriques utilisés pour
localiser des phénomènes actifs à partir de contributions documentées. Il ne
déduit pas une localisation depuis un texte et ne publie aucune géométrie.

Une branche spatiale admissible relie explicitement :

- le point de prise de vue privé et son incertitude ;
- le modèle de caméra, l’orientation et le profil de vue ;
- l’ancrage visuel rattaché à une preuve ;
- la révision immuable du terrain et de la banque de rendus ;
- la pose, le raycast ou la triangulation ;
- le résultat géométrique, son incertitude et ses contrôles.

En l’absence d’orientation, de pose, d’intersection terrain ou de précision
suffisante, le résultat reste un secteur ou une abstention. Le point de prise
de vue n’est jamais assimilé au point actif. Les matchers cross-view restent en
benchmark ou en shadow tant qu’un benchmark événementiel indépendant ne
justifie pas leur promotion.

## Contenu

- `blender/` : génération et contrôles géométriques ;
- `omniverse/` : export et validation structurelle des payloads OpenUSD ;
- `omniverse/contracts/v2/` : contrat futur du catalogue de 295 assets et des
  compositions reproductibles, après reconstruction des couches terrain ;
- `contracts/spatial/` : schémas du package et du catalogue ;
- `tests/` : fixtures synthétiques sans données de production.

## Production

Chaque production utilise un dossier de travail externe sur `D:` et une
configuration de zone explicite. Pour les six incidents actifs, le MNT/MNS de
travail est aligné sur une grille globale de 2 m en `EPSG:2154`. Il est compilé
en tuiles coeur de 500 m avec halo de 10 m, sous la forme d'un quadtree
adaptatif à trois LOD. Les rasters de travail ne constituent pas la livraison
et sont supprimables dès que toutes leurs dépendances sont validées.

Le sol cible référence 72 textures PBR propres, empaquetées dans quatre atlas
runtime (`basecolor`, `normal`, `height`, `ORM`). Cette bibliothèque propre
n'existe pas encore sous une forme acceptée : les images et l'atlas v3 actuels
ne constituent pas ce livrable.

La nouvelle production utilisera temporairement une orthophoto RGB à 1 m par
fenêtre de 500 m avec halo de 10 m. Elle sert uniquement à classifier le sol
vers les 72 profils. Le coeur produit quatre cartes de 500 × 500 pixels : IDs,
weights, confidence et orientation. Le RGB temporaire n'est supprimé qu'après
le scellement de `tile-package.v3.json` et la validation de
`tile.done.v3.json` contre toutes ses sorties. Il n'entre jamais dans Blender,
OpenUSD, un package ou le runtime. Aucun matériau procédural ou profil
`procedural_only` n'est admis. Le compilateur déterministe
[`orthophoto_surface_correspondence.py`](./blender/orthophoto_surface_correspondence.py),
son acquisition temporaire, le backend mono-zone et le package v3 sont reliés.
Cette intégration testée sur données synthétiques ne constitue pas une preuve
de production : aucune orthophoto réelle n'a été téléchargée.

Les parcelles, l'occupation du sol, le transport, l'hydrographie et la géologie
restent des priors et des corrections de la classification, pas la source
principale qui peint le terrain. L'ancienne composition v2 à grille 5 m et ses
overlays vectoriels sont dépréciés pour toute nouvelle écriture. Les lecteurs
v2 sont conservés uniquement pour le replay et l'audit des anciens packages.
Les entrées LiDAR, vecteurs, images, modèles et rendus restent hors du checkout.

La QA terrain Blender headless compose les quatre atlas avec les projections
métriques `world_xy` et `world_triplanar` sur le véritable LOD0 FVTQ, puis
produit des vues et un AOV LOD vérifiés par hash. Le chemin technique
synthétique est qualifié avec Blender 4.5.3 en 512 px : les vues top-down et
oblique ont zéro pixel LOD/couverture invalide et 44 544 pixels utilisent
réellement `world_triplanar`. Ce résultat ne vaut ni acceptation de la future
bibliothèque PBR réelle, ni revue humaine. La couche OpenUSD conserve
volontairement un
`UsdPreviewSurface` magenta de diagnostic : le shader runtime dédié reste
`pending_dedicated_mdl_validation` et `usd_runtime_gate=false`. Ce gate USD
séparé ne bloque pas l'acceptation terrain dans Blender, mais il bloque toute
qualification de rendu texturé Omniverse/RTX. Les 295 assets USD refaits
bloquent leur couche de composition et les packs finaux, mais pas la
reconstruction terrain/sol. Les rails métalliques restent une future géométrie
3D ; les matériaux 2D ne couvrent que ballast, traverses, talus et
accotements.

Le dataset historique issu de la première simulation et son pack autonome de
reproduction complet sont conservés hors Git. Ils servent au replay et à
l'audit, mais ne définissent pas le contrat des futures scènes.

Les packages et banques de rendus sont versionnés par zone et par révision. Les
sorties de localisation conservent les CRS horizontaux, datums verticaux,
transformations, résolutions et empreintes exactes nécessaires au replay.

Le package canonique v3 conserve les trois fichiers FVTQ, le HAG compact, les
quatre cartes de correspondance, `surface-correspondence.json`, le contrat de
matériau v2 et leur provenance. Les packages v2 restent lisibles uniquement
pour le replay et l'audit. Deux compilations depuis les mêmes entrées doivent
être identiques bit à bit. Le streamer sélectionne les tuiles par intersection
de leur AABB 3D avec le frustum réel : seule la couverture LOD0 complète peut
être publiée dans une caméra principale. LOD1 et LOD2 ne servent jamais de
fallback visible pendant une capture.

État au 9 août 2026 : `0/6` zone réelle est produite ou acceptée. Le gate atlas
v3 a échoué en mode fail-closed : 24 cellules micro sont sombres ou plates et
quatre profils `road_surface` restent `procedural_only`, sans contrat
d'échantillonnage atlas. Le rendu n'a donc émis ni reçu `pending`, ni
acceptation visuelle. Il faut
corriger/refaire la bibliothèque PBR et ses quatre atlas, relancer leur rendu
technique complet, les contrôler avec le gate Blender texturé désormais
qualifié sur fixture, puis obtenir la revue humaine. Seulement après ces gates viendront
le preflight et le pilote de Lédenon. Une preuve synthétique ne remplace jamais
cette chaîne ; l'état reste `0/6`.

## Documentation de référence

- `docs/CAMERA_AND_CRS_CONTRACT.md` : repères, caméra, pose et contrôles ;
- `docs/ADAPTIVE_TERRAIN_PRODUCTION.md` : terrain adaptatif, composition,
  streaming caméra et gates de production ;
- `docs/ORTHOPHOTO_TEXTURE_CORRESPONDENCE.md` : orthophoto temporaire 1 m,
  cartes 500 × 500, bibliothèque PBR cible et règles de nettoyage ;
- `blender/ORTHOPHOTO_SURFACE_CORRESPONDENCE.md` : API déterministe, contrat
  des cinq sorties, hashes et qualification synthétique du matcher ;
- `blender/ADAPTIVE_ZONE_SPEC_GENERATOR.md` : génération hors ligne d'un unique
  `zone-spec.v1` 2 m à partir du catalogue signé des six emprises ;
- `docs/PACKAGE_VERSIONING.md` : immutabilité et dépendances des révisions ;
- `docs/RENDER_BANK_SPEC.md` : contenu et portée des banques de rendus ;
- `docs/UAV_REGISTRATION_BENCHMARK.md` : protocole de comparaison des matchers.

La doctrine produit, les contrats transverses et la matrice d’acceptation sont
maintenus dans le dépôt canonique `fireviewer/Fireviewer_doc`.

## Contrôles

```powershell
python -m unittest discover -s tests -v
python -m pytest -q
python -m ruff check .
python omniverse/contracts/v2/validate_contracts.py
```

Ces contrôles valident le code et les contrats. Ils ne prouvent pas une scène
Omniverse, un rendu RTX, un package cartographique ou une publication distante.

## Licences

Le code est sous AGPL-3.0-or-later et la documentation sous CC BY 4.0. Les
sources IGN, OpenStreetMap et autres données externes conservent leurs licences
propres et ne sont pas redistribuées dans ce dépôt.
