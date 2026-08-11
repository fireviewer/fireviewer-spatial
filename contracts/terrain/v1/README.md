# Contrats de production terrain v1

Ces schémas décrivent l'entrée mono-zone et les reçus immuables de
`terrainctl.py`. Les fichiers `zone-plan`, `zone-source-lock`, `tile-done` et
`zone-acceptance` ne contiennent ni chemin machine dans leur identité, ni état
de reprise mutable. `run-state.v1.json` reste local au lot de travail et ne
prouve jamais l'acceptation d'un terrain.

## Entrées verrouillées d'une zone

Une `zone-spec.v1` lie exactement huit dépendances à leur fichier et leur
SHA-256 sur `D:` :

- l'algorithme de compilation ;
- la bibliothèque `clean_pbr_texture_library` acceptée après revue visuelle ;
- le contrat de textures de sol v4 ;
- le contrat de correspondance orthophoto/surface v1 ;
- le modèle de correspondance v1 ;
- le snapshot `surface_features` autonome, incluant les `context_priors`, les
  `approved_corrections` et leur provenance SIG transitive ;
- le contrat quadtree ;
- la chaîne d'outils verrouillée.

Les anciens `atlas_catalog`, `atlas_visual_receipt`, `ground_context` et
`ground_runtime_contract` ne font pas partie de cette identité. La chaîne
refuse une bibliothèque `pending_clean_pbr_library` ou
`generated_pending_visual_review` : le préflight exige
`accepted_clean_pbr_library` et son reçu `accepted_human_visual` hashé.

Le validateur Python impose exactement trois `source_requests` par coeur de
tuile de 500 m, avec la même emprise Lambert-93 :

1. un MNT à 2 m ;
2. un MNS à 2 m ;
3. une orthophoto RGB temporaire à 1 m avec halo de 10 m.

Le MNT et le MNS peuvent être verrouillés par révision HTTPS ou par identité
`file:` complète. L'orthophoto exige toujours une `source_revision_id` et une
requête canonique HTTPS : WMS 1.3.0, ou WMTS 1.0.0 avec style et matrice 1 m
explicites. Les tailles, formats et paramètres de requête participent au hash
du plan. Le JSON Schema contrôle la forme de chaque enregistrement ; le
validateur Python contrôle la cardinalité exacte `3 × nombre_de_tuiles`,
l'unicité du triplet et l'égalité des emprises.

## Cycle de vie de l'orthophoto

L'orthophoto est une source de reconnaissance de production, jamais une
texture du terrain :

1. elle est téléchargée par bande dans le temporaire D: avec `.part`, hash et
   révision fournisseur ;
2. le moteur de correspondance produit
   `ground-profile-ids.png`, `ground-profile-weights.png`,
   `ground-confidence.png`, `ground-orientation.png` et
   `surface-correspondence.json` ;
3. après scellement des reçus dépendants, le RGB canonique, les réponses WMS ou
   WMTS, les `.part` et le répertoire brut sont supprimés immédiatement ;
4. seuls les hashes et la provenance sans chemin orthophoto restent dans les
   reçus.

Aucun pixel, chemin ou fichier orthophoto n'est admissible dans le package,
l'USD ou le runtime. La reprise doit reproduire les quatre cartes bit à bit à
partir d'une source de même identité avant d'autoriser son nettoyage.

## Packages écrits et archives

Toute nouvelle production écrit exclusivement :

- `fireviewer.tile-package.v3` dans `tile-package.v3.json` ;
- `fireviewer.tile.done.v3` dans `tile.done.v3.json` ;
- `fireviewer.ground-material-contract.v2` pour le matériau basé sur la
  bibliothèque PBR propre.

Le package v3 contient les trois FVTQ, le HAG compact, les quatre cartes de
surface et le manifeste de correspondance hashé. Il ne contient ni overlay
vectoriel runtime, ni composition v2, ni orthophoto, ni matériau procédural.

Les packages et reçus v2 existants restent consultables uniquement comme
archives en lecture seule. Aucun chemin de production, reprise, QA ou
acceptation ne peut créer, modifier ou promouvoir du v2.

## Identité, QA et acceptation

Chaque acquisition inscrit taille et SHA-256 réellement observés dans la
provenance de tuile. La QA agrège les sources dans un arbre de Merkle, lie les
hashes de tous les `tile.done.v3.json`, puis calcule le `build_id` final de
`zone.acceptance.v1`. Un remplacement fournisseur produit donc un nouveau
`build_id` sans rendre l'ancien package accepté non reproductible.

Le validateur Python reste l'implémentation de référence pour les invariants
qui dépassent JSON Schema : carré aligné sur la grille Lambert-93, un seul
triplet de sources par tuile, stockage exclusivement sur D:, exhaustivité des
tuiles et jointures, identité bit à bit des cartes communes, absence de résidu
orthophoto et cohérence entre package v3 et reçu done v3.

Les contrats runtime publics complètent les reçus d'orchestration :

- `camera-envelope.schema.json` décrit les poses et cadrages fournis par une
  future scène sans les inscrire dans le terrain ;
- `camera-residency-plan.schema.json` verrouille les ensembles LOD calculés par
  frustum et leurs budgets ;
- `terrain-streaming-state.schema.json` sérialise la transition atomique v2 ;
- `tile-package.schema.json` décrit la vérité canonique d'une tuile, ses trois
  LOD et ses seize variantes de raccord.

Les fixtures conformes sont des preuves de contrat, pas des reçus d'acceptation
d'une zone réelle. La QA produit d'abord un reçu technique ; une
`zone.acceptance.v1` ne devient admissible qu'après une revue visuelle humaine
exhaustive, identifiée et liée aux hashes des captures. Le contrôleur ne
transforme jamais automatiquement une QA technique en acceptation visuelle.
