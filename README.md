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
- `omniverse/` : contrats OpenUSD, composition déterministe et outils Kit ;
- `omniverse/contracts/v2/` : contrat futur du catalogue de 295 assets et des
  compositions reproductibles, après reconstruction des couches terrain ;
- `contracts/spatial/` : schémas du package et du catalogue ;
- `tests/` : fixtures synthétiques sans données de production.

## Production

Chaque production utilise un dossier de travail externe et une configuration de
zone explicite. Pour les six incidents actifs, le relief 3D est dérivé du
MNT/MNS IGN aligné à 0,5 m. Le sol 2D léger combine une palette PBR et des
masques classifiés : sols naturels/brûlés, champs orientés, routes, chemins,
berges et plateformes ferroviaires. Les réseaux, parcelles et classes de sol
doivent provenir de couches vectorielles ou classifiées approuvées ; toute
orthophoto ou imagerie aérienne lourde est interdite. Les entrées LiDAR,
vecteurs, modèles et rendus restent hors du checkout. Les manifestes générés
conservent provenance, licences, tailles et SHA-256.

Les sources ImageGen rapprochées servent uniquement de micro-détail hors ligne.
Elles sont empaquetées dans exactement quatre textures atlas runtime. Les 72
profils de surface ajoutent la variété à 16–64 m et 128–512 m par paramètres,
bruit déterministe et masques de contexte ; aucune source individuelle n’est
importée dans la scène. Les routes, chemins, cours d’eau et voies ferrées
conservent une abscisse UV continue entre tuiles et changent de profil tous les
250 m sans répéter les deux variantes précédentes.

La validation automatisée ne remplace ni l'ouverture isolée dans Kit, ni la
revue visuelle humaine. Les 295 assets USD refaits bloquent leur couche de
composition et les packs finaux, mais pas la reconstruction terrain/sol. Les
rails métalliques restent une future géométrie 3D ; les matériaux 2D ne couvrent
que ballast, traverses, talus et accotements.

Le dataset historique issu de la première simulation et son pack autonome de
reproduction complet sont conservés hors Git. Ils servent au replay et à
l'audit, mais ne définissent pas le contrat des futures scènes.

Les packages et banques de rendus sont versionnés par zone et par révision. Les
sorties de localisation conservent les CRS horizontaux, datums verticaux,
transformations, résolutions et empreintes exactes nécessaires au replay.

Le validateur terrain profond reconstruit une tuile depuis les GeoTIFF MNT/MNS
référencés et compare les hashes sources, la grille, chaque altitude arrondie du
maillage et chaque pixel de la carte de contexte. Cette carte MNT/MNS ne vaut
pas mapping contextuel des sols : tant que les parcelles, transports,
hydrographie, occupation et géologie ne sont pas liés, la production générale
et les placements de routes ou bâtiments restent bloqués.

## Documentation de référence

- `docs/CAMERA_AND_CRS_CONTRACT.md` : repères, caméra, pose et contrôles ;
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
