# Correspondance orthophoto vers surfaces PBR

## Rôle strict

`orthophoto_surface_correspondence.py` transforme temporairement une fenêtre
orthophoto RGB en cartes de sélection de textures. L'orthophoto sert uniquement
pendant la fabrication. Aucun pixel, chemin, extrait colorimétrique ou fichier
orthophoto n'est écrit dans le package terrain.

Le runtime ne reçoit que :

- `ground-profile-ids.png`, RGBA8, indices stables `0..71` ;
- `ground-profile-weights.png`, RGBA8, somme exactement égale à 255 ;
- `ground-confidence.png`, L8 ;
- `ground-orientation.png`, L8, angle non dirigé dans `[0, π[` ;
- `surface-correspondence.json`, manifeste compact et hash-locké.

Il n'existe ni macro-tint issu de l'image, ni bruit de shader, ni matériau
procédural. Les indices sélectionnent exclusivement la bibliothèque
`fireviewer.clean-pbr-texture-library.v1` acceptée : 72 profils dans l'ordre
`stable_index` du contrat v4, quatre sources propres basecolor/normal/height/ORM
par profil, projection `world_xy` ou `world_triplanar`.

## API

```python
result = compile_aligned_window(
    rgb_u8,
    transform=affine_north_up_1m,
    crs="EPSG:2154",
    core_bounds_l93_m=(west, south, east, north),
    orthophoto_sha256=source_file_sha256,
    pbr_library=accepted_clean_pbr_library,
    pbr_library_sha256=expected_library_sha256,  # optionnel, contrôlé si fourni
    correspondence_model=locked_model,
    model_sha256=expected_model_sha256,          # optionnel, contrôlé si fourni
    context_priors=surface_snapshot_priors,      # optionnel
    approved_corrections=corrections,            # optionnel
)

tile = slice_tile(result, (tile_west, tile_south, tile_east, tile_north))
payloads = serialize_tile_outputs(tile)
write_tile_outputs(tile, output_directory_on_d)
```

Le cœur peut contenir une tuile ou une bande rectangulaire de tuiles. Ses
limites sont des multiples de 500 m. La source est strictement nord-haut,
EPSG:2154, RGB8, 1 m par pixel et fournit 10 m de halo sur chaque bord.

## Déterminisme et raccords

Les moyennes RGB et signatures de texture sont calculées aux fenêtres 1, 5 et
17 m avec des images intégrales entières. La complexité est
`O(72 × 3 × pixels)` ; elle ne dépend pas de l'aire des fenêtres. Les scores,
tris, poids, confiance et orientation utilisent des opérations entières. Le
profil est toujours encodé par son `stable_index`, jamais par un tri de nom.

Chaque tuile hash-locke ses propres pixels d'entrée sur exactement
`500 m + 2 × 10 m`. Son identité ne dépend donc ni de la taille de la bande, ni
du worker, ni de l'ordre de compilation. Compiler deux tuiles séparément ou
compiler leur bande puis la découper donne les cinq mêmes fichiers octet pour
octet.

Les hashes du manifeste verrouillent :

- la source orthophoto et l'extrait tuile+halo, sans chemin source ;
- la bibliothèque PBR propre et son contrat v4 ;
- le modèle de correspondance ;
- l'algorithme et son contrat ;
- les priors contextuels et les corrections approuvées.

## Priors et corrections

Un prior fournit une géométrie EPSG:2154 et exactement une restriction :
`allowed_profile_ids` ou `allowed_class_ids`. Les lignes exigent `width_m`.
Les recouvrements sont résolus par priorité puis identifiant stable. Une
correction ne s'applique que si elle porte `approved: true` et un
`approval_sha256`; elle est toujours appliquée après les priors. Si une
restriction ne laisse aucun profil PBR compatible, la compilation échoue.

Le futur adaptateur GPKG fournit ces priors, mais ne remplace pas la
correspondance visuelle. Les ponts, tunnels, gués et autres croisements doivent
être matérialisés par des corrections explicitement approuvées.

## Mesure locale de référence

Sur la fixture CPU synthétique de six classes (champ, forêt, eau, roche, route,
chemin), une fenêtre 520×520 a été compilée en 3,697 s. Le RSS est passé de
73,8 Mio à un pic de 155,5 Mio, soit +81,7 Mio, et les quatre matrices brutes
de sortie occupent 2 500 000 octets. Cette mesure n'est pas une qualification
du futur modèle ni une acceptation visuelle des textures.

Après validation des cinq sorties et de leurs hashes, la source orthophoto de
travail et ses caches doivent être supprimés avant le cas suivant.
