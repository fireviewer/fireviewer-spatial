# Checklist de validation d'une carte V1

Cette checklist doit être reportée dans un reçu conforme à
`unity-validation-receipt.schema.json`. Tant que la décision n'est pas
`accepted`, l'étape `site_upload` reste bloquée.

## Contrat et sources

- [ ] `--plan` termine sans erreur.
- [ ] Tous les GeoJSON sont en EPSG:2154 et leur provenance/date est archivee.
- [ ] L'AOI correspond a la zone voulue et l'enveloppe centrale est correcte.
- [ ] Le nombre attendu de dalles IGN est verrouille apres revue du plan.
- [ ] Un nouvel `package_id` et un nouvel `artifact_root` sont utilises pour la revision.

## Rendu Blender/Unity

- [ ] Terrain et batiments touchent correctement le sol.
- [ ] Aucun arbre n'apparait dans les batiments, routes ou surfaces d'eau.
- [ ] Routes et cours d'eau suivent le relief sans rupture visible.
- [ ] FAR reste continu sur toute l'emprise.
- [ ] MID apparait avant que le fond ne devienne grossier.
- [ ] NEAR charge uniquement les tuiles visibles et montre la vegetation detaillee.
- [ ] Les batiments restent distincts du terrain en vue proche.
- [ ] La camera ne peut pas passer sous la carte.
- [ ] L'acces MID/NEAR et la camera FPS admin respectent l'authentification du site.

## Package

- [ ] `validate_catalog` est marque `complete` dans `.production/state.json`.
- [ ] `site-upload/<package_id>/package-manifest.json` existe.
- [ ] Le manifeste annonce le bon `zone_id`, la bonne revision et le bon nombre d'assets.
- [ ] Il ne reste aucun fichier `.part` dans le catalogue Unity.
- [ ] Le dossier `site-upload/<package_id>` seul est remis a l'administration.
