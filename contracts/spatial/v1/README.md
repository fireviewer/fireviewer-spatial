# Contrat spatial local v1

Ces artefacts décrivent un profil spatial **fictif**, local et versionné pour Fire Viewer.
Ils ne contiennent aucun terrain, incident, asset GLB ou aperçu PNG réel.

| Artefact | Rôle |
| --- | --- |
| `spatial-contract.schema.json` | schéma JSON Draft 2020-12 des fixtures de contrat |
| `fixtures/enu-unity-points.json` | vecteurs de contrôle ENU → glTF → Unity et round-trip |
| `fixtures/zone-registry.json` | registre de zones réutilisables, versionnées et limitées à l'emprise déclarée |
| `fixtures/zone-revision.json` | nouvelle révision d'une zone, sans mutation de la révision précédente |
| `fixtures/spatial-snapshot.json` | snapshot immuable lié à une révision de manifeste d'incident et à son archive PNG |

## Conventions verrouillées

- Le périmètre est la **France continentale rurale locale**. Corse et outre-mer sont hors
  périmètre de ce profil ; ils ne doivent pas être traités comme des cas RAF20/NGF-IGN69.
- Une origine est WGS 84 3D / `EPSG:4979`. Le tableau `origin_wgs84` est toujours
  `[longitude_deg, latitude_deg, ellipsoidal_height_m]`.
- La source verticale est `NGF-IGN69`, convertie localement par RAF20. La grille est
  épinglée à son URI et SHA-256 officiels ; les transformations ne téléchargent rien.
- ENU est physique et métrique. Le GLB stocke `(E, U, -N)` en mètres
  (`gltf_meters_per_unit = 1.0`). Unity représente `(100E, 100U, 100N)` ; son manifeste
  expose donc `meters_per_unit = 0.01`.
- Un PNG appartient à l'archive d'un snapshot de révision de manifeste, pas à toutes les
  zones actives. Il est identifié par URI, hash, dimensions et date de production.
- Aucun globe, jeu de tuiles ou runtime Cesium ne fait partie de ce profil.

L'[ADR-002](../../../docs/adr/ADR-002-spatial-local-unity-contract.md) est la décision
normative ; les fixtures sont des valeurs de contrôle, pas une source de géodonnées.
