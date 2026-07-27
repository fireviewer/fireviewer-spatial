# FireViewer Spatial

Outils source pour préparer, valider et intégrer les packages spatiaux
FireViewer : kit France, scripts Blender, package Unity runtime et contrats de
catalogue.

Ce dépôt ne contient aucune carte, orthophoto, scène générée, archive de
production, dossier d’upload, modèle 3D, atlas, arbre source ou donnée
d’incident. Les packages validés sont publiés séparément dans GitHub Releases
ou un stockage Blob.

## Contenu

- `production-kit-france/` : préparation reproductible d’une zone générique ;
- `blender/` : génération et contrôles géométriques ;
- `unity/` : export, validation et runtime de streaming ;
- `runtime-package/` : package Unity réutilisable ;
- `contracts/spatial/` : schémas du package et du catalogue ;
- `tests/` : fixtures synthétiques sans données de production.

## Production

Chaque production utilise un dossier de travail externe et une configuration de
zone explicite. Les entrées LiDAR, orthophotos, vecteurs, modèles et rendus
restent hors du checkout. Les manifestes générés conservent provenance,
licences, tailles et SHA-256.

La validation automatisée ne remplace pas la revue Unity. Une publication exige
un reçu de validation Unity manuel conforme au schéma du kit.

## Contrôles

```powershell
python -m unittest discover -s tests -v
python -m pytest -q
python -m ruff check .
dotnet build unity/dotnet-bridge-probe/FireViewer.UnityBridge.Probe.csproj
```

Ces contrôles valident le code et les contrats. Ils ne prouvent pas une scène
Unity, un package cartographique ou une publication distante.

## Licences

Le code est sous AGPL-3.0-or-later et la documentation sous CC BY 4.0. Les
sources IGN, OpenStreetMap et autres données externes conservent leurs licences
propres et ne sont pas redistribuées dans ce dépôt.
