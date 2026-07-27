# Kit de production d'une carte FireViewer en France

Ce dossier reproduit la chaine de production validee sur la carte
zone synthétique V1. Il ne fabrique pas un GLB monolithique : il produit le
fond 2,5D et les tuiles de detail chargees par Unity, puis le dossier final
attendu par l'upload du site.

Le profil [`profiles/unity-v1-accepted.json`](profiles/unity-v1-accepted.json)
est volontairement verrouille :

- MNT/MNS LiDAR HD a 0,5 m ;
- tuiles de detail de 500 m avec halo de calcul de 10 m ;
- orthophoto proche a 20 cm, moyenne a 50 cm ;
- fond global MNT a 5 m et orthophoto a 2 m ;
- correction d'affichage acceptee : luminosite `0.78`, contraste `1.08`,
  saturation `1.12` ;
- catalogue Unity distant, traitements par lots de quatre, un seul processus ;
- maximum de 16 tuiles detaillees residentes, publication a 600 m et
  prechargement a 750 m.

## Separation avec l'archive Zone synthétique

Le kit n'embarque ni l'archive zone synthétique, ni ses rasters, ni ses
vecteurs, ni son paquet Unity. L'archive de conservation reste locale sous
`.artifacts/archives`, dossier ignore par Git, et n'est pas une dependance de
la production d'une nouvelle zone. Seuls le profil qualite, les contrats, les
scripts et de petits fixtures de test sont versionnes dans ce dossier.

Le resultat final contient le terrain, chaque arbre detecte par MNS-MNT, les
batiments, les routes et l'hydrographie. Les arbres sont exclus des empreintes
de batiments, des routes et des surfaces/cours d'eau lors de la production des
tuiles.

## Contenu

- `zone.schema.json` : contrat d'une nouvelle zone ;
- `zone.example.json` : configuration a copier et adapter ;
- `profiles/unity-v1-accepted.json` : qualite V1 immuable ;
- `run_production.py` : preflight, execution, reprise et export final ;
- `build_far_rasters.py` : mosaïques MNT/MNS FAR en COG ;
- `build_archive_manifest.py` : inventaire SHA-256 d'une archive locale separee ;
- `Invoke-Production.ps1` : lanceur PowerShell ;
- `CHECKLIST_RELEASE.md` : controle humain avant remise a l'administration.
- `unity-validation-receipt.schema.json` : preuve manuelle obligatoire avant
  la creation du dossier d'upload.

## Prerequis

Depuis la racine du depot :

```powershell
python -m pip install -r blender/requirements.txt
```

Blender 4.5 est requis uniquement si `build_blender_scene` vaut `true`. L'export
du catalogue pour le site n'exige pas que la fenetre Blender reste ouverte.
Prevoir au moins 20 Gio libres ; une emprise proche de 200 km2 demande beaucoup
plus pendant le cache des sources LiDAR et des orthophotos.

Le kit cible la France metropolitaine et la Corse en Lambert-93 (`EPSG:2154`).
Les territoires ultramarins emploient d'autres systemes de coordonnees et ne
sont pas acceptes silencieusement par ce contrat.

## 1. Preparer les sources de zone

Creer un dossier de travail a partir de
[`examples/zone-template/sources`](examples/zone-template/sources/README.md).
Les vecteurs doivent etre des exports locaux et tracables, en EPSG:2154 :

- AOI exacte de la carte ;
- enveloppe centrale de production ;
- batiments et vegetation BD TOPO ;
- routes BD TOPO ;
- cours, segments et/ou surfaces d'eau BD TOPO.

L'AOI doit normalement etre l'enveloppe centrale augmentee de 1 500 m. Le
fichier `production-envelope` sert uniquement a cadrer la production ; il
n'ajoute aucun perimetre d'incendie au catalogue Unity.

Une famille de donnees reellement vide reste representee par un
`FeatureCollection` vide. Elle ne doit pas etre supprimee de la configuration :
cela permet de distinguer « aucune entite trouvee » de « source oubliee ».

Le MNT/MNS LiDAR HD et les orthophotos IGN sont telecharges par la chaine. Les
vecteurs ne le sont pas automatiquement : leur pagination, leur date d'export
et leur provenance doivent rester controlees par l'operateur.

## 2. Configurer la zone

Copier `zone.example.json`, puis modifier :

- `zone_id`, `revision`, `package_id`, `artifact_slug` et `label` ;
- `origin_l93_m` : XY dans l'AOI et Z proche de l'altitude moyenne ;
- tous les chemins `inputs` ;
- au moins une `attention_zone`, utilisee seulement pour prioriser l'export ;
- `artifact_root`, qui doit etre propre a cette revision ;
- les chemins futurs `unity_validation_receipt` et `unity_preview_png` ; ils
  peuvent etre absents pendant la production mais deviennent obligatoires pour
  l'etape `site_upload` ;
- le chemin de Blender, ou `build_blender_scene: false` pour sauter le `.blend` ;
- `near_lod_enabled: false` pour une grande zone : aucun téléchargement 0,2 m,
  imagerie de détail plafonnée à 0,5 m et runtime Unity contraint au LOD `mid`
  même à courte distance.

Ne reutilisez pas le meme `artifact_root` apres avoir change la configuration :
le hash de contrat bloque volontairement le melange de deux productions.

## 3. Faire le preflight sans ecriture ni reseau

```powershell
python production-kit-france/run_production.py `
  --config C:/cartes/ma-zone/zone.json `
  --plan
```

Le JSON affiche les sources lues, l'emprise, les dix etapes, leurs commandes et
le dossier final. Le preflight refuse notamment :

- des coordonnees qui ressemblent a des longitudes/latitudes ;
- une emprise vide ou superieure a 250 km2 avec le profil V1 ;
- un fichier obligatoire absent ;
- l'absence totale de source routiere ou hydrographique ;
- une zone d'attention hors AOI ;
- une modification d'une valeur qualite acceptee.

Pour verrouiller aussi le nombre de dalles IGN 1 km, lancer le producteur en
`--dry-run`, relever `summary.source_tile_count`, puis reporter la valeur dans
`expected_source_tile_count` avant la premiere execution :

```powershell
python blender/prepare_global_05m.py `
  --aoi C:/cartes/ma-zone/sources/aoi.l93.geojson `
  --output-root C:/cartes/ma-zone/dry-run `
  --dry-run
```

## 4. Produire et reprendre

```powershell
python production-kit-france/run_production.py `
  --config C:/cartes/ma-zone/zone.json `
  --execute
```

La meme commande reprend une execution interrompue. Les sources, receipts et
sorties immuables deja valides sont conserves. L'etat est enregistre dans
`<artifact_root>/.production/state.json`.

Une seule etape peut aussi etre lancee pour le diagnostic :

```powershell
python production-kit-france/run_production.py `
  --config C:/cartes/ma-zone/zone.json `
  --execute `
  --stage unity_catalog
```

Ses dependances doivent deja etre valides. L'ordre complet est :

1. plan des dalles 0,5 m ;
2. MNT/MNS, MID et tuiles 0,5 m ;
3. orthophotos NEAR 20 cm ;
4. MNT/MNS FAR 5 m ;
5. orthophoto FAR 2 m ;
6. package vectoriel global ;
7. scene de controle Blender facultative ;
8. catalogue Unity distant ;
9. validation exhaustive du catalogue ;
10. dossier exact d'upload du site, uniquement après reçu Unity accepté.

## Gate manuel Unity

Après `validate_catalog`, servir le catalogue local et l'ouvrir avec le projet
Unity du dépôt en version `6000.3.18f1`. Effectuer la checklist de release,
capturer une vue PNG représentative puis créer un reçu conforme à
`unity-validation-receipt.schema.json`. Le reçu doit lier le `package_id`, la
zone, la révision, le SHA-256 du catalogue validé et celui de la capture.

L'étape `site_upload` refuse un reçu absent, rejeté ou périmé. Elle exige aussi
la déclaration explicite `ACCEPTÉ POUR PUBLICATION`. Un refus impose une
nouvelle racine d'artefacts et un nouveau `package_id`; une V1 existante n'est
jamais écrasée.

## 5. Resultat

La derniere ligne indique `upload_root`. Ce dossier contient uniquement le
contrat publiable :

```text
site-upload/<package_id>/
|-- package-manifest.json
|-- catalog.json
`-- assets/
    |-- far/
    |-- detail/
    |-- imagery/
    `-- validation/unity-preview.png
```

Les fichiers lourds de travail, le `.blend`, les GeoTIFF et le fichier source
du reçu Unity restent hors du dossier d'upload. Sa forme normalisée est incluse
dans le manifeste et la capture est incluse dans le catalogue. Le kit ne
declenche aucun upload :
l'administration du site recoit exactement le dossier `site-upload/<package_id>`
apres la checklist de release.

## Verification du kit

```powershell
python -m pytest -q production-kit-france
ruff check production-kit-france
```

La production d'une nouvelle zone reste une operation longue et depend de la
couverture effective des produits IGN. Un preflight valide prouve le contrat et
les sources locales, pas encore la disponibilite distante de chaque dalle ;
celle-ci est verifiee pendant les etapes de telechargement.
