# FireViewer Spatial Tiles for Unity

Ce package UPM consomme directement le contrat produit par
`export_remote_catalog.py`. Il ne dépend ni de prefabs préparés à l'avance ni
d'un catalogue Addressables : `catalog.json`, `.fwterrain`, `.fwtile` et les
orthophotos immuables sont les ressources distantes de production.

## Installation locale

Dans `Packages/manifest.json` du projet Unity :

```json
{
  "dependencies": {
    "com.fireviewer.spatial-tiles": "file:./unity/runtime-package/com.fireviewer.spatial-tiles"
  }
}
```

Le chargement vérifie le nombre d'octets et le SHA-256 du fichier, puis les
SHA-256 stocké et brut de chaque section avant de créer un objet Unity.

## Scène prête à ouvrir

Le menu `FireViewer > Create or Replace Spatial Demo Scene` crée :

`Assets/FireViewerSpatial/FireViewerSpatialDemo.unity`

avec caméra, lumière, repère ENU, cible, racine de contenu et contrôleur. La
commande équivalente est :

```powershell
Unity.exe -batchmode -quit -projectPath C:\projet\unity `
  -executeMethod FireViewer.SpatialTiles.Editor.FwSpatialDemoSceneBuilder.CreateFromCommandLine `
  -fireviewerCatalogUrl https://cdn.example/fireviewer/catalog.json
```

La scène démarre sur Montmaur à environ 1 414 m de la cible, donc en bande
`mid`. L'inspecteur du bootstrap expose les boutons Montmaur, Barsac et Ausson.
Les mêmes actions sont accessibles par `FocusMontmaur()`, `FocusBarsac()`,
`FocusAusson()` ou `FocusZone(id)`.

## LOD et atomicité

- `far`, distance de vue strictement supérieure à 3 000 m : MNT et imagerie
  globaux seulement ;
- `mid`, distance supérieure à 750 m et inférieure ou égale à 3 000 m :
  voisinage de tuiles 0,5 m complet ;
- `near`, distance inférieure ou égale à 750 m : même détail complet et
  contrôlable.

Le contexte global reste actif dans les trois bandes, mais le shader FAR est
découpé exactement sous les rectangles détaillés publiés afin d'empêcher le
MNT 5 m de traverser le terrain, les routes, l'eau ou la végétation. Les tuiles détaillées
sont téléchargées une par une, restent masquées pendant la préparation et ne
sont publiées qu'une fois tout le voisinage résident. Un échec ou un besoin
supérieur au budget global de 16 conserve uniquement le contexte global. Le
Le runtime utilise 1 unité Unity par mètre. Les arbres sont ancrés sur la grille
DETAIL rendue ; les candidats situés sur une chaussée ou une surface d'eau ne
sont pas affichés. Les bâtiments sont conservés. Plusieurs profils feuillus et
conifères instanciés sont utilisés à courte distance, avec un profil plus léger
à moyenne distance.

## Preuves locales

```powershell
python -m pytest unity -q
python unity/run_unity_bridge_smoke.py
```

Le smoke lance Unity 6000.3.18f1 en batch, récupère réellement le catalogue,
le `.fwtile` et son image sur HTTP local, puis exige les GameObjects terrain,
arbres et vecteurs attendus.
