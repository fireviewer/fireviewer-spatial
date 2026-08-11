# Contrat caméra et systèmes de coordonnées

## Repère de calcul

Les calculs utilisent un repère métrique local rattaché à une révision de zone.

WGS84 est une représentation dérivée, pas le repère principal du PnP ou du raycast.

## Références obligatoires

- CRS horizontal ;
- datum vertical ;
- origine locale ;
- unité ;
- ordre des axes ;
- révision canonique FVTQ et bornes verticales LOD2 ;
- contrat des sources MNT/MNS 2 m ;
- plan de résidence caméra ;
- empreintes ;
- révision.

## Modèle de caméra

Le contrat conserve :

- type de caméra ;
- intrinsics ;
- distorsion ;
- dimensions ;
- orientation image ;
- provenance des paramètres ;
- incertitude ou statut estimé.

## Pose

La pose est représentée dans le repère local avec :

- translation ;
- rotation ;
- modèle de caméra ;
- covariance locale si disponible ;
- rapport de contrôles.

## Contrôles

- cheirality ;
- reprojection ;
- altitude ;
- FOV ;
- horizon ;
- caméra au-dessus du terrain ;
- stabilité ;
- cohérence des profondeurs.

Le frustum utilise les six plans near/far inclus et les AABB 3D issues de
LOD2. Une caméra principale n'est publiée que lorsque toutes ses tuiles
visibles sont résidentes en LOD0. Les LOD1 et LOD2 sont interdits dans les
pixels terrain RGB ou thermiques ; l'AOV `fireviewer:terrain_lod` le contrôle.

## Conversion

Toute conversion vers WGS84 conserve la référence source, la méthode et la révision du terrain.
