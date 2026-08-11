# Contrat caméra et systèmes de coordonnées

## Référentiels

- demande utilisateur : `EPSG:4326`, latitude et longitude explicites ;
- production : `EPSG:2154`, grille métrique de tuiles de 500 m ;
- altitude : `NGF-IGN69` ;
- OpenUSD : mètres, axe vertical Z ;
- texture bakée : alignée sur le coeur de la tuile et sa provenance source.

Toute conversion conserve le CRS source, le CRS cible, l'ordre des axes, la
méthode, l'emprise, la résolution et la révision de package.

## Carte et placements

Le relief FVTG est compilé depuis le MNT. Les hauteurs et positions automatiques
d'objets proviennent du couple MNS−MNT puis sont confirmées par le contexte
disponible. Un placement GPS explicite est transformé en Lambert-93, affecté à
une seule tuile propriétaire et posé sur le MNT.

Une scène unifiée translate chaque tuile par rapport à l'origine de zone. Les
instances conservent une altitude monde cohérente avec le terrain ; aucune
double application du datum vertical n'est admise.

## Captures de contrôle

Le reçu de galerie verrouille exactement vingt captures : quatre vues générales
puis seize détails. Chaque caméra conserve sa pose, son cadrage, sa résolution,
le package observé et le SHA-256 de l'image. Ces captures sont une preuve de
contrôle ; elles ne remplacent pas `zone.usda`.

## Périmètres et viewer

Les observations WGS84 ou Lambert-93 sont normalisées vers `EPSG:2154` avant la
création du calque. Les vues GLB sont dérivées du même package et superposées à
la carte validée. Leur caméra ne modifie ni la géométrie autoritative ni la
timeline.

## Contrôles obligatoires

- emprise et origine de tuile exactes ;
- altitude terrain/instance cohérente ;
- références relatives confinées au package ;
- absence de chemin machine dans les livrables ;
- parité entre carte, timeline et build référencé ;
- ouverture indépendante du ZIP extrait ;
- vingt captures présentes, lisibles et hashées.
