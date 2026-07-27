# Sources a deposer ici

Tous les fichiers doivent etre des GeoJSON en **EPSG:2154 (Lambert-93)**,
decoupes au minimum a l'emprise de production :

- `aoi.l93.geojson` : polygone exact de la carte ;
- `production-envelope.l93.geojson` : meme polygone ou enveloppe de travail ;
- `buildings.l93.geojson` : empreintes BD TOPO, avec hauteurs si disponibles ;
- `vegetation.l93.geojson` : surfaces de vegetation BD TOPO ;
- `roads.l93.geojson` : axes routiers BD TOPO ;
- `water-courses.l93.geojson` : cours d'eau nommes ;
- `water-surfaces.l93.geojson` : surfaces d'eau.

Le kit telecharge le MNT/MNS LiDAR HD et les orthophotos IGN. Il ne telecharge
pas silencieusement les vecteurs : les exports BD TOPO locaux sont conserves
comme preuves de source et controles avant une production longue.

Ne laissez pas les fichiers d'exemple vides : le preflight refuse les couches
obligatoires absentes et refuse une hydrographie entierement vide.
