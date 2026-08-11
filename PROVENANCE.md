# Provenance

Ce dépôt contient uniquement le code, les contrats et de petites fixtures
synthétiques du pipeline spatial FireViewer.

La production active relie explicitement :

- la demande GPS et son plan de tuiles ;
- les révisions et requêtes des sources MNT, MNS et orthophoto ;
- les SHA-256 observés avant suppression des rasters temporaires ;
- le compilateur FVTG, la texture bakée et l'inventaire MNS−MNT ;
- le catalogue d'assets, les USD/textures réellement utilisés et leurs hashes ;
- la scène unifiée, les vingt captures et le package portable ;
- les observations de périmètre et le build exact de la carte de base.

Ce lot actif n'ajoute aucun modèle, dataset, asset 3D, carte, orthophoto,
archive de production, credential ou reçu réel. Les payloads et reçus
historiques déjà suivis restent intacts jusqu'à une décision de nettoyage
séparée. Les modèles et datasets distribués sur Hugging Face restent des
dépendances externes versionnées ; leur présence distante ne dispense jamais
de vérifier leur révision et leur hash au démarrage du producteur.

Les anciens pipelines Unity, terrain adaptatif/FVTQ-PBR et registres globaux de
packs ne définissent plus le chemin actif. Leur suppression physique reste une
opération de nettoyage séparée, bornée et revue avant exécution.
