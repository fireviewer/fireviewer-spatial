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
archive de production ou credential. Le dépôt peut conserver de petits reçus
de provenance sans embarquer leur ressource lourde. Les modèles et datasets
distribués sur Hugging Face restent des
dépendances externes versionnées ; leur présence distante ne dispense jamais
de vérifier leur révision et leur hash au démarrage du producteur.

Les anciens pipelines Unity, terrain adaptatif, atlas de sol et registres
globaux de packs ont été retirés. Leur historique Git ne définit aucun contrat
actif et leurs anciens outputs locaux ne doivent pas être recréés.
