# Contribuer à Fire Viewer

## Règle principale

Avant G2, les contributions utilisent exclusivement des données fictives ou des données publiques dont la provenance, la licence et la précision sont documentées. Ne soumettez pas de preuve réelle, de position sensible, de token ou d'asset sans métadonnées.

## Premier périmètre de contribution

Le backlog initial est dans [docs/PLAN_DE_SUITE.md](docs/PLAN_DE_SUITE.md). Les contributions prioritaires portent sur :

1. le test de contrat entre le manifeste backend et l'UI ;
2. le contrat WGS84 / ENU / Unity et les tests métriques ;
3. le vertical slice fictif `FR-83-00042` ;
4. les fallbacks texte, WebGL indisponible et erreurs réseau.

## Licence des contributions

Tout code soumis à ce dépôt est proposé sous **AGPL-3.0-or-later**. Toute documentation, roadmap, illustration ou diagramme soumis est proposé sous **CC BY 4.0**. En ouvrant une contribution, vous confirmez disposer des droits nécessaires pour accorder cette licence et conserver les attributions requises.

## Avant une pull request

- Décrivez le comportement et la preuve de test, pas seulement les fichiers modifiés.
- Gardez les changements concentrés : n'introduisez ni données réelles, ni dépendance non justifiée, ni asset généré non traçable.
- Exécutez les contrôles ciblés disponibles et indiquez explicitement ceux qui n'ont pas été exécutés.
- Vérifiez que `git status` ne contient pas de `.env`, base locale, build, cache ou archive ZIP reçue.
- Exécutez un détecteur de secrets sur le diff et l'historique avant une publication publique.
- Utilisez une adresse Git de type `noreply` si votre adresse personnelle ne doit pas apparaître dans l'historique public.
- Inscrivez chaque image, document binaire ou média dans [ASSET_PROVENANCE.md](ASSET_PROVENANCE.md) avec sa licence et sa preuve d'origine.

## Convention de sécurité

Le DOM texte, l'horodatage, l'incertitude et le mode dégradé sont des exigences de produit. Une contribution qui améliore le rendu 3D mais retire ces garanties n'est pas acceptable.
