"""Les cinq techniques de prompting comparées.

Chaque technique produit le même type de sortie (un résumé en français d'environ
N mots) ; seule la manière de le demander change. Les exemples du few-shot sont
volontairement *hors* du corpus d'évaluation pour éviter toute fuite.
"""
from __future__ import annotations

from dataclasses import dataclass

COMMON_RULES = (
    "Écris uniquement le résumé, en français, en un seul paragraphe, sans titre, "
    "sans liste, sans introduction du type « Voici le résumé ». "
    "N'ajoute aucune information absente du texte."
)

EXAMPLES = [
    (
        "Le protocole DHCP attribue automatiquement une adresse IP aux équipements d'un réseau. "
        "Le client diffuse une demande, le serveur propose une adresse, le client la sollicite "
        "et le serveur confirme : ce sont les quatre étapes DORA. L'adresse est louée pour une "
        "durée limitée, renouvelée avant expiration. Un relais DHCP permet de servir plusieurs "
        "sous-réseaux avec un seul serveur, et des réservations associent une adresse fixe à une "
        "adresse MAC pour les imprimantes ou serveurs.",
        "DHCP attribue automatiquement les adresses IP en quatre étapes (DORA) : demande, "
        "proposition, sollicitation et confirmation. Les adresses sont louées pour une durée "
        "limitée puis renouvelées ; un relais permet de servir plusieurs sous-réseaux et des "
        "réservations fixent l'adresse de certains équipements selon leur adresse MAC.",
    ),
    (
        "Docker exécute des applications dans des conteneurs, des environnements isolés qui "
        "partagent le noyau du système hôte. Contrairement aux machines virtuelles, ils "
        "démarrent en quelques secondes et consomment peu de ressources. Une image décrit le "
        "contenu du conteneur à partir d'un Dockerfile ; elle est construite en couches "
        "réutilisables et publiée dans un registre. Docker Compose orchestre plusieurs "
        "conteneurs, par exemple une application et sa base de données.",
        "Docker isole les applications dans des conteneurs qui partagent le noyau de l'hôte, "
        "plus légers et rapides que des machines virtuelles. Les images, décrites par un "
        "Dockerfile et construites en couches, sont publiées dans un registre, et Docker "
        "Compose orchestre plusieurs conteneurs ensemble.",
    ),
]


@dataclass(frozen=True)
class Technique:
    key: str
    label: str
    description: str

    def system(self) -> str:
        if self.key == "role":
            return ("Tu es un ingénieur rédacteur technique senior. Tu rédiges des synthèses "
                    "exactes, denses et fidèles pour des ingénieurs pressés. " + COMMON_RULES)
        return "Tu résumes des documents techniques. " + COMMON_RULES

    def prompt(self, text: str, n_words: int) -> str:
        target = f"Résume le texte suivant en {n_words} mots environ."
        if self.key in ("zero_shot", "role"):
            return f"{target}\n\nTexte :\n{text}"
        if self.key in ("one_shot", "few_shot"):
            shots = EXAMPLES[:1] if self.key == "one_shot" else EXAMPLES
            demo = "\n\n".join(f"Texte :\n{src}\nRésumé :\n{dst}" for src, dst in shots)
            return f"Exemples de résumés attendus :\n\n{demo}\n\n{target}\n\nTexte :\n{text}\nRésumé :"
        if self.key == "chain_of_thought":
            return (
                f"{target}\n"
                "Procède ainsi, sans montrer ton raisonnement : (1) repère l'idée principale, "
                "(2) relève les 3 à 5 faits techniques essentiels (noms, chiffres, mécanismes), "
                "(3) rédige un paragraphe qui les relie. "
                "Réponds uniquement avec le paragraphe final.\n\n"
                f"Texte :\n{text}"
            )
        raise KeyError(self.key)


TECHNIQUES: dict[str, Technique] = {t.key: t for t in [
    Technique("zero_shot", "Zero-shot", "Consigne seule"),
    Technique("one_shot", "One-shot", "Un exemple de résumé"),
    Technique("few_shot", "Few-shot", "Deux exemples de résumés"),
    Technique("chain_of_thought", "Chain of thought", "Étapes de réflexion guidées, sortie finale seule"),
    Technique("role", "Role prompting", "Persona d'ingénieur rédacteur"),
]}
