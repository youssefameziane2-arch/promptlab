# Résultats du banc

Modèle : `replay:ollama:qwen2.5:3b` · 6 documents · longueurs [30, 60] mots · graines [0] · tolérance de longueur ±15 %

Moyenne et intervalle de confiance à 95 % (bootstrap).

| Système | Faits clés | ROUGE-L | Fidélité (support) | Respect longueur | Vérifs OK | Nombres inventés | Tentatives | Temps (s) |
|---|---|---|---|---|---|---|---|---|
| `baseline:centrality` | 28 % [21–35] | 20 % [17–22] | 100 % [100–100] | 89 % [84–94] | 83 % | 0.00 | 1.00 | 0.0 |
| `baseline:lead` | 22 % [15–30] | 26 % [22–29] | 100 % [100–100] | 81 % [73–88] | 33 % | 0.00 | 1.00 | 0.0 |
| `chain_of_thought` | 46 % [34–57] | 34 % [28–40] | 86 % [81–91] | 88 % [81–94] | 67 % | 0.00 | 1.00 | 61.0 |
| `chain_of_thought+repair` | 53 % [40–67] | 36 % [30–41] | 88 % [84–92] | 92 % [87–96] | 83 % | 0.00 | 2.25 | 113.6 |
| `few_shot` | 52 % [40–66] | 35 % [31–40] | 89 % [86–92] | 93 % [88–97] | 83 % | 0.00 | 1.00 | 64.3 |
| `few_shot+repair` | 52 % [40–66] | 35 % [31–40] | 89 % [86–92] | 93 % [88–97] | 83 % | 0.00 | 2.33 | 120.5 |
| `one_shot` | 53 % [43–64] | 36 % [32–41] | 89 % [86–93] | 94 % [90–98] | 92 % | 0.00 | 1.00 | 67.5 |
| `one_shot+repair` | 57 % [46–67] | 36 % [32–41] | 90 % [87–93] | 93 % [88–97] | 92 % | 0.00 | 2.17 | 122.4 |
| `role` | 58 % [47–69] | 36 % [31–42] | 88 % [84–91] | 94 % [91–97] | 92 % | 0.00 | 1.00 | 63.1 |
| `role+repair` | 58 % [47–69] | 36 % [31–42] | 88 % [84–91] | 94 % [91–97] | 92 % | 0.00 | 2.00 | 109.8 |
| `zero_shot` | 49 % [38–62] | 33 % [28–38] | 89 % [84–94] | 90 % [86–95] | 83 % | 0.00 | 1.00 | 61.0 |
| `zero_shot+repair` | 51 % [39–65] | 33 % [27–39] | 89 % [84–94] | 88 % [85–92] | 83 % | 0.00 | 1.92 | 104.9 |

## Comparaisons appariées

| A | B | Métrique | Écart A − B | IC 95 % | Significatif |
|---|---|---|---|---|---|
| `one_shot+repair` | `zero_shot+repair` | key_facts | +5.4 pts | [-4.0 ; +14.6] | non |
| `one_shot+repair` | `zero_shot+repair` | rouge_l | +3.5 pts | [+0.3 ; +6.4] | oui |
| `few_shot+repair` | `zero_shot+repair` | key_facts | +1.2 pts | [-8.3 ; +10.7] | non |
| `few_shot+repair` | `zero_shot+repair` | rouge_l | +2.3 pts | [-1.0 ; +6.2] | non |
| `chain_of_thought+repair` | `zero_shot+repair` | key_facts | +1.6 pts | [-13.5 ; +17.3] | non |
| `chain_of_thought+repair` | `zero_shot+repair` | rouge_l | +3.2 pts | [-2.1 ; +9.0] | non |
| `role+repair` | `zero_shot+repair` | key_facts | +6.8 pts | [-2.1 ; +17.9] | non |
| `role+repair` | `zero_shot+repair` | rouge_l | +3.7 pts | [+1.0 ; +6.4] | oui |
| `zero_shot+repair` | `zero_shot` | length_accuracy | -1.9 pts | [-4.7 ; +0.0] | non |
| `one_shot+repair` | `one_shot` | length_accuracy | -1.3 pts | [-3.8 ; +0.6] | non |
| `few_shot+repair` | `few_shot` | length_accuracy | +0.0 pts | [+0.0 ; +0.0] | non |
| `chain_of_thought+repair` | `chain_of_thought` | length_accuracy | +3.9 pts | [-0.3 ; +9.9] | non |
| `role+repair` | `role` | length_accuracy | +0.0 pts | [+0.0 ; +0.0] | non |

*Faits clés* : part des faits essentiels du texte présents dans le résumé. *Fidélité* : part des mots porteurs de sens du résumé présents dans la source. *Vérifs OK* : longueur, format et absence de nombre inventé respectés.
