# General Knowledge — Plan final (synthèse v5 → v10)

> Équipe 4neurons. Modèle de base imposé : `Qwen/Qwen3-1.7B`.
> Ce document retrace tout le parcours, les diagnostics, et la décision finale.
> La **conclusion** (section 8) sera complétée avec le verdict CI 16k (V9b vs V10).

## 1. Le problème central

Optimiser le pass@1 d'un QCM (2-20 options) à partir d'un 1.7B, sans tomber dans :
- **Goodhart** : sur-optimiser un dev set in-distribution.
- **Format collapse** : raisonnement trop court → mauvaise généralisation.
- **Fuite de données** : dev set memorisé par le pré-entraînement (faux signal).

Contrat de sortie de la CI : réponse dans `\boxed{LETTER}`, `enable_thinking` **non**
passé en kwarg → doit être **baked dans le chat template**. Éval n=8, cap 1800s,
4k tokens (→ 16k à partir du 31 mai).

## 2. Le parcours, version courte

| Version | Idée | Données / réglages | Résultat | Leçon |
|---|---|---|---|---|
| v1 | SFT CoTs synthétiques | LoRA léger | Goodhart, CI baisse | dev in-distribution = piège |
| v5 | Strict CoT + distill 14B (non-think, court) | r?, CoTs ~150 tok | format collapse persistant | CoTs trop courts |
| v6 | Re-distill chirurgicale, CoTs longs 14B | r=64 α=128, "justify-given-answer" | **dev_full 0.541** | teacher 14B concis, label parfait |
| v7 | Distill 14B **thinking** (aveugle) | r=64 | régression | imitation de surface |
| v8 | Self-distill 1.7B (aveugle) + LoRA faible | r=8 lr=5e-5 | dev_small 0.39 | **format cassé** (51% boxed) |
| v9 | = données v8, **LoRA fort** | r=64 α=128 | dev_small 0.60, **86% boxed** | format = LoRA, +21 pts |
| v9b | v9 + `select_best` (anti-boucle) | r=64, trace propre la plus courte | dev_full 0.609, 94.5% boxed | base la plus solide |
| v10 | v6 + prompt **contrastif** 14B sur STEM | r=64, caches last-wins | dev_full 0.554 | off-policy < on-policy |
| v11 | DPO on-policy sur v9b (β=0.1, lr=5e-6, 1 ép.) | r=16 | dev_full 0.610 (= v9b) | DPO timide = no-op |
| **v11b** | DPO on-policy **fort** sur v9b (β=0.05, lr=1e-5, 3 ép.) | r=16, best-ckpt sur eval_loss | **pass@1 n=8 = 0.600** (vs v9b 0.580) | **modèle final** : +2.0 pts sur 4/4 sources, pass@8 intact |

## 3. Les diagnostics clés (ce qu'on a *prouvé*, pas supposé)

### 3.1 L'échec v8 = un artefact de format, pas de raisonnement
Re-scoring des complétions v8 avec l'extracteur **officiel de la CI**
(`evaluate/benchmarks.py`) : sans `\boxed{}`, le scorer cherche une lettre dans
tout le texte et ne la garde que s'il y en a **exactement une** → dans un long CoT,
plusieurs lettres → `None` → faux. Seulement **51% des sorties v8 avaient `\boxed{}`**.
Score réel (intention) = 0.52 vs 0.39 mesuré. **+13 pts cachés par le format.**

### 3.2 Le format est piloté par la force du LoRA
v8 → v9 : même dataset, LoRA r=8 → r=64. `\boxed{}` 51% → 86%, pass@1 0.39 → 0.60.
**+21 pts uniquement en renforçant l'apprentissage.**

### 3.3 Le 1.7B s'auto-distille en bouclant
`train_v8` : **59.3% des traces** ont des répétitions pathologiques (médiane 3521
chars), vs **1.6%** pour le 14B de v6. Les 31 ratés de v9 sur dev_small étaient
**100% des boucles** (médiane 20 répétitions d'un 4-gram, jamais de `\boxed{}`).

### 3.4 La correction (v9b) : `select_best`
À l'assemblage, parmi les 4 échantillons corrects par question, on garde la trace
**propre la plus courte** ; si les 4 bouclent, fallback sur le CoT v6. Résultat :
`\boxed{}` 94.5%, longueur ÷1.5, et **dev_full 0.609**.

### 3.5 LE résultat majeur : on-policy > off-policy
Sur dev_full (1000 questions, ±0.031), classement **fiable** :

| | global | boolq | mmlu | mmlu_pro | csqa |
|---|---:|---:|---:|---:|---:|
| v6 (14B distill) | 0.541 | 0.547 | 0.532 | 0.520 | 0.600 |
| **v9b (self-distill 1.7B propre)** | **0.609** | **0.567** | **0.647** | **0.537** | **0.693** |
| v10 (14B contrastif) | 0.554 | 0.513 | 0.585 | 0.517 | 0.587 |

**v9b bat v6 et v10 sur TOUTES les sources.** Le 1.7B apprend bien mieux de son
**propre raisonnement nettoyé** (on-policy) que de celui d'un 14B qu'il ne peut pas
reproduire (off-policy). Le contrastif 14B (v10) ajoute du raisonnement "trop fort
pour l'élève" → transfert plus faible. Résultat connu en distillation, démontré ici
empiriquement.

### 3.6 v11 : DPO on-policy → gain neutre (résultat négatif documenté)
On a échantillonné 8 complétions de v9b sur 4000 questions, gardé les paires
(correct vs incorrect, même format `\boxed{}`) → 2201 paires, puis DPO LoRA
(β=0.1, lr=5e-6, 1 époque). **dev_full v11 = 0.610 vs v9b 0.609** : strictement neutre.

Avec β=0.1, lr=5e-6, 1 époque : `train_loss` reste collée à **ln(2) ≈ 0.693**,
`rewards/accuracies ≈ 0.50`, `rewards/margins ≈ 0` → **no-op** (dev_full 0.610 = v9b).
Cause **structurelle** : `chosen` et `rejected` viennent du même modèle, même format,
longueurs voisines — ils ne diffèrent que par la lettre finale → signal préférentiel
trop faible pour un β/lr prudents.

### 3.7 v11b : DPO **fort** → +3.1 pts, nouveau meilleur modèle
En relançant avec β=0.05, lr=1e-5, **3 époques** (sur les mêmes 2201 paires), le signal
décolle : `rewards/margins` → **~0.13**, `rewards/accuracies` train → 0.85. L'éval
sature (eval_acc ~0.56-0.61, eval_loss min à l'époque 2.29) → `load_best_model_at_end`
garde le checkpoint le moins sur-appris. Résultat : **dev_full 0.640 vs 0.609**, gain
**réparti sur toutes les sources** (boolq +7.3, mmlu +3.1, mmlu_pro +3.0, csqa stable).

**Correction n=8 (important)** : le 0.640 était un **unique tirage n=1 chanceux**. Il a
fallu mesurer les deux modèles en **n=8** pour une comparaison équitable (même piège que
dev_small vs dev_full en §4 : jamais juger sur un seul tirage).

**Comparaison équitable v9b vs v11b, dev_full n=8 :**

| | pass@1 (n=8) | pass@8 | boolq | csqa | mmlu | mmlu_pro |
|---|---:|---:|---:|---:|---:|---:|
| v9b | 0.580 | **0.892** | 0.574 | 0.644 | 0.603 | 0.521 |
| **v11b** | **0.600** | 0.883 | 0.584 | 0.663 | 0.619 | 0.550 |
| Δ | **+2.0** | −0.9 | +1.0 | +1.9 | +1.6 | +2.9 |

**v11b gagne réellement** : +2.0 pts de pass@1, **gain sur les 4/4 sources** (donc robuste,
pas du bruit). Coût sur pass@8 = −0.9 (dans le bruit). Mécanisme : le DPO **resserre la
distribution** → plus de consistance au pass@1 contre une micro-perte de diversité. Comme
la **CI score le pass@1**, c'est le bon arbitrage.

**Leçon majeure** : le DPO on-policy fonctionne quand on lui donne assez de signal
(lr/β/époques), et le levier était bien **préférentiel** (apprendre à v9b à préférer ses
propres draws corrects), pas un manque de connaissance.

### Tableau complet dev_full (n=1000)

| | global | boolq | mmlu | mmlu_pro | csqa |
|---|---:|---:|---:|---:|---:|
| v6 | 0.541 | 0.547 | 0.532 | 0.520 | 0.600 |
| v9b | 0.609 | 0.567 | 0.647 | 0.537 | **0.693** |
| v10 | 0.554 | 0.513 | 0.585 | 0.517 | 0.587 |
| v11 (DPO timide) | 0.610 | 0.593 | 0.655 | 0.530 | 0.667 |
| v11b (DPO fort, n=1) | 0.640 | 0.640 | 0.678 | 0.567 | 0.687 |

> Lignes ci-dessus = tirages **n=1** (bruit ±~3pp), utiles pour les tendances mais pas
> pour départager des modèles proches. Le verdict final v9b vs v11b est fait en **n=8**
> (voir §3.7) : v11b 0.600 > v9b 0.580 en pass@1, gain sur 4/4 sources.

## 4. Erreur de méthode rattrapée : dev_small trop petit
dev_small (n=220) donnait **0.60 pile pour v9, v9b ET v10** → on a cru à un plateau,
voire un bug. En réalité ±0.065 de bruit + buckets minuscules (humanities n=16).
Passage à **dev_full (n=1000, ±0.031)** → signal net, v9b clairement devant.
**Leçon : ne jamais départager des modèles proches sur un dev set trop petit.**

## 5. Ce qu'on a bien fait
- Strict CoT, distillation 14B propre, format `\boxed{}`, thinking baked.
- Diagnostics empiriques systématiques (extracteur CI, longueurs, répétitions).
- Pipeline distill resumable + `select_best` (anti-boucle).
- Correction de la métrique (dev_small → dev_full).

## 6. Ce qu'on a mal fait (et corrigé)
- Jugé v7/v8 sur un score confondu par le format sans le démêler.
- Pas lu l'extracteur officiel CI assez tôt (la clé de tout).
- Sur-interprété le bruit de bucket de dev_small.

## 7. Modèle final retenu
**v11b** = self-distill 1.7B propre (`select_best`) → SFT (v9b) → **DPO on-policy fort**.
- Chaîne complète et cohérente : on apprend au modèle son propre raisonnement nettoyé,
  puis on lui apprend à préférer ses propres réponses correctes.
- dev_full **n=8** : pass@1 **0.600** (vs v9b 0.580, +2.0 sur 4/4 sources), pass@8 0.883.
- Format inchangé (`\boxed{}` + thinking baked via merge_lora).
- **À pousser sur HF** comme soumission finale.
- Repli = **v9b** (déjà poussé, commit 632402a) si la CI 16k contredisait dev_full.

## 8. Conclusion (à compléter avec la CI 16k)
> À remplir quand la CI 16k aura évalué v9b (et éventuellement v10) :
> - CI pass@1 v9b = ___
> - CI pass@1 v10 = ___
> - Modèle final confirmé = ___
> - Cohérence dev_full ↔ CI = ___
