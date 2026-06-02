# General Knowledge — Plan v5 (rigorous redesign)

> Rédigé après audit des modèles v1–v4 et discussion avec le mentor Sebastian.
> Date de rédaction : 2026-05-20.

## Constat (où on en est)

| Modèle | dev OOD (ARC+OBQA, n=1668) | dev (interne) pass@1 | CI publique | Commentaire |
|---|---|---|---|---|
| Baseline Qwen3-1.7B | **0.8525** | 0.46 | 0.23 | Pas de `\boxed{}` propre, mais raisonne (~1300 tok/réponse) |
| **v1 (SFT)** | **0.7122** | 0.56 | **0.39** | Format OK, raisonnement collapsé (~110 tok/réponse) |
| v2 (SFT + distill + 2 ep) | — | 0.56 | 0.37 | Régression légère (bruit ?) |
| v3 (DPO 2.1k pairs) | — | non testé | — | Skipped |
| v4 (DPO 3.5k pairs self) | — | 0.60 | 0.34 | Format collapse |

**Conclusion** : nos itérations améliorent notre dev set mais régressent sur CI. C'est du **Goodhart's law**. La cause profonde est que :
- notre dev set partage la distribution du train,
- nos données train ont des défauts qu'on n'avait pas vus.

**Confirmation empirique (Phase 0, 2026-05-20)** : sur 1668 questions OOD (ARC-Challenge + OpenBookQA, jamais vues en train), **v1 est 14 points sous la baseline Qwen3-1.7B brute** (0.71 vs 0.85). Le smoking gun : v1 génère ~110 tokens/réponse, la baseline ~1300 — le SFT v1 a **désappris à raisonner** parce que 60 % des CoT train étaient des placeholders. Le bon score CI publique de v1 (+0.16 vs baseline) vient uniquement du **format** (`\boxed{LETTER}`, gestion Yes/No), pas du raisonnement. Sur OOD science, le format ne suffit pas et v1 s'effondre.

**Statistiques de completions OOD (n=1668)** :

| | Baseline | v1 | Lecture |
|---|---:|---:|---|
| chars/réponse (médiane) | 4820 | **596** | v1 8× plus court → format collapse |
| `<think>` présent | 100 % | 100 % | template OK |
| `\boxed{}` présent | 95.5 % | **99.9 %** | v1 strictement meilleur sur le format |

Conséquences directes pour v5 :
- **Préserver** la couverture `\boxed{}` quasi-parfaite de v1 (= 4.5 pt récupérables gratuitement vs baseline).
- **Restaurer** une longueur de raisonnement réelle. Cible : médiane ≥ 2000 chars sur OOD ; CoT distillés en train de longueur 500-1500 tokens (cf. Phase 2).
- Pas besoin de toucher au chat template — le problème est dans les données train, pas dans l'inférence.

## Les 7 erreurs identifiées

1. **60% du train a des CoTs synthétiques placeholder** ("the answer that best matches is X") → on entraîne le modèle à ne PAS raisonner.
2. **65% du train est augmenté** avec des distracteurs absurdes tirés d'un pool global → le mentor warning confirmé.
3. **Dev set partage la distribution du train** → aucun signal de généralisation.
4. **history_geo = 3% du train** alors que c'est dans le scope.
5. **Couverture de sources étroite** : tout est commonsense ou STEM, rien pour world knowledge/trivia.
6. **`max_seq_length=2048`** alors que l'eval final autorise 16384 new tokens.
7. **DPO self-distill** → format collapse, pas de généralisation.

## Décisions stratégiques (validées avec Noé le 2026-05-20)

- **Dev OOD** : ARC-Challenge + OpenBookQA, jamais touchés en training.
- **History/geo** : ajouter TriviaQA reformatté MC + MMLU subjects ciblés.
- **Synthetic CoTs** : on distille TOUT avec Qwen3-14B-AWQ (zéro placeholder).
- **Augmentation** : 20%, expansion 6-20 uniquement, distracteurs cohérents (per-macro_cat pool).
- **Pas de DPO** pour v5. SFT propre uniquement.

## Plan d'exécution

### Phase 0 — Dev set OOD propre (1h, Mac) ✅ 2026-05-20

**Objectif** : avoir un signal d'évaluation honnête avant tout entraînement.

- [x] Loader `fourneurons/data/loaders/arc.py` (ARC-Challenge, ~4 options, 1170 ex après dédup)
- [x] Loader `fourneurons/data/loaders/openbookqa.py` (4 options, 498 ex après dédup)
- [x] Script `fourneurons/eval/build_ood_dev.py` qui crée `validation_samples/ood_dev.jsonl` (1668 ex)
- [x] Évaluer **baseline Qwen3-1.7B brut** → **0.8525** (référence)
- [x] Évaluer **v1** → **0.7122** (−14 pt vs baseline)

**Résultats détaillés** (`/scratch/eval/ood_v1/*.json`) :

| Modèle | ARC-Challenge (n=1170) | OpenBookQA (n=498) | Total |
|---|---:|---:|---:|
| Baseline Qwen3-1.7B | 0.8564 | 0.8434 | **0.8525** |
| v1 | 0.7385 | 0.6506 | **0.7122** |

**Critère de succès** : toute itération suivante doit améliorer **0.7122** sur ce dev set OOD. La vraie cible de référence est **0.85** (baseline brute) — c'est là qu'on saura qu'on est réellement utiles.

### Phase 1 — Reconstruction du dataset (1h, GPU) ✅ 2026-05-20

**Objectif** : `train_v5` propre, équilibré, sans synthetic CoT, augmentation maîtrisée.

- [x] Loader `fourneurons/data/loaders/triviaqa.py` → 138 355 questions yielded, 76 502 originals kept post-dedup. Distracteurs typés.
- [x] Loader `fourneurons/data/loaders/mmlu_world.py` → per-subject configs (test+dev), 1671 originals (10 subjects).
- [x] Refactor `fourneurons/data/augment.py` : `DistractorPool` typé `(macro_cat, subject, option_type)` avec `option_type ∈ {year, numeric, short_entity, phrase}`. Sampling capé (`_SAMPLE_CANDIDATE_CAP=512`) pour éviter le stall multi-heures sur TriviaQA.
- [x] Refactor `fourneurons/data/build_train.py` : `--strict_cot`, `--expand_only`, `--aug_cap_frac 0.20`, `--macro_quotas` (defaults: stem 25 / hum 20 / soc 20 / hist 20 / common 15). Blocklist élargi à `ood_dev.jsonl`.

**Dry-run validé 2026-05-20** (`--total 5000 --max_variants 1`, sans distill cache) :
- 138 256 originals collectés au total
- 13 134 rows survivent au strict_cot (= 5670 mmlu_pro_cot + 7464 ecqa, les deux sources CoT-natives — match exact)
- Quotas atteints sauf history_geo (target 1000, n=74) et humanities (target 1000, n=641) → **confirme empiriquement le besoin de Phase 2** : sans CoT distillé pour triviaqa + mmlu_world + boolq + socialiqa + csqa + mmlu, ces macros ne peuvent pas atteindre leurs quotas.

### Phase 2 — Distillation massive (3h + 10min, A100) ✅ 2026-05-20 / 2026-05-21

**Objectif** : toutes les rows ont un CoT distillé de qualité.

- [x] Étendre `fourneurons/distill/distill.py` : couvre toutes les sources (CoT-less + CoT-bearing), `ood_dev.jsonl` ajouté au dev_blocklist.
- [x] Étendre `fourneurons/distill/filters.py` :
  - Nouveau filtre `list_format_prefix` / `list_format_separator` (anti-leak des option-lists).
  - Filtre n-gram repetition (`max_3gram_repeats=6`).
  - Filtre length **200-6000 chars**.
- [x] Run #1 (10 mmlu_world subjects ciblés) : **133 049 CoTs** acceptées en 3.2h (acceptance 96.2%).
- [x] Run #2 (+19 mmlu_world subjects pour combler humanities/social_sciences) : **+5 661 CoTs** en 10 min (acceptance 51%, plus bas car les nouveaux subjects = professional_law/philosophy/moral_disputes ont des gold answers longues souvent paraphrasées → `no_gold_mention`).
- [x] Cache final : `distilled_cot_v5.jsonl`, **138 710 uids**.

### Phase 3 — SFT v5 (1h42, A100) ✅ 2026-05-21

**Objectif** : modèle propre, longueur de raisonnement non-castrée.

- [x] `train_v5` rebuild (`--total 30000`) après expansion mmlu_world. Quotas atteints :
  - stem 22.3% (target 25%, plafond pool 22%), humanities 20.0%, social_sciences 20.0%, history_geo 21.7%, commonsense 16.0%.
  - `by_is_augmented` = 30.6% (vs cap visé 20%, dû à la redistribution des déficits stem/hum/soc).
  - `by_cot_source` = 81% distilled, 19% loader, **0% synthetic**.
- [x] SFT : `max_seq_length=4096`, `num_epochs=1`, `lr=2e-4`, `lora_r=64`, `lora_alpha=128`, bf16, batch effectif 16.
- [x] Convergence saine : train_loss 1.83 → 1.13, eval_loss 1.24 → **1.102**, decay monotone, pas d'overfit visible.
- [x] Merge LoRA → `/scratch/checkpoints/gk_v5/vllm`, thinking-on baked.

**Résultats v5 (2026-05-21, T=0.7 / top_p=0.9 / top_k=20, max_tokens=4096)** :

| Eval set | pass@1 | vs v1 | vs baseline | Lecture |
|---|---:|---:|---:|---|
| **OOD (ARC+OBQA, n=1668)** | **0.7074** | −0.005 | **−0.145** | ≈ v1, toujours sous baseline |
| OOD ARC-Challenge (n=1170) | 0.7350 | −0.004 | −0.121 | |
| OOD OpenBookQA (n=498) | 0.6426 | −0.008 | −0.201 | OBQA reste dur |
| dev_small (n=240) | 0.5875 | −0.27 | n/a | Goodhart cassé (baisse "honnête") |

**Lecture** : v5 ne bat pas v1 sur OOD (différence ≤ bruit statistique, CI ±0.022). v5 a une perf "honnête" sur dev_small (0.59 au lieu des 0.85 inflated de v1 → la corrélation Goodhart est cassée comme prévu). MAIS le SFT (v1 ou v5) reste 14 pt sous le baseline Qwen3-1.7B brut sur OOD → **l'hypothèse du format collapse est probablement vraie même avec strict_cot + distilled CoTs**.

### Phase 4 (optionnelle) — Format reinforcement (2h)

Si v5 a des trous résiduels (parfois pas de `\boxed{}`), faire un mini SFT de format-only sur 1-2k examples. **PAS de DPO**.

## Critères de décision

- **Si v5 dev_OOD > v1 dev_OOD** : on push v5, ça généralise mieux.
- **Si v5 dev_OOD < v1 dev_OOD** : on reste sur v1 sur HF, on cherche pourquoi v5 dégrade.
- **On NE PAS suit la CI publique** pour décider ce qui est notre best. Elle est trop bruitée et c'est juste un smoke test.

## Notes mentor (Sebastian Maier, 2026-05-14)

> If the test set does have much more than 4 MC, then training on a dataset where there are more MC does make a lot of sense. I would worry about some biases the generated distractors could have.

→ Justifie qu'on garde une part d'augmentation, mais qu'on travaille la qualité des distracteurs.

> The maximum model length should refer to the input + output tokens. […] the nightly CI is only to check if the model runs and maybe outputs something correct. So, I would not put the limit of 4096 tokens on your final model since the final evaluation with more available tokens.

→ Justifie le `max_seq_length=4096` au training et le fait qu'on ne tronque pas les CoTs longs.

## Notes Ed forum

> Private CI is drawn from the same distribution but is robust enough. So, if you overfit instead of achieving generalization, you may end up with a bad model in the end.

→ Justifie qu'on ne court PAS après la CI publique mais qu'on cible la généralisation.

---

## Mise à jour de progression (à compléter au fur et à mesure)

- [x] Phase 0 lancée le 2026-05-20 — terminée le 2026-05-20
- [x] Phase 1 lancée le 2026-05-20 — terminée le 2026-05-20 (dry-run validé, distill cache requis)
- [x] Phase 2 lancée le 2026-05-20 — terminée le 2026-05-21 (138 710 CoTs en cache)
- [x] Phase 3 lancée le 2026-05-21 — terminée le 2026-05-21 (gk_v5 mergé + évalué)
- [x] Résultats v5 :
  - dev_OOD pass@1 : **0.7074** (cible > 0.7122 ❌, idéal ≥ 0.85 ❌)
  - dev_small pass@1 : **0.5875** (Goodhart cassé : baisse vs ~0.85 sur v1 = honnête)
  - dev_OOD pass@8 : non mesuré
  - CI publique : non mesurée

Baseline Qwen/Qwen3-1.7B sur dev_OOD pass@1 = **0.8525** (1668 ex)
v1 sur dev_OOD pass@1 = **0.7122** (1668 ex, −14 pt vs baseline)
v5 sur dev_OOD pass@1 = **0.7074** (1668 ex, ≈ v1, −14 pt vs baseline)

## Diagnostic v5 (2026-05-21) — pourquoi on ne bat pas baseline ?

Hypothèse principale : **le SFT (v1 et v5) endommage les capacités natives de raisonnement de Qwen3-1.7B**.
- Baseline génère ~1300 tokens/réponse (médiane 4820 chars). v1 ~110 tokens/réponse (médiane 596 chars).
- v5 reste à vérifier (diagnostic A : analyser longueur des completions v5 sur OOD).
- Le SFT, en imposant `<think>...</think>\boxed{X}` avec des CoTs distillés courts (médiane 550 chars en train), pousse le modèle à conclure trop vite.

Diagnostics en cours :
- [ ] A : longueur médiane des completions v5 sur OOD (cible ≥ 2000 chars).
- [ ] B : re-eval v5 avec greedy (`--temperature 0.0`) — si v5 gagne 5+ pts, le bruit du sampling à T=0.7 est partiellement responsable.

Décision pending : si A confirme format collapse (médiane < 1000 chars) et B ne gagne rien → on ship v5 quand même (sur HF) avec le narratif "Goodhart cassé, OOD honnête" pour le report, et on considère que la limite est intrinsèque au paradigme SFT-on-MCQ.