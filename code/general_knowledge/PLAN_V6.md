# General Knowledge — Plan v6 (CoT depth restoration)

> Suit le PLAN_V5 qui a corrigé la pollution dataset (synthetic CoTs, mauvais distracteurs, déséquilibre de macros) mais n'a pas restauré la profondeur de raisonnement.
> Date de rédaction : 2026-05-21.

## Constat post-v5

| Modèle | OOD pass@1 | OOD median chars | ~tokens | Lecture |
|---|---:|---:|---:|---|
| Baseline Qwen3-1.7B brut | **0.8525** | 4820 | ~1300 | Reasoning natif profond |
| v1 (SFT, synthetic CoTs) | 0.7122 | 596 | ~150 | Format collapse |
| **v5 (SFT, distilled CoTs, T=0.7)** | **0.7074** | **606** | **~150** | Format collapse identique |
| v5 (SFT, distilled CoTs, greedy) | 0.7170 | 606 | ~150 | +1pt, dans le bruit (CI ±0.02) |

**Le smoking gun** : v5 produit en moyenne **606 chars (~150 tokens)** de raisonnement, **8× moins** que le baseline. Le greedy decoding ne change rien. Le sampling n'est pas le bottleneck.

**Cause racine identifiée** : le prompt utilisateur pour la distillation Qwen3-14B-AWQ demande explicitement **"3 to 5 sentences"** (`fourneurons/distill/prompts.py`, ligne 52). Les CoTs distillées avaient une médiane de **~546 chars** en cache. Le student (Qwen3-1.7B) a fidèlement appris à émettre ~150 tokens puis `\boxed{X}`.

Le strict_cot, la distillation, les distracteurs typés ont tous fonctionné individuellement. Mais aucun n'attaquait le bon problème : **la longueur cible du raisonnement dans les données train**.

## Décision stratégique : re-distillation chirurgicale

Re-distiller toutes les sources avec des CoTs longs serait inutile (et coûteux) pour des questions de fact-lookup ou de raisonnement causal court. Stratégie surgical :

### Tier LONG (re-distillation, ~14 700 questions)

Sources où la profondeur du raisonnement compte vraiment :

| Source | Originals | Macro principale | Raison |
|---|---:|---|---|
| `mmlu` (validation) | 1 527 | mixed STEM/hum/soc | Toutes subjects, raisonnement multi-étapes |
| `mmlu_world` (test+dev, 29 subjects) | 7 539 | hum + soc + history | Philosophie, droit, économie, etc. |
| `mmlu_pro_cot` (train) | 5 670 | mostly STEM | Maths, sciences, multi-step |
| **Total** | **~14 700** | | |

Pour ces sources, le prompt teacher demandera explicitement **10-20 phrases / 300-700 mots de raisonnement explicite étape par étape**, et le filtre exigera **min_chars=1000** (~250 tokens).

### Tier SHORT (cache v5 conservé tel quel, ~129 000 questions)

Sources où le format actuel (3-5 phrases) est adapté :

| Source | Originals | Macro |
|---|---:|---|
| `triviaqa` | 76 502 | history_geo (fact lookup) |
| `socialiqa` | 33 004 | commonsense (causal court) |
| `boolq` | 9 427 | commonsense (binary) |
| `ecqa` | 7 464 | commonsense (CoT natif court) |
| `commonsenseqa` | 2 991 | commonsense |

Pour ces sources, on **garde le cache `distilled_cot_v5.jsonl`** existant tel quel. Les CoTs courts y sont adaptés au contenu : il n'y a pas grand-chose à expliquer pour "What is the capital of France?".

## Implementation

### 1. `fourneurons/distill/prompts.py`

- Garder `build_user_prompt(question, gold_text)` comme variante **short** (3-5 sentences).
- Ajouter `build_user_prompt_long(question, gold_text)` qui demande explicitement :
  - 10-20 phrases de raisonnement étape par étape
  - Couvrir les concepts/faits clés, les déductions intermédiaires, comparaisons, computations
  - Conclure naturellement sur pourquoi la réponse est correcte
  - Pas de préambule, pas de label, juste l'explication
- Adapter `build_messages(question, gold_text, style="short")` pour router.

### 2. `fourneurons/distill/distill.py`

- Ajouter `LONG_REASONING_SOURCES = frozenset({"mmlu", "mmlu_world", "mmlu_pro_cot"})`.
- Router via `ex.source` dans `_build_prompts`.
- Passer `min_chars=1000` à `quality_check` pour les sources LONG, sinon valeur par défaut (200).
- Bumper `--max_tokens` par défaut à **2048** (vs 1024) pour laisser place aux longues explications.

### 3. `fourneurons/data/build_train.py`

- `--distilled_cot_cache` accepte maintenant `nargs="+"` (plusieurs chemins).
- Chargement last-wins : si un uid apparaît dans plusieurs caches, on garde la version du dernier path (donc le v6_long override le v5 pour les uids communs aux sources LONG).

## Commande de re-distillation

```bash
python -m fourneurons.distill.distill \
    --teacher Qwen/Qwen3-14B-AWQ \
    --output  /scratch/data/distilled_cot_v6_long.jsonl \
    --sources mmlu mmlu_world mmlu_pro_cot \
    --max_tokens 2048
```

- Sortie : **nouveau fichier**, pas de reprise du v5 (le v5 contient des CoTs courts pour ces uids, on les remplace).
- Volume : ~14 700 prompts. À ~600 tokens output médian × 2000 toks/s vLLM A100 → **~2-3h**.

## Commande de re-build train

```bash
python -m fourneurons.data.build_train \
    --output_dir          /scratch/data/train_v6 \
    --total               30000 \
    --max_variants        1 \
    --distilled_cot_cache /scratch/data/distilled_cot_v5.jsonl /scratch/data/distilled_cot_v6_long.jsonl \
    --seed                42
```

- L'ordre compte : `v6_long` après `v5` → override pour les uids LONG.
- On garde `--total 30000` (mêmes quotas que v5 r2, qui marchaient bien).

## Commande SFT v6

Identique à v5 (mêmes hyperparams pour isoler l'effet du dataset) :

```bash
python -m fourneurons.scripts.train \
    --dataset_dir       /scratch/data/train_v6 \
    --output_dir        /scratch/checkpoints/gk_v6 \
    --final_model_dir   /scratch/checkpoints/gk_v6/adapter \
    --num_epochs        1 \
    --learning_rate     2e-4 \
    --per_device_batch_size 2 \
    --grad_accum        8 \
    --lora_r            64 \
    --lora_alpha        128 \
    --max_seq_length    4096 \
    --eval_steps        200 \
    --save_steps        200 \
    --logging_steps     20 \
    --bf16
```

Estimation : ~2h SFT (l'augmentation des CoTs longs allonge un peu chaque step car plus de tokens à attribuer à la loss).

## Merge + Eval

```bash
python -m fourneurons.scripts.merge_lora \
    --adapter_dir /scratch/checkpoints/gk_v6/adapter \
    --output_dir  /scratch/checkpoints/gk_v6/vllm \
    --base_model  Qwen/Qwen3-1.7B \
    --device      cpu

# OOD (la vraie métrique)
python -m fourneurons.eval.run_inference \
    --model      /scratch/checkpoints/gk_v6/vllm \
    --input      validation_samples/ood_dev.jsonl \
    --output     /scratch/eval_v6/ood_v6_generations.jsonl \
    --n 1 --max_tokens 4096 --max_model_len 4096 \
    --gpu_memory_utilization 0.90

python -m fourneurons.eval.report_by_bucket \
    --generations /scratch/eval_v6/ood_v6_generations.jsonl \
    --output      /scratch/eval_v6/ood_v6_report.json

# dev_small (pour Goodhart)
python -m fourneurons.eval.run_inference \
    --model      /scratch/checkpoints/gk_v6/vllm \
    --input      validation_samples/general_knowledge_dev_small.jsonl \
    --output     /scratch/eval_v6/devsmall_v6_generations.jsonl \
    --n 1 --max_tokens 4096

python -m fourneurons.eval.report_by_bucket \
    --generations /scratch/eval_v6/devsmall_v6_generations.jsonl \
    --output      /scratch/eval_v6/devsmall_v6_report.json
```

Et l'analyse de longueur :

```bash
python -c "
import json, statistics
for name, path in [('v6_ood', '/scratch/eval_v6/ood_v6_generations.jsonl'),
                   ('v6_devsmall', '/scratch/eval_v6/devsmall_v6_generations.jsonl')]:
    chars = []
    with open(path) as f:
        for line in f:
            chars.append(len(json.loads(line)['completions'][0]))
    chars.sort()
    n = len(chars)
    print(f'{name}: n={n}, median={statistics.median(chars):.0f}, mean={statistics.mean(chars):.0f}, p10={chars[n//10]}, p90={chars[int(n*0.9)]}')
"
```

## Critères de succès

- **Longueur** : médiane v6 sur OOD ≥ **1500 chars** (~375 tokens, 10× v5).
- **pass@1 OOD** : v6 ≥ **0.78** (objectif réaliste vu qu'on attaque la cause racine, mais sans atteindre forcément 0.85).
- **dev_small** : attendu ~0.60-0.65 (honnête, sans rebond Goodhart).
- **Format coverage** : `\boxed{}` toujours à 100% (pas régression sur le format).

## Décision finale

- Si v6 OOD ≥ 0.78 : on push v6 sur HF, c'est notre meilleur modèle.
- Si v6 OOD ∈ [0.73, 0.78] : amélioration modérée, on push, mais on note le plafond intrinsèque du paradigme SFT-on-MCQ pour le report.
- Si v6 OOD < 0.73 : on revient sur v5 et on ship v5 avec le narratif "Goodhart cassé, plafond LoRA-SFT".

## Notes pour le report

Ce parcours v1 → v5 → v6 raconte une histoire pédagogiquement très solide :

1. **v1** : on a mesuré sur un dev set in-distribution, on a optimisé Goodhart, le CI public a baissé.
2. **v5** : on a construit un dev OOD propre (ARC + OpenBookQA), refait le dataset entièrement (strict_cot, distillation, distracteurs typés), cassé Goodhart, mais on a découvert que la perf OOD ne progressait pas — diagnostic empirique : le format collapse persiste indépendamment de la qualité des CoTs.
3. **v6** : ablation chirurgicale sur la profondeur du raisonnement dans le teacher prompt. C'est le test propre de l'hypothèse "longer CoTs at training time → longer reasoning at inference time → better OOD generalization".
