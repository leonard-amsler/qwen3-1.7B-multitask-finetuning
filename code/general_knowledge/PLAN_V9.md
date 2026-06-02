# General Knowledge — Plan v9 (format-fix experiment on v8 data)

> Suit PLAN_V8. Date : 2026-05-29.
> Objectif : **confirmer empiriquement** que l'échec de v8 (0.3864 dev_small) est
> un **artefact de format**, pas un déficit de raisonnement, en réentraînant
> EXACTEMENT les mêmes données (`train_v8`) avec un LoRA fort (réglages v6).

## Le diagnostic qui motive v9

On a re-scoré `validation_samples/dev_v8.jsonl` (220 complétions du modèle v8) de
deux façons :

| Extracteur | pass@1 | Lecture |
|---|---:|---|
| **Officiel (CI, `evaluate/benchmarks.py`)** | **0.3864** | ce que la CI compte |
| **Lenient (vraie intention du modèle)** | **0.5182** | ce que le modèle « voulait » répondre |

- **Seulement 51.4 %** des complétions v8 contiennent `\boxed{}`.
- **48.6 %** des complétions renvoient `None` à l'extraction officielle → comptées **fausses**.
- **Écart caché par le format : +13.2 points.**

### Pourquoi le format est si décisif (et on ne le savait pas)

`evaluate/benchmarks.py` (méthode `knowledge`, lignes 27-76) :

1. Cherche d'abord `\boxed{}`. Si trouvé → c'est le candidat (robuste, court-circuit).
2. **Sinon, le candidat = TOUTE la complétion.** Puis `_extract_choice_label`
   cherche une lettre, mais **ne la retient QUE s'il y a exactement UN match**
   (`len(matches) == 1`). Dans un long CoT qui discute A, B, C, D… il y a
   forcément **plusieurs** lettres isolées → `len(matches) != 1` → **`None` → faux**.

Conclusion : **sans `\boxed{}`, une bonne réponse est presque toujours perdue.**
Le LoRA r=8 / lr=5e-5 de v8 était trop faible pour imposer `\boxed{}` au-dessus
du prior natif du 1.7B (qui répond en langage naturel).

## Hypothèse v9

Réentraîner `train_v8` avec le LoRA fort de v6 (r=64, alpha=128, lr=2e-4) doit :

- ramener la couverture `\boxed{}` à ~100 % ;
- récupérer les ~13 points perdus → **v9 ≈ 0.50-0.55 dev_small attendu**.

**Important — attente réaliste :** v9 va probablement **rester sous v6 (0.62)**.
Raison : le contenu de `train_v8` est auto-distillé du **1.7B** (plafond de
raisonnement = le 1.7B lui-même), alors que v6 est distillé du **14B**. Le
plafond « oracle » mesuré de la donnée v8 est **0.52**. v9 sert donc à :

1. **Prouver** que le problème de v8 était le format (valeur scientifique pour le rapport).
2. **Dé-risquer** la gestion du format pour v10 (on confirme que r=64 impose `\boxed{}`).

Si v9 atterrit ~0.52 comme prédit, on aura **bouclé la boucle de compréhension** :
format réglé, et la barre suivante (v10) doit venir d'un **meilleur contenu**, pas
d'un meilleur format.

## Commande SFT v9 (aucune re-distillation, données déjà prêtes)

```bash
python -m fourneurons.scripts.train \
    --dataset_dir       /scratch/data/train_v8 \
    --output_dir        /scratch/checkpoints/gk_v9 \
    --final_model_dir   /scratch/checkpoints/gk_v9/adapter \
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

Coût : ~2-3 h SFT (pas de GPU pour la distillation, le dataset existe).

## Merge + Eval

```bash
python -m fourneurons.scripts.merge_lora \
    --adapter_dir /scratch/checkpoints/gk_v9/adapter \
    --output_dir  /scratch/checkpoints/gk_v9/vllm \
    --base_model  Qwen/Qwen3-1.7B \
    --device      cpu

python -m fourneurons.eval.run_inference \
    --model      /scratch/checkpoints/gk_v9/vllm \
    --input      validation_samples/general_knowledge_dev_small.jsonl \
    --output     /scratch/eval_v9/devsmall_v9_generations.jsonl \
    --n 1 --max_tokens 4096 --max_model_len 4096 \
    --gpu_memory_utilization 0.90

python -m fourneurons.eval.report_by_bucket \
    --generations /scratch/eval_v9/devsmall_v9_generations.jsonl \
    --output      /scratch/eval_v9/devsmall_v9_report.json
```

### Vérification clé : couverture du format

```bash
python -c "
import json
rows = [json.loads(l) for l in open('/scratch/eval_v9/devsmall_v9_generations.jsonl')]
n_boxed = sum(1 for r in rows if '\\\\boxed' in r['completions'][0])
print(f'\\\\boxed coverage: {n_boxed}/{len(rows)} = {100*n_boxed/len(rows):.1f}%')
"
```

## Critères de lecture

- **Couverture `\boxed{}` ≥ 97 %** → l'hypothèse format est confirmée (l'objectif #1 de v9).
- **pass@1 ∈ [0.48, 0.56]** → cohérent avec le plafond oracle 0.52 de la donnée v8.
- Si v9 ≥ v6 (improbable) : surprise positive, on push v9.
- Sinon : on garde v6 comme baseline-à-battre, et **v10 est le vrai pari** (voir PLAN_V10).
