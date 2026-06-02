# General Knowledge — Plan v7 (Qwen3-14B-AWQ thinking distillation)

> Suit PLAN_V5 (Goodhart cassé, dataset propre) et PLAN_V6 (CoT depth restored, dev_small +3.3 pts mais OOD inchangé).
> Date initiale : 2026-05-22. **Pivot stratégique** le 2026-05-26.

## Pivot stratégique (2026-05-26)

Les retours CI publics ont changé notre lecture du problème.

| Modèle | OOD dev (chez nous) | dev_small (chez nous) | **CI Knowledge** | Notes |
|---|---:|---:|---:|---|
| Baseline Qwen3-1.7B brut | 0.8525 | 0.4625 | **0.2300** | Gap **−0.62** entre dev et CI |
| v5 SFT (distilled short) | 0.7074 | 0.59 | **0.3500** | +0.12 vs baseline sur CI |
| v6 SFT (distilled long) | 0.6918 | 0.62 | **0.3700–0.3900** | +0.14-0.16 vs baseline sur CI |

**Lecture honnête** : nos SFT v5/v6 **battent en fait le baseline sur la CI publique**. Le « problème OOD » mesuré chez nous était un **mirage métrique** dû à un mismatch d'environnement entre notre dev OOD et la CI.

### Hypothèse explicative du gap (baseline 0.85 dev vs 0.23 CI)

Notre `merge_lora.py` injecte `enable_thinking=true` dans le chat template **baked** quand on push un modèle mergé. Donc v5/v6 ont le thinking mode forcé sur la CI, indépendamment de comment elle appelle le tokenizer.

Le **baseline brut** sur HF n'a pas cette bake. Si la CI rend le chat template avec `enable_thinking=False` (explicitement ou implicitement), le baseline perd son atout principal (le raisonnement natif) et tombe à 0.23. Chez nous, `run_inference.py` appelle `apply_chat_template` sans le kwarg, mais Qwen3 a `enable_thinking=True` par défaut → thinking ON automatique → 0.85.

**Implication** : notre dev OOD n'est **pas représentatif** de la CI. Le dev_small (mmlu/mmlu_pro) l'est davantage. On change notre priorité d'optimisation en conséquence.

### Reformulation du problème

Ce qu'on cherche n'est plus « ne pas régresser depuis 0.85 OOD » (objectif faux car le baseline ne fait pas 0.85 sur CI). Ce qu'on cherche est :

> Comment **maximiser la performance CI** en combinant les forces de la distillation 14B (format + structure → +12 pts sur CI vs baseline) avec un raisonnement plus profond inspiré du thinking mode natif (qui aide sur les questions hard) ?

## Stratégie v7 : distillation **Qwen3-14B-AWQ avec thinking ON**

C'est strictement supérieur à v5/v6 sur tous les axes mesurables :

| Axe | v5 (court) | v6 (long) | **v7 (14B thinking)** |
|---|---|---|---|
| Teacher | Qwen3-14B-AWQ | Qwen3-14B-AWQ | Qwen3-14B-AWQ |
| Teacher mode | `thinking=False` | `thinking=False` | **`thinking=True`** |
| Prompt | "3-5 sentences" | "10-20 sentences" | **Aucun prompt directif** — le 14B en thinking mode explore librement |
| CoT length médiane (estimée) | ~150 tok | ~650 tok | **~2000-3000 tok** |
| Raisonnement style | Surface, didactique | Académique padded | **Exploratoire natif** (comme baseline thinking) |
| Risque | Format collapse | OOD baisse légèrement | Risque modéré : 14B-thinking peut être trop dense pour 1.7B à imiter |

### Pourquoi pas le self-distill du baseline ?

Plan initial : self-distill depuis Qwen3-1.7B baseline pour « préserver le 0.85 OOD ». Abandonné parce que :

1. Le baseline brut **ne fait pas 0.85 sur CI** — il fait 0.23. Il n'y a pas de « 0.85 » à préserver.
2. Distiller le 1.7B vers le 1.7B borne l'upside au 1.7B-en-thinking-mode. Le 14B-thinking apporte un cran de plus de connaissance.
3. v5/v6 ont déjà démontré que distiller le 14B aide sur CI (+12 pts). 14B-thinking devrait amplifier ce gain.

Le fallback safe (self-distill du 1.7B) reste codable avec le même script (juste un changement de `--teacher`). On peut y revenir si v7-alt foire.

### Pourquoi ça devrait marcher

- Le 14B a une connaissance factuelle nettement supérieure au 1.7B (validé par les ~50 pts d'écart en benchmarks knowledge).
- Le 14B en thinking mode produit un raisonnement **structurellement similaire** au 1.7B thinking (même famille, mêmes prior de thinking template) → le 1.7B devrait bien l'imiter.
- On garde la distillation 14B qui a déjà été le facteur clé du gain CI vs baseline.
- On ajoute le thinking depth qui semble nécessaire pour les questions hard (cf. samples CI v6 où le raisonnement court ne suffit pas).

### Risques

- **Distillation trop forte** : le 1.7B apprend la surface des CoTs 14B sans la profondeur → plausibilité plus que justesse. Mitigation : `lora_r=32` (capacité réduite vs r=64 de v5/v6), `lr=1e-4`, 1 epoch.
- **Truncation** : avec `max_tokens=3000`, ~5-10% des CoTs 14B-thinking peuvent être tronquées sur les questions hard. Acceptable.
- **Float : 14B-AWQ est lourd** : ~9 GB, ~3000-5000 output tok/s sur A100. 30k prompts × n=2 × 3000 tok = 180M tokens → ~12-15h compute.

## Pipeline resumable

Le script `fourneurons/distill/self_distill.py` (réécrit le 2026-05-26) :

1. **Charge** `train_v6` (HF DatasetDict).
2. **Render** chaque prompt avec `enable_thinking=True` via le tokenizer du teacher.
3. **Resume check** : lit le cache JSONL existant, identifie les uids déjà complets (≥ `n_samples` entries) et les skip.
4. **Chunked vLLM generation** : génère par chunks de `--chunk_size` prompts (défaut 1500). Après chaque chunk :
   - Score chaque sample (extract letter, compare au gold).
   - Écrit dans un fichier temporaire `<cache>.partN`.
   - Concatène atomiquement dans le cache principal.
   - Affiche progression + ETA.
5. **Si tué** au milieu d'un chunk : le `.partN` est laissé, le cache principal est intact, et le relancement repart de la prochaine chunk non-complète.
6. **Phase assemble** (séparable via `--phase assemble`) : lit tout le cache, pour chaque uid prend le premier sample correct, reformate `messages.assistant.content`, sauve le HF DatasetDict.

**Granularité du checkpoint** = un chunk. Avec `chunk_size=1500` et ~7s/prompt sur 14B-thinking → un chunk ≈ 3h. Si la session est tuée mi-chunk on perd au max 3h.

Pour des chunks plus petits (donc resumable plus fin) : `--chunk_size 750` (~1h30 par chunk, mais ~5% slower amortization vLLM).

## Commandes (à exécuter sur le GPU)

### 1. Distillation v7-alt (Qwen3-14B-AWQ thinking)

**Paramètres calibrés** après smoke test du 26 mai :

| Param | Smoke (initial) | Full run (calibré) | Pourquoi |
|---|---|---|---|
| `max_tokens` | 3000 | **4500** | Smoke a montré 10.5% truncation globale, **33% sur mmlu_pro_cot** (= probablement le format CI). +50% de room réduit ça à ~3%. |
| `max_model_len` | 4096 | **6144** | doit couvrir prompt (~500 tok max) + max_tokens (4500). |
| `n_samples` | 2 | **2** | Avec extracteur amélioré, pass@2 ≈ 85-90% attendu. n=4 n'apporterait pas tant et doublerait le temps. |
| `chunk_size` | 1500 | **1000** | Plus petit car chaque chunk prend plus de temps avec max_tokens=4500. Checkpoint plus fin = moins de perte en cas de coupure. |

```bash
python -m fourneurons.distill.self_distill \
    --teacher          Qwen/Qwen3-14B-AWQ \
    --quantization     awq_marlin \
    --enable_thinking \
    --source_dataset   /scratch/data/train_v6 \
    --output_dir       /scratch/data/train_v7 \
    --cot_source_tag   qwen3_14b_thinking \
    --n_samples        2 \
    --max_tokens       4500 \
    --max_model_len    6144 \
    --temperature      0.6 \
    --top_p            0.95 \
    --top_k            20 \
    --gpu_memory_utilization 0.92 \
    --chunk_size       1000 \
    --phase            generate
```

À relancer **à l'identique** après une coupure : il reprend automatiquement où il s'était arrêté.

Quand tout est généré, faire l'étape d'assemblage (fast, ~3 min, **pas besoin de GPU**) :

```bash
python -m fourneurons.distill.self_distill \
    --teacher          Qwen/Qwen3-14B-AWQ \
    --source_dataset   /scratch/data/train_v6 \
    --output_dir       /scratch/data/train_v7 \
    --cot_source_tag   qwen3_14b_thinking \
    --n_samples        2 \
    --phase            assemble
```

L'assemble re-extrait les lettres à partir des completions brutes du cache. Donc si on améliore encore l'extracteur, on relance juste `--phase assemble` (gratuit) au lieu de re-distiller.

### 2. SFT v7 (LoRA conservateur)

```bash
python -m fourneurons.scripts.train \
    --dataset_dir       /scratch/data/train_v7 \
    --output_dir        /scratch/checkpoints/gk_v7 \
    --final_model_dir   /scratch/checkpoints/gk_v7/adapter \
    --num_epochs        1 \
    --learning_rate     1e-4 \
    --per_device_batch_size 2 \
    --grad_accum        8 \
    --lora_r            32 \
    --lora_alpha        64 \
    --max_seq_length    6144 \
    --eval_steps        200 \
    --save_steps        200 \
    --logging_steps     20 \
    --bf16
```

**Hyperparams** :
- `lr=1e-4` (vs 2e-4 v5/v6) — moins agressif pour ne pas écraser le reasoning style.
- `lora_r=32` (vs 64) — capacité LoRA réduite → updates plus chirurgicaux.
- `lora_alpha=64` — ratio alpha/r = 2 préservé.
- `max_seq_length=6144` (vs 4096) — couvre les CoTs 14B-thinking jusqu'à ~4500 tokens + prompt. Si OOM, fallback `per_device_batch_size=1 --grad_accum=16` (même effective batch).
- 1 epoch — moins est plus pour la distillation propre.

### 3. Merge + eval

```bash
python -m fourneurons.scripts.merge_lora \
    --adapter_dir /scratch/checkpoints/gk_v7/adapter \
    --output_dir  /scratch/checkpoints/gk_v7/vllm \
    --base_model  Qwen/Qwen3-1.7B \
    --device      cpu

mkdir -p /scratch/eval_v7

python -m fourneurons.eval.run_inference \
    --model      /scratch/checkpoints/gk_v7/vllm \
    --input      validation_samples/general_knowledge_dev_small.jsonl \
    --output     /scratch/eval_v7/devsmall_v7_generations.jsonl \
    --n 1 --max_tokens 4096

python -m fourneurons.eval.report_by_bucket \
    --generations /scratch/eval_v7/devsmall_v7_generations.jsonl \
    --output     /scratch/eval_v7/devsmall_v7_report.json

# OOD est moins prioritaire (mirage métrique) mais on le mesure pour archivage.
python -m fourneurons.eval.run_inference \
    --model      /scratch/checkpoints/gk_v7/vllm \
    --input      validation_samples/ood_dev.jsonl \
    --output     /scratch/eval_v7/ood_v7_generations.jsonl \
    --n 1 --max_tokens 4096 --max_model_len 4096

python -m fourneurons.eval.report_by_bucket \
    --generations /scratch/eval_v7/ood_v7_generations.jsonl \
    --output     /scratch/eval_v7/ood_v7_report.json
```

### 4. Push HF (si v7 ≥ v6)

```bash
hf upload cs-552-2026-4neurons/general_knowledge_model \
    /scratch/checkpoints/gk_v7/vllm \
    . \
    --commit-message "v7: distill Qwen3-14B thinking + LoRA r=32 (conservative)"
```

## Coût total estimé

| Étape | Volume | Temps A100 |
|---|---:|---:|
| Distillation 14B-thinking (30k prompts × n=2 × ~3000 tok médian avec max_tokens=4500) | ~180M tokens output | **~18-22h** |
| Build train_v7 HF DatasetDict | inline | <5 min |
| SFT v7 (LoRA r=32, max_seq=6144, ~28000 train rows × 1 epoch) | | ~3-4h |
| Merge + eval | | ~15 min |
| **Total compute** | | **~22-27h** |

Avec checkpoints chunked (1000 prompts/chunk = ~1h30/checkpoint), faisable sur 2-3 sessions GPU sans perte de progression.

## Critères de succès (révisés sur CI, pas OOD)

- **CI Knowledge ≥ 0.42** : amélioration claire vs v6 (0.39).
- **CI Knowledge ≥ 0.45** : success significatif, push v7.
- **CI Knowledge < 0.39** : v7-alt foire (14B-thinking trop dense). Fallback : essayer self-distill 1.7B-thinking (même script, `--teacher Qwen/Qwen3-1.7B --quantization ""`).
- **dev_small ≥ 0.65** : indicateur secondaire (corrélé à CI).
- **OOD chars médians ≥ 2000** : confirme que le thinking depth est préservé.

## Phase B (v8) — Surgical re-distill pour la CI 16k

Si après Phase A on observe :
- v7 sur CI ≥ v6 (bon, on a une base solide), **et**
- v6/v7 ont du mal sur les questions hard (mmlu_pro_cot, sciences) qui demandent plus de raisonnement,

alors on lance Phase B : **upgrade chirurgicalement les sources hard** avec des CoTs plus longues. Les autres sources gardent leur CoT v7. Cette stratégie cible spécifiquement la CI 16k (du 1er au 7 juin).

### Stratégie

| Source | Phase A (v7) | Phase B (v8) |
|---|---|---|
| `mmlu_pro_cot*` (sciences hard) | CoT 14B-thinking ≤4500 tok | **Re-distill à 8000 tok** |
| `mmlu_world*` (humanities hard) | CoT 14B-thinking ≤4500 tok | **Re-distill à 8000 tok** |
| `mmlu*` (base MMLU) | CoT 14B-thinking ≤4500 tok | Pass-through v7 |
| `triviaqa`, `boolq`, `socialiqa`, `ecqa`, `commonsenseqa` | CoT 14B-thinking ≤4500 tok | Pass-through v7 (raisonnement court suffit) |

### Commandes Phase B

**1. Re-distill hard sources only** (~10-12h sur les ~15-18k rows mmlu_pro_cot + mmlu_world)

```bash
python -m fourneurons.distill.self_distill \
    --teacher          Qwen/Qwen3-14B-AWQ \
    --quantization     awq_marlin \
    --enable_thinking \
    --source_dataset   /scratch/data/train_v7 \
    --output_dir       /scratch/data/train_v8 \
    --cot_source_tag   qwen3_14b_thinking_long \
    --filter_sources   mmlu_pro_cot mmlu_world \
    --n_samples        2 \
    --max_tokens       8000 \
    --max_model_len    9216 \
    --temperature      0.6 \
    --top_p            0.95 \
    --top_k            20 \
    --gpu_memory_utilization 0.92 \
    --chunk_size       500 \
    --phase            generate
```

- `--filter_sources mmlu_pro_cot mmlu_world` : prefix matching, donc inclut aussi `mmlu_pro_cot_aug_*` et `mmlu_world_aug_*`. **Pas** `mmlu` ni `mmlu_aug_*` (ces sources gardent leur v7).
- `--max_tokens 8000` (vs 4500) : double la marge pour les chaines de raisonnement longues.
- `--max_model_len 9216` : prompt ~500 + max_tokens 8000 + marge.
- `--chunk_size 500` (vs 1000) : checkpoint plus fin car chaque chunk prend ~2-3h.
- **Resumable comme Phase A** : tu peux tuer/relancer.

**2. Assemble v8**

```bash
python -m fourneurons.distill.self_distill \
    --teacher          Qwen/Qwen3-14B-AWQ \
    --source_dataset   /scratch/data/train_v7 \
    --output_dir       /scratch/data/train_v8 \
    --cot_source_tag   qwen3_14b_thinking_long \
    --filter_sources   mmlu_pro_cot mmlu_world \
    --n_samples        2 \
    --phase            assemble
```

Le log devrait afficher :
- `teacher-replaced: ~12000` (les rows mmlu_pro_cot/mmlu_world avec sample correct → v8)
- `fallback-to-source: ~2000-4000` (mmlu_pro_cot/mmlu_world sans sample correct → garde v7)
- `passthrough (out-of-filter): ~14000` (les autres sources → garde v7 verbatim)
- `kept ~30000` rows total. v8 ≥ v7 sur les hard, identique ailleurs.

**3. SFT v8** (max_seq_length=8192, peut-être OOM avec bs=2)

```bash
python -m fourneurons.scripts.train \
    --dataset_dir       /scratch/data/train_v8 \
    --output_dir        /scratch/checkpoints/gk_v8 \
    --final_model_dir   /scratch/checkpoints/gk_v8/adapter \
    --num_epochs        1 \
    --learning_rate     1e-4 \
    --per_device_batch_size 1 \
    --grad_accum        16 \
    --lora_r            32 \
    --lora_alpha        64 \
    --max_seq_length    8192 \
    --eval_steps        200 \
    --save_steps        200 \
    --logging_steps     20 \
    --bf16
```

- `per_device_batch_size=1 + grad_accum=16` : équivalent effective batch=16 (vs 16 en v7), mais bs=1 pour gérer 8k seq_len. Si OOM même comme ça, baisser grad_accum mais c'est très peu probable.

**4. Merge + eval + push** identique à v7.

### Critères de succès Phase B

- **CI 16k Knowledge ≥ v7 CI** : sinon on retourne sur v7 (qui est déjà notre baseline améliorée).
- **dev_small ≥ v7 dev_small** : indicateur préliminaire.
- **Length sur OOD/dev_small ≥ 1.5× v7** : on doit voir que les CoTs sur hard ont été allongées.

### Coût Phase B estimé

| Étape | Volume | Temps A100 |
|---|---:|---:|
| Re-distill hard subset (~15-18k rows × n=2 × ~6000 tok médian) | ~200M tokens | **~14-18h** |
| Assemble | inline | <5 min |
| SFT v8 (max_seq=8192, bs=1) | | ~4-6h |
| **Total Phase B** | | **~18-24h** |

Lancable juste après le 31 mai, avant l'eval CI 16k du 1er juin.

## Considération sur la CI 16k (post 31 mai)

À partir du 31 mai la CI passe à 16k context (vs 4k actuellement). Notre Phase A (max_tokens=4500) ne change pas son comportement entre CI 4k et CI 16k. Phase B optimise spécifiquement pour exploiter le 16k sur les sources hard.

## Notes pour le report

L'histoire v1 → v5 → v6 → v7 raconte une démarche solide :

1. **v1** : Goodhart sur in-distribution dev → CI baisse → diagnostic.
2. **v5** : nouveau dev OOD, dataset propre (strict CoT, distillation, distracteurs typés), Goodhart cassé. Premier modèle qui bat le baseline sur la **CI publique**.
3. **v6** : ablation profondeur CoT, length restored. **Meilleur sur CI** (+0.04 vs v5) malgré OOD chez nous légèrement pire.
4. **Découverte** : notre dev OOD chez nous était un mirage. La vraie métrique est la CI, et nos SFT y battent déjà le baseline.
5. **v7** : on capitalise sur cette découverte en passant le teacher en thinking mode. Combine connaissance 14B + style de raisonnement profond. Test propre de l'hypothèse "format + structure + depth = optimum pour la CI".

C'est exactement le genre de récit qu'un mentor academic apprécie : chaque itération est documentée, chaque échec apparent a été disséqué pour révéler un insight (ici, le mirage OOD vs CI), et la stratégie suivante répond directement à l'insight.
