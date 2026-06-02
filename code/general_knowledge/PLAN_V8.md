# General Knowledge — Plan v8 (self-distill 1.7B baseline)

> Suit PLAN_V7.md. Date : 2026-05-28.
> v7 (Qwen3-14B-AWQ + thinking mode comme teacher) a régressé vs v6 : dev_small 0.57 (vs v6 0.62). Diagnostic clair. Pivot net.

## Récap honnête de ce qu'on a appris

### Vérité #1 — Le 0.23 CI baseline n'est PAS un problème de thinking mode

Le baseline est poussé avec `merge_lora.py` qui **bake `enable_thinking=true`** dans le chat template. La CI évalue donc le baseline en thinking ON, comme nos modèles fine-tunés.

L'explication réelle de l'écart 0.45 dev_small → 0.23 CI est que **le dev de la CI est structurellement plus difficile** que notre dev_small. Les samples CI v5/v6 disponibles montrent des questions du genre :

- Gamma-rays interagissant avec CMB photons (Klein-Nishina, énergie seuil)
- Configurations énergétiques de 13 particules chargées dans un dodécagone
- Acétylcholine et Alzheimer (clinique avancée)

Ce sont des questions MMLU-Pro hard / niche, pas du MMLU vanilla. Notre dev_small (40% mmlu vanilla, 30% mmlu_pro, reste boolq/csqa) est donc un proxy **biaisé vers les questions faciles**.

**Conséquence pratique** : dev_small reste utile (+0.16 sur dev_small se traduit +0.16 sur CI pour v6), mais on doit chercher à attaquer les questions HARD spécifiquement.

### Vérité #2 — L'OOD dev set (ARC + OBQA) est leakage Qwen pretraining

Baseline thinking ON OOD = 0.85, baseline dev_small = 0.45. Quand le baseline mémorise des questions vues en pretraining, le score explose artificiellement. Toute perturbation des poids (= n'importe quel SFT) casse la mémorisation → score plonge.

On abandonne définitivement l'OOD dev comme indicateur. dev_small + CI sont nos deux seuls signaux fiables.

### Vérité #3 — v7 a régressé par surface imitation du 14B-thinking

3 exemples de questions où v6 répond juste et v7 se trompe :

1. **Fallacy appeal to spite (gold=C)** : v6 conclut C calmement. v7 fait une auto-discussion ("Therefore D. Let me just check again... So D is correct"), se persuade, conclut D faux.
2. **Slide projector focal length (gold=D)** : v7 boucle ("Wait, maybe the screen is not at 120 inches... Wait..."), **dépasse `max_tokens` sans produire `\boxed{}`** → unscoreable.
3. **Proactive vs retroactive inhibition (gold=D)** : v7 confond les deux définitions et conclut avec **confiance** sur la mauvaise.

**Pattern** : le 1.7B a copié le **style** Qwen3-14B-thinking ("Wait... Alternatively... Let me reconsider..."), mais sans la capacité du 14B à juger correctement ses propres pivots. Résultat : sur-raisonnement plausible → mauvaise conclusion.

C'est exactement le risque qu'on avait identifié comme « hypothèse A » avant le lancement. La conclusion est claire : **le 14B est trop loin du 1.7B pour servir de teacher en mode thinking**. Un teacher doit être proche du student capacity-wise pour que l'imitation soit honnête.

## Stratégie v8 : self-distill du baseline 1.7B

### Pourquoi ça doit marcher

Le baseline 1.7B en thinking ON a deux propriétés clés :

1. **Reasoning natif fonctionnel** (0.45 dev_small, médiane 4820 chars sur OOD = vrai raisonnement déployé).
2. **Format imparfait** (95% `\boxed{}` coverage observé, donc 5% des outputs sont non-scoreables).

L'objectif v8 est **uniquement** de corriger la propriété 2 sans toucher à la propriété 1. Pour ça :

- **Teacher = baseline lui-même** → par construction, on ne peut pas avoir d'imitation surface : le 1.7B sait déjà produire ce qu'on lui demande de reproduire.
- **On garde seulement les samples où le baseline a déjà la bonne réponse**. Le LoRA n'a plus qu'à apprendre la consistance du format à la fin.
- **LoRA très petit, peu d'updates**. r=8, lr=5e-5, 1 epoch → modification minimale des poids.

Sur les questions hard où le baseline se trompe (~55% du dev_small CI-like), on ne demande pas au modèle de faire mieux. On le laisse faire son mieux et on assure juste le format. C'est borné comme upside, mais c'est SAFE.

### Comparaison des stratégies

| Stratégie | Teacher | Mode | Capacité gap | Risque | Upside vs v6 |
|---|---|---|---|---|---|
| v5/v6 | 14B-AWQ | thinking OFF | gros (14B → 1.7B) | format collapse | démontré +0.04 CI |
| **v7** | **14B-AWQ** | **thinking ON** | **gros** | **surface imitation** | **−0.05 dev_small (foiré)** |
| **v8** | **1.7B baseline** | **thinking ON** | **zero (self)** | **borné par baseline** | **+0.02 à +0.05 CI espéré** |

v8 n'est pas plus ambitieux que v6 en absolu — c'est volontaire. On a essayé l'ambitieux (v7) et ça a foiré. On revient sur du conservatif qui ne peut, par construction, pas régresser sur la qualité du raisonnement.

### Critères de succès v8

- **dev_small ≥ 0.62** : on doit au moins égaler v6.
- **dev_small ≥ 0.65** : meilleur que v6, on push.
- **CI Knowledge ≥ 0.40** : objectif réaliste vu qu'on garde le reasoning natif et qu'on ajoute le format.
- **Médiane chars dev_small ≥ 2500** : confirme qu'on préserve le thinking deployment du baseline.
- **`\boxed{}` coverage = 100%** : le seul vrai gain mécanique vs baseline brut.

Si v8 < v6 sur dev_small : on tient v6 comme push final et on documente honnêtement v7/v8 comme échecs instructifs dans le report.

## Implementation

### Distillation v8

```bash
python -m fourneurons.distill.self_distill \
    --teacher          Qwen/Qwen3-1.7B \
    --quantization     "" \
    --enable_thinking \
    --source_dataset   /scratch/data/train_v6 \
    --output_dir       /scratch/data/train_v8 \
    --cot_source_tag   self_distill_baseline \
    --n_samples        4 \
    --max_tokens       3000 \
    --max_model_len    4096 \
    --temperature      0.6 \
    --top_p            0.95 \
    --top_k            20 \
    --gpu_memory_utilization 0.92 \
    --chunk_size       2000 \
    --no_fallback_to_source \
    --phase            generate
```

**Paramètres clés :**

| Param | Valeur | Justification |
|---|---|---|
| `--teacher` | `Qwen/Qwen3-1.7B` | Le baseline lui-même (≠ v7 qui était 14B-AWQ) |
| `--quantization ""` | (none, FP16) | 1.7B fits in FP16, pas besoin d'AWQ |
| `--n_samples 4` | 4 | Baseline pass@1 ≈ 0.45 → pass@4 = 1−0.55^4 = 0.91. On garde ~90% des questions. |
| `--max_tokens 3000` | 3000 | 1.7B-thinking médiane ≈ 1500-2000 tokens, queue à 2500. 3000 = 99e percentile. |
| `--max_model_len 4096` | 4096 | Marge prompt + max_tokens. |
| `--temperature 0.6` | 0.6 | Recommandation officielle Qwen3 thinking. |
| `--no_fallback_to_source` | **OUI** | Sur les ~10% de questions où baseline échoue, on drop. Pas de mixed-style training. |
| `--chunk_size 2000` | 2000 | 1.7B 3-5× plus rapide que 14B → chunks plus gros OK. Checkpoint /~1h. |

**Coût estimé** : 30k prompts × n=4 × ~1800 tok médian = ~216M tokens output sur 1.7B. À ~5000 tok/s sur A100 → **~12h**.

Resume-able comme v7, on peut tuer/relancer la commande identique.

### Assemble v8

Après que la distillation soit complète (vérifier que tous les uids ont ≥4 samples dans le cache) :

```bash
python -m fourneurons.distill.self_distill \
    --teacher          Qwen/Qwen3-1.7B \
    --source_dataset   /scratch/data/train_v6 \
    --output_dir       /scratch/data/train_v8 \
    --cot_source_tag   self_distill_baseline \
    --n_samples        4 \
    --no_fallback_to_source \
    --phase            assemble
```

Cible attendue dans le log :
- `correct ≥ 70%` per-sample (1.7B pass rate)
- `teacher-replaced ≥ 88%` of attempted (pass@4 ≈ 91%)
- `dropped ≥ 8%` (questions trop hard pour le baseline → on les laisse)
- ~26k-28k rows total (vs 30k v6/v7).

### SFT v8 (LoRA ULTRA conservateur)

```bash
python -m fourneurons.scripts.train \
    --dataset_dir       /scratch/data/train_v8 \
    --output_dir        /scratch/checkpoints/gk_v8 \
    --final_model_dir   /scratch/checkpoints/gk_v8/adapter \
    --num_epochs        1 \
    --learning_rate     5e-5 \
    --per_device_batch_size 2 \
    --grad_accum        8 \
    --lora_r            8 \
    --lora_alpha        16 \
    --max_seq_length    4096 \
    --eval_steps        200 \
    --save_steps        200 \
    --logging_steps     20 \
    --bf16
```

**Hyperparams plus light que v7 :**

| Param | v7 | **v8** | Pourquoi v8 < v7 |
|---|---|---|---|
| `lr` | 1e-4 | **5e-5** | Updates encore plus petits |
| `lora_r` | 16 | **8** | Capacité LoRA divisée par 2 |
| `lora_alpha` | 32 | **16** | Ratio préservé (alpha/r=2) |
| `max_seq_length` | 6144 | **4096** | Self-distill CoTs ≤ 3000 tokens, 4096 suffit |

**Effective batch = 16** (pareil que v6/v7), 1 epoch, ~3-4h SFT.

### Merge + eval + push

```bash
python -m fourneurons.scripts.merge_lora \
    --adapter_dir /scratch/checkpoints/gk_v8/adapter \
    --output_dir  /scratch/checkpoints/gk_v8/vllm \
    --base_model  Qwen/Qwen3-1.7B \
    --device      cpu

mkdir -p /scratch/eval_v8

python -m fourneurons.eval.run_inference \
    --model      /scratch/checkpoints/gk_v8/vllm \
    --input      validation_samples/general_knowledge_dev_small.jsonl \
    --output     /scratch/eval_v8/devsmall_v8_generations.jsonl \
    --n 1 --max_tokens 4096 --max_model_len 4096

python -m fourneurons.eval.report_by_bucket \
    --generations /scratch/eval_v8/devsmall_v8_generations.jsonl \
    --output      /scratch/eval_v8/devsmall_v8_report.json

# Length sanity check
python -c "
import json, statistics
chars = [len(json.loads(l)['completions'][0]) for l in open('/scratch/eval_v8/devsmall_v8_generations.jsonl')]
chars.sort()
n = len(chars)
print(f'v8 dev_small: n={n}, median={statistics.median(chars):.0f}, p10={chars[n//10]}, p90={chars[int(n*0.9)]}')
"
```

Si dev_small ≥ v6 (0.62) ET médiane chars ≥ 2500 : push.

```bash
hf upload cs-552-2026-4neurons/general_knowledge_model \
    /scratch/checkpoints/gk_v8/vllm \
    . \
    --commit-message "v8: self-distill 1.7B baseline + tiny LoRA r=8 (preserve native reasoning, add boxed format)"
```

## Coût total v8

| Étape | Volume | Temps A100 |
|---|---:|---:|
| Self-distill 1.7B (30k × n=4 × ~1800 tok) | ~216M tokens | **~10-14h** |
| Assemble | inline | <3 min |
| SFT v8 (LoRA r=8, max_seq=4096) | ~26k train rows × 1 epoch | **~2-3h** |
| Merge + eval | | ~15 min |
| **Total** | | **~13-17h** |

Plus rapide que v7 (le 1.7B est 3-5× plus rapide que le 14B-AWQ en génération).

## Plan B (si v8 régresse aussi)

v8 est notre dernière tentative SFT. Si v8 < v6 sur dev_small :

1. **Push v6 final** (notre meilleur résultat vérifié)
2. **Documenter v7 + v8 honnêtement** dans le report comme « ablations qui ont confirmé que le SFT au-delà de v6 dégrade plus qu'il n'aide »
3. **Optionnellement** : tester un v6-bis avec exactly v6 mais SFT plus court (0.5 epoch au lieu de 1) pour voir si v6 lui-même est over-trained → peut-être un easy win en cas de désespoir.

## Notes pour le report

Le parcours v1 → v8 raconte une démarche scientifique honnête :

1. **v1** : Goodhart sur in-distribution dev. CI baisse → on découvre qu'optimiser un dev biaisé est une erreur.
2. **v5** : nouveau dev OOD (qu'on découvrira plus tard biaisé par leakage), dataset propre (strict CoT, distillation, distracteurs typés). Premier modèle qui bat le baseline sur CI publique (+0.12).
3. **v6** : ablation profondeur CoT (re-distill avec long-reasoning prompts sur sources hard). +0.04 CI vs v5. **Notre best stable.**
4. **v7** : tentative ambitieuse de combiner 14B knowledge + thinking style. Régression dev_small (−0.05). Diagnostic : surface imitation du 14B-thinking par le 1.7B → sur-raisonnement plausible mais faux. Échec instructif.
5. **v8** : retour sur safe — self-distill du baseline lui-même pour apprendre **uniquement le format**, sans toucher au reasoning natif. Test de l'hypothèse : « le coût caché du SFT vient du gap de capacité teacher/student ».

Si v8 > v6 : on a démontré qu'une distillation à capacité égale est l'approche correcte pour les small models.
Si v8 ≈ v6 : on a confirmé que v6 capture déjà l'essentiel et que tout SFT additionnel a un coût caché.
Si v8 < v6 : on a confirmé que tout SFT au-delà d'un certain point dégrade plus qu'il n'aide. Le report devient une étude de cas honnête sur les limites du distillation-based SFT pour les small models knowledge-heavy.

Dans tous les cas, le report sera solide. La science est dans la rigueur du diagnostic, pas dans le score final.
