# General Knowledge — Plan v10 (battre v6 sur tous les buckets)

> Date : 2026-05-29. Objectif : un modèle **strictement ≥ v6** partout, avec un
> gain ciblé sur les buckets où v6 plafonne.

## 1. Comprendre POURQUOI v6 est notre meilleur modèle

v6 utilise la distillation **« rationale-given-answer »** (`fourneurons/distill/prompts.py`) :

- On **donne la bonne réponse** au teacher Qwen3-14B et on lui demande
  *« explique pourquoi X est correct »*, en mode **non-thinking**.
- Conséquence : le CoT est **toujours cohérent avec le gold** (impossible de
  raisonner vers une mauvaise réponse). Label parfait + teacher intelligent (14B)
  + format propre + LoRA fort (r=64).

C'est structurellement supérieur aux alternatives qu'on a testées :

| Modèle | Teacher | Mode | Label | Résultat | Pourquoi |
|---|---|---|---|---|---|
| **v6** | 14B | non-think, *given answer* | parfait | **0.62 / 0.39 CI** | référence |
| v7 | 14B | thinking, *blind* | bruité | régression | imitation de surface (style sans raisonnement) |
| v8 | 1.7B | thinking, *blind* | bruité | 0.39 (réel 0.52) | plafonné par le 1.7B + format cassé (LoRA faible) |

## 2. La FAIBLESSE de v6 (là où v10 attaque)

Rapport par bucket de v6/v-courant sur dev_small :

| Bucket | pass@1 | Type de question |
|---|---:|---|
| boolq (2 options) | 0.675 | justifier un fait binaire |
| commonsense | 0.656 | justifier un fait |
| **mmlu_pro** | **0.383** | **dériver, calculer, multi-étapes** |
| **stem** | **0.413** | **dériver** |
| **humanities** | **0.313** | **éliminer des options proches** |
| **6-10 options** | **0.357** | **éliminer beaucoup de distracteurs** |

**Le pattern est net :** v6 excelle à *justifier une réponse* (ce qu'on lui a
appris) mais plafonne quand il faut *trouver* la réponse en **dérivant** et en
**éliminant des options** — exactement ce que la distillation v6 lui interdit
d'apprendre (les prompts disent *« do not compare with other options »*).

**Mismatch train/test :**
- Entraînement v6 : « voici la réponse X, justifie-la » (option-agnostique).
- Test (CI, MMLU-Pro) : « voici 4-10 options, trouve la bonne » (il faut comparer/éliminer).

Le modèle n'a **jamais vu** de raisonnement d'élimination. Sur une question dure
il produit une justification confiante de ce vers quoi il penche — souvent faux.

## 3. Stratégie v10 : re-distillation CONTRASTIVE sur le STEM (option-agnostique)

### Contrainte d'architecture (importante)

Les CoTs de v6 sont **option-agnostiques** : ils expliquent le gold *par son
contenu*, jamais par lettre. C'est ce qui permet de **réutiliser** la même CoT
sur la question originale ET ses variantes augmentées (lettres mélangées,
options ajoutées). Un prompt « élimine l'option B/C/D » serait *option-spécifique*
→ casserait cette réutilisation et reviendrait au piège v7 (thinking aveugle).

### Le prompt contrastif (v10)

On garde **tout** ce qui fait marcher v6 :
- **réponse donnée** au teacher → label parfait, raisonnement correct par construction ;
- **non-thinking** → style déclaratif propre, pas de hedging exploratoire (≠ v7) ;
- **option-agnostique** → réutilisable sur les variantes.

Et on ajoute **un seul élément** : après avoir dérivé la réponse, le teacher doit
**nommer et réfuter le raisonnement erroné le plus tentant**. Le student voit donc
du raisonnement *discriminant* (le skill manquant) sans qu'il soit lié à une
liste d'options précise.

Implémenté dans `prompts.py::build_user_prompt_contrastive` (style `contrastive`),
routé par `distill.py --reasoning_style contrastive`.

### Composition du dataset v10

```
train_v10 = build_train avec caches en last-wins :
   distilled_cot_v5.jsonl  (short, sources faciles)
   distilled_cot_v6_long.jsonl  (long, STEM)
   distilled_cot_v10_contrastive.jsonl  (contrastif, STEM)  <- override les uids STEM
```

`build_train` charge déjà plusieurs caches en **last-wins** (cf. PLAN_V6 §3) :
le cache v10 (dernier) remplace les CoTs STEM de v6 par leur version contrastive.
Tous les autres uids (boolq, commonsense, etc.) gardent leur CoT v6 **inchangé**
→ aucune régression possible sur les buckets forts.

## 4. Implémentation (faite)

### 4a. `fourneurons/distill/prompts.py`
Ajout de `build_user_prompt_contrastive(question, gold_text)` + route `style="contrastive"`
dans `build_messages`. Option-agnostique, donné-réponse, non-thinking, 10-18 phrases.

### 4b. `fourneurons/distill/distill.py`
Ajout du flag `--reasoning_style {auto,short,long,contrastive}`. `auto` = comportement
v6. `contrastive` force le prompt contrastif pour toutes les sources du run.
`min_chars` passe automatiquement à 1000 pour `contrastive`.

### 4c. Commande de distillation v10 (STEM uniquement)

```bash
python -m fourneurons.distill.distill \
    --teacher Qwen/Qwen3-14B-AWQ \
    --quantization awq_marlin \
    --output  /scratch/data/distilled_cot_v10_contrastive.jsonl \
    --sources mmlu mmlu_pro_cot \
    --reasoning_style contrastive \
    --max_tokens 2048 \
    --max_model_len 4096
```

- Volume : ~7 200 questions (mmlu validation ~1 500 + mmlu_pro_cot ~5 670).
- ~600-800 tokens output médian → **~2-3h** sur A100.
- Smoke test d'abord : ajouter `--limit 50` et inspecter le cache + `.failed.jsonl`.

### 4d. Re-build train_v10 (last-wins)

```bash
python -m fourneurons.data.build_train \
    --output_dir          /scratch/data/train_v10 \
    --total               30000 \
    --max_variants        1 \
    --distilled_cot_cache /scratch/data/distilled_cot_v5.jsonl \
                          /scratch/data/distilled_cot_v6_long.jsonl \
                          /scratch/data/distilled_cot_v10_contrastive.jsonl \
    --seed                42
```

### 4e. SFT v10 (réglages v6 exacts, pour isoler l'effet des données)

```bash
python -m fourneurons.scripts.train \
    --dataset_dir       /scratch/data/train_v10 \
    --output_dir        /scratch/checkpoints/gk_v10 \
    --final_model_dir   /scratch/checkpoints/gk_v10/adapter \
    --num_epochs        1 \
    --learning_rate     2e-4 \
    --lora_r            64 --lora_alpha 128 \
    --max_seq_length    4096 \
    --bf16
```

## 5. Pourquoi v10 devrait battre v6 « dans tous les cas »

- **Buckets forts** : CoTs v6 conservées à 100 % (last-wins ne touche que STEM)
  → aucune régression attendue sur boolq / commonsense / social.
- **Buckets faibles (STEM, mmlu_pro)** : CoTs enrichies de raisonnement
  **discriminant** (réfutation du piège) → attaque directe du déficit diagnostiqué.
- **Format** : `\boxed{}` à 100 % (builder v6 inchangé, LoRA r=64).
- **Risque maîtrisé** : on reste dans l'architecture v6 prouvée (donné-réponse,
  non-thinking, option-agnostique). Pas de blind-solve, pas de thinking → pas
  d'imitation de surface type v7, pas de plafond 1.7B type v8.

## 6. Critères de succès

- dev_small global **> 0.62** (au moins +2 pts).
- mmlu_pro **> 0.45** (vs 0.38), stem **> 0.48** (vs 0.41) : c'est la cible.
- Aucun bucket fort ne régresse de plus de 1 pt.
- CI : viser **> 0.42** (vs 0.39) — la CI étant MMLU-Pro-like, c'est le bucket
  qu'on renforce directement.

## 7. Ordre d'exécution recommandé

1. **v9 d'abord** (cheap, ~2.5h, données prêtes) : confirme l'hypothèse format,
   dé-risque le LoRA r=64 sur `\boxed{}`.
2. **Pendant la distillation solve_verified** (GPU long), v9 tourne / on analyse.
3. **v10** : merge + SFT + eval. Décision finale v6 vs v10 sur dev_small + CI.
