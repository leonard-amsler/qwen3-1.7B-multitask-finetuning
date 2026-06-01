from collections import defaultdict

def _per_language_metrics(items, method):
    from .score_wandb import _gold, extract_benchmark_answer, is_correct_benchmark_answer
    
    language_to_correct = defaultdict(int)
    language_to_total = defaultdict(int)

    for item in items:
        language = item.get("lang")
        reference = _gold(item)
        completions = item.get("completions", [])
        for comp in completions:
            comp_text = str(comp)
            extracted = extract_benchmark_answer(comp_text, method, reference)
            correct = is_correct_benchmark_answer(extracted, reference, method)
            language_to_correct[language] += int(correct)
            language_to_total[language] += 1

    language_metrics = {}
    for language in language_to_total:
        total = language_to_total[language]
        correct = language_to_correct[language]
        accuracy = (correct / total * 100) if total > 0 else 0.0
        language_metrics[language] = {
            "correct": correct,
            "total": total,
            "accuracy_pct": accuracy,
        }

    return {"per_language_metrics": language_metrics}

def _per_subject_metrics(items, method):
    from .score_wandb import _gold, extract_benchmark_answer, is_correct_benchmark_answer

    topic_to_correct = defaultdict(int)
    topic_to_total = defaultdict(int)

    for item in items:
        topic = item.get("subject")
        reference = _gold(item)
        completions = item.get("completions", [])
        for comp in completions:
            comp_text = str(comp)
            extracted = extract_benchmark_answer(comp_text, method, reference)
            correct = is_correct_benchmark_answer(extracted, reference, method)
            topic_to_correct[topic] += int(correct)
            topic_to_total[topic] += 1

    topic_metrics = {}
    for topic in topic_to_total:
        total = topic_to_total[topic]
        correct = topic_to_correct[topic]
        accuracy = (correct / total * 100) if total > 0 else 0.0
        topic_metrics[topic] = {
            "correct": correct,
            "total": total,
            "accuracy_pct": accuracy,
        }

    return {"per_subject_metrics": topic_metrics}

def compute_multilingual_metrics(items, method):
    """
    Compute additional metrics specific to the multilingual benchmark, such as per-language performance.
    """
    return {
        **_per_language_metrics(items, method),
        **_per_subject_metrics(items, method),
    }