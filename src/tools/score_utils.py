import re
import json
from typing import Callable, Iterable, List, Optional, Set


OBSERVE_PATTERN = re.compile(r'<observe>(.*?)</observe>', re.DOTALL)
THINK_PATTERN   = re.compile(r'<think>(.*?)</think>',     re.DOTALL)
ANSWER_PATTERN  = re.compile(r'<answer>(.*?)</answer>',   re.DOTALL)

FORMAT_FULL_PATTERN = re.compile(
    r'<observe>.*?</observe>\s*<think>.*?</think>\s*<answer>.*?</answer>',
    re.DOTALL,
)


def compute_format_reward(text: str) -> float:
    return 1.0 if FORMAT_FULL_PATTERN.search(text) is not None else 0.0


def extract_reasoning_span(text: str) -> str:
    obs = OBSERVE_PATTERN.search(text)
    thk = THINK_PATTERN.search(text)
    parts = []
    if obs:
        parts.append(obs.group(1))
    if thk:
        parts.append(thk.group(1))
    return " ".join(parts) if parts else text


def extract_predicted_answer(text: str) -> str:
    m = ANSWER_PATTERN.search(text)
    return m.group(1).strip() if m else text.strip()


def build_entity_matcher(entity_lexicon: Iterable[str]) -> Callable[[str], Set[str]]:
    canonical_forms = sorted(set(entity_lexicon), key=len, reverse=True)

    try:
        import ahocorasick
        A = ahocorasick.Automaton()
        for ent in canonical_forms:
            A.add_word(ent.lower(), ent)
        A.make_automaton()

        def match(text: str) -> Set[str]:
            hits: Set[str] = set()
            t = text.lower()
            for _, canonical in A.iter(t):
                hits.add(canonical)
            return hits

        return match

    except ImportError:
        # Fallback: naive substring match. Slower but has no extra dependency.
        lowered = [(ent.lower(), ent) for ent in canonical_forms]

        def match(text: str) -> Set[str]:
            t = text.lower()
            return {canonical for surf, canonical in lowered if surf in t}

        return match


def compute_entity_reward(
    pred_text: str,
    gt_entities: List[str],
    entity_matcher: Callable[[str], Set[str]],
    sim_fn: Optional[Callable[[str, str], float]] = None,
    beta: float = 0.5,
    eps: float = 1e-8,
) -> float:
    
    E_pred = entity_matcher(pred_text)
    E_gt = set(gt_entities)

    if len(E_pred) == 0 and len(E_gt) == 0:
        return 0.0

    hard_hit = len(E_pred & E_gt)

    soft_sum = 0.0
    if sim_fn is not None and len(E_gt) > 0:
        for e in (E_pred - E_gt):
            best = 0.0
            for g in E_gt:
                s = sim_fn(e, g)
                if s > best:
                    best = s
            soft_sum += best

    I_soft = hard_hit + beta * soft_sum
    dice = (2.0 * I_soft) / (len(E_pred) + len(E_gt) + eps)
    return float(min(1.0, dice))


RUBRIC_TEMPLATE = """You are a pathology expert grading a model's diagnostic reasoning.
Score the prediction against the ground truth on a scale from 0.0 to 1.0.

Grading criteria:
- Clinical accuracy of morphological observations.
- Logical consistency of the deductive chain.
- Correctness of the final diagnosis (synonyms and standard terminology variations are acceptable).

Return ONLY a single number in [0, 1], with no explanation.

Ground truth:
{gt}

Prediction:
{pred}

Score:"""


def build_semantic_judger(
    endpoint_url: str,
    model_name: str = "qwen2.5-32b-instruct",
    timeout_s: float = 30.0,
    temperature: float = 0.0,
    max_tokens: int = 8,
    concurrency: int = 8,
) -> Callable[[List[str], List[str]], List[float]]:

    from openai import OpenAI
    from concurrent.futures import ThreadPoolExecutor

    client = OpenAI(base_url=endpoint_url, api_key="EMPTY", timeout=timeout_s)
    _num_re = re.compile(r"[-+]?\d*\.?\d+")

    def _score_one(pred: str, gt: str) -> float:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user",
                           "content": RUBRIC_TEMPLATE.format(gt=gt, pred=pred)}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            out = resp.choices[0].message.content.strip()
            m = _num_re.search(out)
            if m is None:
                return 0.0
            val = float(m.group(0))
            return float(max(0.0, min(1.0, val)))
        except Exception as e:
            print(f"[semantic judge error] {e}", flush=True)
            return 0.0

    def score_batch(preds: List[str], gts: List[str]) -> List[float]:
        assert len(preds) == len(gts), "preds and gts must be aligned"
        if len(preds) == 0:
            return []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            return list(pool.map(_score_one, preds, gts))

    return score_batch


def compute_multi_granular_rewards(
    decoded_texts: List[str],
    gt_entities_list: List[List[str]],
    gt_answers_list: List[str],
    entity_matcher: Callable[[str], Set[str]],
    semantic_scorer: Callable[[List[str], List[str]], List[float]],
    alpha: float = 0.5,
    beta: float = 0.5,
    entity_sim_fn: Optional[Callable[[str, str], float]] = None,
):

    n = len(decoded_texts)
    assert len(gt_entities_list) == n and len(gt_answers_list) == n

    format_rewards: List[float] = []
    entity_rewards: List[float] = []
    pred_answers_for_judge: List[str] = []

    for text, gt_ents in zip(decoded_texts, gt_entities_list):
        # Format
        format_rewards.append(compute_format_reward(text))

        # Entity (restrict matching to reasoning span only)
        reasoning_span = extract_reasoning_span(text)
        entity_rewards.append(
            compute_entity_reward(
                reasoning_span, gt_ents, entity_matcher,
                sim_fn=entity_sim_fn, beta=beta,
            )
        )

        # Prepare <answer> content for the judge
        pred_answers_for_judge.append(extract_predicted_answer(text))

    # Semantic (batched remote call)
    semantic_rewards = semantic_scorer(pred_answers_for_judge, gt_answers_list)

    correctness = [
        f + s + alpha * e
        for f, s, e in zip(format_rewards, semantic_rewards, entity_rewards)
    ]

    reward_components = {
        'r_format':   format_rewards,
        'r_semantic': semantic_rewards,
        'r_entity':   entity_rewards,
    }
    return correctness, reward_components


def load_entity_lexicon(path: str) -> List[str]:
    with open(path, 'r') as f:
        obj = json.load(f)

    return list(obj)
