import json
import os
import re

MEMORY_FILE = os.path.join(os.path.dirname(__file__), 'episodes.json')


def _normalize_text(text):
    return str(text).lower().replace('_', ' ').strip()


def _tokenize(text):
    return set(re.findall(r'\w+', _normalize_text(text)))


def _normalize_objects(objects):
    if objects is None:
        return []
    return [_normalize_text(o) for o in objects if o is not None]


def _object_tokens(objects):
    tokens = set()
    for obj in _normalize_objects(objects):
        tokens.add(obj)
        tokens |= _tokenize(obj)
    return tokens


def _scene_tokens(task=None, objects=None):
    tokens = set()
    if task:
        task_norm = _normalize_text(task)
        tokens.add(task_norm)
        tokens |= _tokenize(task_norm)
    tokens |= _object_tokens(objects)
    return tokens


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _prepare_episode(episode):
    """Attach scene_tags so old episodes remain compatible and new ones are richer."""
    ep = dict(episode)
    ep['scene_tags'] = sorted(_scene_tokens(ep.get('task'), ep.get('objects', [])))
    return ep


def load_memory(memory_file=MEMORY_FILE):
    memory_file = memory_file or MEMORY_FILE
    if not os.path.exists(memory_file):
        return []

    with open(memory_file, 'r', encoding='utf-8') as f:
        episodes = json.load(f)

    return [_prepare_episode(ep) for ep in episodes]


def save_memory(episodes, memory_file=MEMORY_FILE):
    memory_file = memory_file or MEMORY_FILE
    prepared = [_prepare_episode(ep) for ep in episodes]
    with open(memory_file, 'w', encoding='utf-8') as f:
        json.dump(prepared, f, indent=2, ensure_ascii=False)


def add_episode(episode, memory_file=MEMORY_FILE):
    episodes = load_memory(memory_file)
    episodes.append(_prepare_episode(episode))
    save_memory(episodes, memory_file)


def _planner_score(instruction, objects, episode, task=None):
    inst_score = _jaccard(
        _tokenize(instruction),
        _tokenize(episode.get('instruction', ''))
    )
    obj_score = _jaccard(
        _object_tokens(objects),
        _object_tokens(episode.get('objects', []))
    )
    task_score = 0.0
    if task and episode.get('task'):
        task_score = 1.0 if _normalize_text(task) == _normalize_text(episode.get('task')) else 0.0

    # weighted score: instruction matters most, then scene/object overlap
    return 0.55 * inst_score + 0.30 * obj_score + 0.15 * task_score


def _affordance_score(query, objects, affordance_query, episode, task=None):
    query_score = _jaccard(
        _tokenize(query),
        _tokenize(affordance_query)
    )
    scene_score = _jaccard(
        _scene_tokens(task, objects),
        set(episode.get('scene_tags', []))
    )
    task_score = 0.0
    if task and episode.get('task'):
        task_score = 1.0 if _normalize_text(task) == _normalize_text(episode.get('task')) else 0.0

    # affordance query text still matters most, but we now also use scene/task info
    return 0.60 * query_score + 0.25 * scene_score + 0.15 * task_score


def retrieve_affordance_hint(query, objects=None, task=None, top_k=1,
                             min_score=0.15, max_score=0.99,
                             memory_file=MEMORY_FILE):
    """
    Retrieve past successful affordance-map code for a similar query + scene.
    Deduplicates repeated identical code blocks.
    """
    episodes = load_memory(memory_file)
    scored = []

    for ep in episodes:
        if ep.get('outcome') != 'success':
            continue

        for amap in ep.get('affordance_maps', []):
            score = _affordance_score(
                query=query,
                objects=objects or [],
                affordance_query=amap.get('query', ''),
                episode=ep,
                task=task,
            )
            if min_score <= score < max_score:
                scored.append((score, ep, amap))

    scored.sort(key=lambda x: x[0], reverse=True)

    deduped = []
    seen_code = set()
    for score, ep, amap in scored:
        code = amap.get('code', '').strip()
        if not code or code in seen_code:
            continue
        seen_code.add(code)
        deduped.append({
            'score': score,
            'task': ep.get('task'),
            'instruction': ep.get('instruction'),
            'objects': ep.get('objects', []),
            'query': amap.get('query', ''),
            'code': code,
        })
        if len(deduped) >= top_k:
            break

    return deduped


def retrieve(instruction, objects, task=None, top_k=1, outcome_filter='success',
             min_score=0.1, max_score=0.99, memory_file=MEMORY_FILE):
    """
    Return top_k most similar episodes using instruction + object/scene overlap.
    Excludes near-exact matches by the max_score threshold.
    """
    episodes = load_memory(memory_file)
    candidates = [ep for ep in episodes if ep.get('outcome') == outcome_filter]

    scored = []
    for ep in candidates:
        score = _planner_score(
            instruction=instruction,
            objects=objects or [],
            episode=ep,
            task=task,
        )
        if min_score <= score < max_score:
            scored.append((score, ep))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [ep for _, ep in scored[:top_k]]


def retrieve_failures(instruction, objects, task=None, top_k=1,
                      min_score=0.1, max_score=0.99,
                      memory_file=MEMORY_FILE):
    return retrieve(
        instruction=instruction,
        objects=objects,
        task=task,
        top_k=top_k,
        outcome_filter='failure',
        min_score=min_score,
        max_score=max_score,
        memory_file=memory_file,
    )


def build_memory_from_logs(logs_dir, memory_file=MEMORY_FILE):
    """Parse all logs and write to memory file. Overwrites existing memory."""
    from log_parser import parse_logs_dir

    episodes = parse_logs_dir(logs_dir)
    episodes = [_prepare_episode(ep) for ep in episodes]
    save_memory(episodes, memory_file)
    print(f"Saved {len(episodes)} episodes to {memory_file}")
    return episodes


if __name__ == '__main__':
    import sys

    logs_dir = sys.argv[1] if len(sys.argv) > 1 else 'logs'
    episodes = build_memory_from_logs(logs_dir)

    outcome_counts = {}
    for ep in episodes:
        outcome_counts[ep['outcome']] = outcome_counts.get(ep['outcome'], 0) + 1
    print("Outcome breakdown:", outcome_counts)