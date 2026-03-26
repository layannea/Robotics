import json
import os
import re

MEMORY_FILE = os.path.join(os.path.dirname(__file__), 'episodes.json')


def _tokenize(text):
    return set(re.findall(r'\w+', text.lower()))


def _similarity(instruction_a, objects_a, instruction_b, objects_b):
    """Jaccard similarity over instruction words + object names."""
    tokens_a = _tokenize(instruction_a) | set(o.lower() for o in objects_a)
    tokens_b = _tokenize(instruction_b) | set(o.lower() for o in objects_b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def load_memory(memory_file=MEMORY_FILE):
    if not os.path.exists(memory_file):
        return []
    with open(memory_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_memory(episodes, memory_file=MEMORY_FILE):
    with open(memory_file, 'w', encoding='utf-8') as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)


def add_episode(episode, memory_file=MEMORY_FILE):
    episodes = load_memory(memory_file)
    episodes.append(episode)
    save_memory(episodes, memory_file)


def retrieve_affordance_hint(query, top_k=1, min_score=0.15, max_score=0.99,
                             memory_file=MEMORY_FILE):
    """Retrieve a past successful affordance map code for a similar query."""
    episodes = load_memory(memory_file)
    scored = []
    for ep in episodes:
        if ep.get('outcome') != 'success':
            continue
        for amap in ep.get('affordance_maps', []):
            score = _similarity(query, [], amap.get('query', ''), [])
            if min_score <= score < max_score:
                scored.append((score, amap['query'], amap['code']))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{'query': q, 'code': c} for _, q, c in scored[:top_k]]


def retrieve(instruction, objects, top_k=1, outcome_filter='success',
             min_score=0.1, max_score=0.99, memory_file=MEMORY_FILE):
    """Return top_k most similar successful episodes.

    Excludes exact matches (score >= max_score) to avoid confusing the LLM
    when the same instruction is already in memory.
    Only injects when score >= min_score (meaningful similarity).
    """
    episodes = load_memory(memory_file)
    candidates = [ep for ep in episodes if ep.get('outcome') == outcome_filter]

    scored = []
    for ep in candidates:
        score = _similarity(instruction, objects,
                            ep.get('instruction', ''), ep.get('objects', []))
        if min_score <= score < max_score:
            scored.append((score, ep))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [ep for _, ep in scored[:top_k]]


def retrieve_failures(instruction, objects, top_k=1, min_score=0.1, max_score=0.99,
                      memory_file=MEMORY_FILE):
    """Return top_k most similar failed episodes for failure warning injection."""
    return retrieve(instruction, objects, top_k=top_k, outcome_filter='failure',
                    min_score=min_score, max_score=max_score, memory_file=memory_file)


def build_memory_from_logs(logs_dir, memory_file=MEMORY_FILE):
    """Parse all logs and write to memory file. Overwrites existing memory."""
    from log_parser import parse_logs_dir
    episodes = parse_logs_dir(logs_dir)
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
