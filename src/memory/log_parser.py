import re
import os
import json


def strip_ansi(text):
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


def parse_log(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    episode = {
        'task': None,
        'instruction': None,
        'objects': [],
        'planner_code': None,
        'affordance_maps': [],   # list of {query, code}
        'outcome': 'stuck',
        'reward': None,
        'log_file': filepath,
    }

    planner_code_lines = []
    in_planner_block = False
    in_affordance_block = False
    affordance_query = None
    affordance_code_lines = []

    for i, line in enumerate(lines):
        line_clean = strip_ansi(line).rstrip()

        if line_clean.startswith('Task:'):
            episode['task'] = line_clean.split('Task:', 1)[1].strip()

        elif line_clean.startswith('Instruction:'):
            episode['instruction'] = line_clean.split('Instruction:', 1)[1].strip()

        elif '## "planner" generated code' in line_clean:
            in_planner_block = True
            in_affordance_block = False
            planner_code_lines = []

        elif '## "get_affordance_map" generated code' in line_clean:
            # save previous affordance block if any
            if in_affordance_block and affordance_query and affordance_code_lines:
                episode['affordance_maps'].append({
                    'query': affordance_query,
                    'code': '\n'.join(affordance_code_lines).strip(),
                })
            in_affordance_block = True
            in_planner_block = False
            affordance_query = None
            affordance_code_lines = []

        elif in_planner_block:
            if line_clean.startswith('## context:'):
                match = re.search(r"objects = (\[.*?\])", line_clean)
                if match:
                    try:
                        episode['objects'] = json.loads(match.group(1).replace("'", '"'))
                    except Exception:
                        pass
            elif line_clean.startswith('## "') or line_clean.startswith('*** OpenAI'):
                in_planner_block = False
                episode['planner_code'] = '\n'.join(planner_code_lines).strip()
            elif not line_clean.startswith('####'):
                planner_code_lines.append(line_clean)

        elif in_affordance_block:
            if line_clean.startswith('## "') or line_clean.startswith('*** OpenAI'):
                if affordance_query and affordance_code_lines:
                    episode['affordance_maps'].append({
                        'query': affordance_query,
                        'code': '\n'.join(affordance_code_lines).strip(),
                    })
                in_affordance_block = False
                affordance_query = None
                affordance_code_lines = []
            elif line_clean.startswith('# Query:'):
                affordance_query = line_clean.split('# Query:', 1)[1].strip()
            elif not line_clean.startswith('####'):
                affordance_code_lines.append(line_clean)

        if line_clean.startswith('Reward:'):
            try:
                reward = float(line_clean.split('Reward:', 1)[1].strip())
                episode['reward'] = reward
                episode['outcome'] = 'success' if reward == 1.0 else 'failure'
            except ValueError:
                pass

    if in_planner_block and planner_code_lines:
        episode['planner_code'] = '\n'.join(planner_code_lines).strip()
    if in_affordance_block and affordance_query and affordance_code_lines:
        episode['affordance_maps'].append({
            'query': affordance_query,
            'code': '\n'.join(affordance_code_lines).strip(),
        })

    return episode


def parse_logs_dir(logs_dir):
    episodes = []
    for root, dirs, files in os.walk(logs_dir):
        for fname in files:
            if fname.endswith('.txt'):
                fpath = os.path.join(root, fname)
                ep = parse_log(fpath)
                if ep['instruction'] is not None:
                    episodes.append(ep)
    return episodes


if __name__ == '__main__':
    import sys
    logs_dir = sys.argv[1] if len(sys.argv) > 1 else 'logs'
    episodes = parse_logs_dir(logs_dir)
    print(f"Parsed {len(episodes)} episodes")
    for ep in episodes:
        print(f"  [{ep['outcome']:7s}] {ep['task']} | {ep['instruction'][:60]}")
