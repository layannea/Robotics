
import openai
import ast
import re
from time import sleep
from openai.error import RateLimitError, APIConnectionError, APIError, ServiceUnavailableError, Timeout
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from utils import load_prompt, DynamicObservation, IterableDynamicObservation
from planning_verifier import verify_and_repair_generated_code
import time
from LLM_cache import DiskCache

class LMP:
    """Language Model Program (LMP), adopted from Code as Policies."""
    def __init__(self, name, cfg, fixed_vars, variable_vars, debug=False, env='rlbench'):
        self._name = name
        self._cfg = cfg
        self._debug = debug
        self._base_prompt = load_prompt(f"{env}/{self._cfg['prompt_fname']}.txt")
        self._stop_tokens = list(self._cfg['stop'])
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ''
        self._context = None
        self._feedback_context = ''
        self._planning_verifier_enabled = True
        self._cache = DiskCache(load_cache=self._cfg['load_cache'])

    def clear_exec_hist(self):
        self.exec_hist = ''

    def set_feedback_context(self, feedback_context):
        self._feedback_context = feedback_context or ''

    def clear_feedback_context(self):
        self._feedback_context = ''

    def set_planning_verifier_enabled(self, enabled):
        self._planning_verifier_enabled = bool(enabled)

    def build_prompt(self, query):
        if len(self._variable_vars) > 0:
            variable_vars_imports_str = f"from utils import {', '.join(self._variable_vars.keys())}"
        else:
            variable_vars_imports_str = ''
        prompt = self._base_prompt.replace('{variable_vars_imports}', variable_vars_imports_str)

        if self._cfg['maintain_session'] and self.exec_hist != '':
            prompt += f'\n{self.exec_hist}'
        
        prompt += '\n'  # separate prompted examples with the query part

        if self._cfg['include_context']:
            assert self._context is not None, 'context is None'
            prompt += f'\n{self._context}'

        if self._feedback_context:
            prompt += f'\n{self._feedback_context}'

        user_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'
        prompt += f'\n{user_query}'

        return prompt, user_query

    def _is_chat_model(self, model_name):
        if model_name == 'gpt-3.5-turbo-instruct':
            return False
        chat_model_hints = ['gpt-3.5', 'gpt-4', 'gpt-5', 'codex', 'o1', 'o3', 'o4']
        return any(chat_model in model_name for chat_model in chat_model_hints)

    def _build_chat_messages(self, prompt):
        new_query = '# Query:' + prompt.split('# Query:')[-1]
        context_block = ''.join(prompt.split('# Query:')[:-1]).strip()
        user1 = (
            "I would like you to help me write Python code to control a robot arm operating in a tabletop "
            "environment. Please complete the code every time when I give you a new query. Pay attention "
            "to appeared patterns in the given context code. Be thorough and thoughtful in your code. "
            "Do not include any import statement. Do not repeat my question. Do not provide any text "
            "explanation. Do not ask clarifying questions. Make the most reasonable assumption from the "
            "existing examples and object list, and return executable Python code only. I will first give "
            "you the context of the code below:\n\n```\n"
            f"{context_block}\n"
            "```\n\n"
            "Note that x is back to front, y is left to right, and z is bottom to up."
        )
        assistant1 = 'Understood. I will reply with executable Python code only.'
        user2 = new_query
        # handle given context (this was written originally for completion endpoint)
        object_context_matches = list(re.finditer(r'^objects\s*=\s*\[[^\n]*\]$', user1, flags=re.MULTILINE))
        if object_context_matches:
            match = object_context_matches[-1]
            obj_context = match.group(0)
            user1 = user1[:match.start()] + user1[match.end():]
            user1 = re.sub(r'\n{3,}', '\n\n', user1).strip()
            user2 = obj_context.strip() + '\n' + user2
        return [
            {
                "role": "system",
                "content": (
                    "You are a careful coding assistant for robotic manipulation. "
                    "Return only raw Python code that follows the examples. "
                    "Never return markdown fences, natural language, or clarifying questions."
                ),
            },
            {"role": "user", "content": user1},
            {"role": "assistant", "content": assistant1},
            {"role": "user", "content": user2},
        ]

    def _normalize_code_response(self, response_text):
        response_text = response_text.strip()
        if '```' in response_text:
            code_blocks = response_text.split('```')
            for block in code_blocks:
                stripped_block = block.strip()
                if not stripped_block:
                    continue
                if stripped_block.startswith('python'):
                    stripped_block = stripped_block[len('python'):].lstrip()
                if self._is_valid_python(stripped_block):
                    return stripped_block.strip()
        if response_text.startswith('python\n'):
            response_text = response_text[len('python\n'):]
        return response_text.strip()

    def _is_valid_python(self, code_str):
        if code_str.strip() == '':
            return False
        try:
            ast.parse(code_str)
            return True
        except SyntaxError:
            return False

    def _chat_api_call(self, **kwargs):
        prompt = kwargs.pop('prompt')
        messages = self._build_chat_messages(prompt)
        kwargs['messages'] = messages
        max_attempts = 3

        for attempt in range(max_attempts):
            cache_key = dict(kwargs)
            if cache_key in self._cache:
                print('(using cache)', end=' ')
                ret = self._cache[cache_key]
            else:
                ret = openai.ChatCompletion.create(**kwargs)['choices'][0]['message']['content']
                self._cache[cache_key] = ret

            normalized_ret = self._normalize_code_response(ret)
            if self._is_valid_python(normalized_ret):
                return normalized_ret

            if attempt == max_attempts - 1:
                raise ValueError(
                    f'{self._name} returned non-Python output after {max_attempts} attempts:\n{ret}'
                )

            kwargs['messages'] = kwargs['messages'] + [
                {"role": "assistant", "content": ret},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid Python code. "
                        "Reply again with executable Python code only. "
                        "Do not include markdown, explanations, or questions."
                    ),
                },
            ]

    def _cached_api_call(self, **kwargs):
        # check whether completion endpoint or chat endpoint is used
        if self._is_chat_model(kwargs['model']):
            return self._chat_api_call(**kwargs)
        else:
            if kwargs in self._cache:
                print('(using cache)', end=' ')
                return self._cache[kwargs]
            else:
                ret = openai.Completion.create(**kwargs)['choices'][0]['text'].strip()
                self._cache[kwargs] = ret
                return ret

    def __call__(self, query, **kwargs):
        prompt, user_query = self.build_prompt(query)

        start_time = time.time()
        retry_count = 0
        while True:
            try:
                code_str = self._cached_api_call(
                    prompt=prompt,
                    stop=self._stop_tokens,
                    temperature=self._cfg['temperature'],
                    model=self._cfg['model'],
                    max_tokens=self._cfg['max_tokens']
                )
                break
            except (RateLimitError, APIConnectionError, APIError, ServiceUnavailableError, Timeout) as e:
                retry_count += 1
                retry_delay = min(30, 3 * retry_count)
                print(f'OpenAI API got err {e}')
                print(f'Retrying after {retry_delay}s.')
                sleep(retry_delay)
        print(f'*** OpenAI API call took {time.time() - start_time:.2f}s ***')

        if self._planning_verifier_enabled:
            verifier_result = verify_and_repair_generated_code(
                lmp_name=self._name,
                query=query,
                code=code_str,
                objects=self._context_objects(),
            )
            if verifier_result.changed:
                for diagnostic in verifier_result.diagnostics:
                    print(f'[planning_verifier] {diagnostic}')
                code_str = verifier_result.code

        if self._cfg['include_context']:
            assert self._context is not None, 'context is None'
            to_exec = f'{self._context}\n{code_str}'
            to_log = f'{self._context}\n{user_query}\n{code_str}'
        else:
            to_exec = code_str
            to_log = f'{user_query}\n{to_exec}'

        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())

        if self._cfg['include_context']:
            print('#'*40 + f'\n## "{self._name}" generated code\n' + f'## context: "{self._context}"\n' + '#'*40 + f'\n{to_log_pretty}\n')
        else:
            print('#'*40 + f'\n## "{self._name}" generated code\n' + '#'*40 + f'\n{to_log_pretty}\n')

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs

        # return function instead of executing it so we can replan using latest obs（do not do this for high-level UIs)
        if not self._name in ['composer', 'planner']:
            to_exec = 'def ret_val():\n' + to_exec.replace('ret_val = ', 'return ')
            to_exec = to_exec.replace('\n', '\n    ')

        if self._debug:
            # only "execute" function performs actions in environment, so we comment it out
            action_str = ['execute(']
            try:
                for s in action_str:
                    exec_safe(to_exec.replace(s, f'# {s}'), gvars, lvars)
            except Exception as e:
                print(f'Error: {e}')
                import pdb ; pdb.set_trace()
        else:
            exec_safe(to_exec, gvars, lvars)

        self.exec_hist += f'\n{to_log.strip()}'

        if self._cfg['maintain_session']:
            self._variable_vars.update(lvars)

        if self._cfg['has_return']:
            if self._name == 'parse_query_obj':
                ret_fn = self._wrap_parse_query_obj_return(
                    lvars[self._cfg['return_val_name']],
                    query,
                )
                try:
                    # there may be multiple objects returned, but we also want them to be unevaluated functions so that we can access latest obs
                    return IterableDynamicObservation(ret_fn)
                except AssertionError:
                    return DynamicObservation(ret_fn)
            return lvars[self._cfg['return_val_name']]

    def _wrap_parse_query_obj_return(self, ret_fn, query):
        if not callable(ret_fn):
            return ret_fn

        def repaired_ret_fn():
            try:
                parsed = ret_fn()
                if parsed is not None:
                    return parsed
                print(f'[LMP.py] parse_query_obj returned None for {query!r}; trying canonical object repair.')
            except Exception as exc:
                if 'Unknown object name' not in str(exc):
                    raise
                print(f'[LMP.py] parse_query_obj failed for {query!r}: {exc}; trying canonical object repair.')

            repaired_name = self._canonicalize_query_object(query)
            if repaired_name is None:
                return None
            print(f'[LMP.py] parse_query_obj repair: {query!r} -> {repaired_name!r}')
            return self._variable_vars['detect'](repaired_name)

        return repaired_ret_fn

    def _context_objects(self):
        if not self._context:
            return []
        match = re.search(r'objects\s*=\s*(\[[^\n]*\])', self._context)
        if not match:
            return []
        try:
            objects = ast.literal_eval(match.group(1))
        except Exception:
            return []
        return objects if isinstance(objects, list) else []

    def _canonicalize_query_object(self, query):
        detect = self._variable_vars.get('detect')
        if detect is None:
            return None
        query_norm = self._normalize_object_text(query)
        if query_norm in {'ee', 'endeffector', 'end effector', 'end effector', 'gripper', 'hand'}:
            return 'gripper'
        if query_norm in {'table', 'desk', 'workstation', 'workspace'}:
            return 'table'

        objects = self._context_objects()
        if not objects:
            return None

        object_norms = [(obj, self._normalize_object_text(obj)) for obj in objects]
        for obj, obj_norm in object_norms:
            if query_norm == obj_norm:
                return obj

        query_tokens = self._object_tokens(query_norm)
        candidates = []
        for obj, obj_norm in object_norms:
            obj_tokens = self._object_tokens(obj_norm)
            if not obj_tokens:
                continue
            if obj_tokens.issubset(query_tokens):
                candidates.append((obj, len(obj_tokens)))

        if not candidates:
            candidates = self._semantic_alias_candidates(query_norm, object_norms)

        if not candidates and len(objects) == 1:
            obj = objects[0]
            obj_tokens = self._object_tokens(self._normalize_object_text(obj))
            if obj_tokens and (obj_tokens & query_tokens):
                candidates = [(obj, len(obj_tokens))]

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[1], reverse=True)
        if len(candidates) > 1 and candidates[0][1] == candidates[1][1]:
            return None
        return candidates[0][0]

    def _semantic_alias_candidates(self, query_norm, object_norms):
        alias_groups = {
            'button': {'button', 'switch'},
            'scale': {'scale', 'scales', 'weighing scale', 'weighing scales'},
            'meat': {'meat', 'steak', 'chicken'},
            'saucepan lid': {'saucepan lid', 'lid', 'saucepan lid handle', 'lid handle'},
            'umbrella': {'umbrella', 'umbrella handle', 'handle'},
            'stand': {'stand', 'umbrella stand'},
        }
        candidates = []
        for obj, obj_norm in object_norms:
            aliases = alias_groups.get(obj_norm)
            if not aliases:
                continue
            if any(self._normalize_object_text(alias) in query_norm for alias in aliases):
                candidates.append((obj, len(self._object_tokens(obj_norm))))
        return candidates

    def _normalize_object_text(self, text):
        text = str(text).lower().replace('_', ' ')
        text = re.sub(r'[^a-z0-9\s]+', ' ', text)
        text = re.sub(r'\b(the|a|an|to|of|on|in|at|from)\b', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _object_tokens(self, text):
        tokens = []
        for token in self._normalize_object_text(text).split():
            if len(token) > 3 and token.endswith('s'):
                token = token[:-1]
            tokens.append(token)
        return set(tokens)


def merge_dicts(dicts):
    return {
        k : v 
        for d in dicts
        for k, v in d.items()
    }
    

def exec_safe(code_str, gvars=None, lvars=None):
    banned_phrases = ['import', '__']
    for phrase in banned_phrases:
        assert phrase not in code_str
  
    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    try:
        exec(code_str, custom_gvars, lvars)
    except Exception as e:
        print(f'Error executing code:\n{code_str}')
        raise e
