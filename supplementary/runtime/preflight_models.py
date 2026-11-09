"""Read-only pinned-weight HEAD and CPU tokenizer audit; never loads weights."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from huggingface_hub import HfApi, get_hf_file_metadata, get_token, hf_hub_download, hf_hub_url
import huggingface_hub
import requests
from transformers import AutoConfig, AutoTokenizer
import transformers
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name('model_preflight.json')
TOKEN = get_token()


def failure(exc):
    response = getattr(exc, 'response', None)
    # Exceptions may carry signed CDN query strings, so do not persist messages/URLs.
    result = {'exception_class': type(exc).__name__,
              'http_status': None if response is None else response.status_code}
    if response is not None and 'not in the authorized list' in response.headers.get('x-error-message', ''):
        result['access_reason'] = 'Account behind active token is not in this repository authorized list'
    return result


def audit(item):
    manifest_name, interface, endpoint = item
    repo, revision = endpoint['model_id'], endpoint['revision']
    row = {'manifest': manifest_name, 'interface': interface, **endpoint}
    try:
        info = HfApi().model_info(repo, revision=revision, token=TOKEN, timeout=30)
        row.update(repository_access=True, resolved_revision=info.sha,
                   revision_matches=info.sha == revision, gated=info.gated)
        names = [s.rfilename for s in info.siblings]
        weights = sorted(n for n in names if n.endswith('.safetensors') and '/' not in n)
        if not weights:
            weights = sorted(n for n in names if n.startswith('pytorch_model') and n.endswith('.bin') and '/' not in n)
        if not weights:
            row['weight_head'] = {'accessible': False, 'reason': 'No recognized root checkpoint weight file'}
        else:
            filename = weights[0]
            url = hf_hub_url(repo, filename, revision=revision)
            metadata = get_hf_file_metadata(url, token=TOKEN, timeout=30)
            # Validate the actual redirected object using HEAD, without reading any body.
            response = requests.head(metadata.location, allow_redirects=True, timeout=45)
            row['weight_head'] = {'filename': filename, 'accessible': response.status_code == 200,
                                  'http_status': response.status_code, 'size_bytes': metadata.size,
                                  'commit_hash': metadata.commit_hash,
                                  'downloaded_weight_bytes': 0}
    except Exception as exc:
        row['access_error'] = failure(exc)
        row.setdefault('weight_head', {'accessible': False, **failure(exc), 'downloaded_weight_bytes': 0})
    try:
        config_path = hf_hub_download(repo, 'config.json', revision=revision, token=TOKEN)
        raw = json.loads(Path(config_path).read_text())
        config = AutoConfig.from_pretrained(repo, revision=revision, token=TOKEN, trust_remote_code=False)
        mapping = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES if interface == 'judge' else MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
        row['config'] = {'model_type': raw.get('model_type'), 'architectures': raw.get('architectures'),
                         'declared_torch_dtype': raw.get('torch_dtype'), 'declared_dtype': raw.get('dtype'),
                         'config_class': type(config).__name__, 'auto_loader_class': mapping.get(config.model_type),
                         'text_config_model_type': raw.get('text_config', {}).get('model_type'),
                         'text_config_torch_dtype': raw.get('text_config', {}).get('torch_dtype'),
                         'id2label': raw.get('id2label')}
        if interface == 'nli':
            labels = raw.get('id2label', {})
            row['nli_canonical_labels'] = [labels.get(str(i)) for i in endpoint['canonical_indices']]
            row['nli_mapping_matches'] = [str(v).lower() for v in row['nli_canonical_labels']] == ['entailment', 'neutral', 'contradiction']
    except Exception as exc:
        row['config_error'] = failure(exc)
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, token=TOKEN, trust_remote_code=False)
        row['tokenizer'] = {'class': type(tokenizer).__name__, 'is_fast': tokenizer.is_fast,
                            'pad_token_id': tokenizer.pad_token_id, 'eos_token_id': tokenizer.eos_token_id}
        if interface == 'judge':
            encodings = {label: tokenizer.encode(' ' + label, add_special_tokens=False) for label in ['A', 'B', 'C']}
            row['label_audit'] = {'prefix': ' ', 'token_ids': encodings,
                                  'token_strings': {label: tokenizer.convert_ids_to_tokens(ids) for label, ids in encodings.items()},
                                  'counts': {label: len(ids) for label, ids in encodings.items()},
                                  'consistent_counts': len({len(ids) for ids in encodings.values()}) == 1,
                                  'all_single_token': all(len(ids) == 1 for ids in encodings.values()),
                                  'distinct_sequences': len({tuple(ids) for ids in encodings.values()}) == 3}
            prompt = 'Compare responses A and B. Reply A, B, or C.\nVerdict:'
            try:
                rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                                          tokenize=False, add_generation_prompt=True, enable_thinking=False)
                prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
                full_ids = {label: tokenizer.encode(rendered + ' ' + label, add_special_tokens=False)
                            for label in encodings}
                row['chat_template'] = {'supported': True,
                                         'sha256': hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
                                         'rendered_suffix': rendered[-120:],
                                         'continuation_boundary_matches_standalone_label': {
                                             label: full_ids[label] == prompt_ids + ids
                                             for label, ids in encodings.items()},
                                         'contextual_label_ids_if_prefix_preserved': {
                                             label: ids[len(prompt_ids):] if ids[:len(prompt_ids)] == prompt_ids else None
                                             for label, ids in full_ids.items()}}
            except Exception as exc:
                row['chat_template'] = {'supported': False, **failure(exc)}
        else:
            encoded = tokenizer('A child plays outside.', 'Someone is outdoors.', add_special_tokens=True)
            row['tokenizer']['pair_encoding_succeeded'] = len(encoded['input_ids']) > 0
    except Exception as exc:
        row['tokenizer_error'] = failure(exc)
    return row


def main():
    jobs = []
    for name in ['core20.locked.json', 'independent_judges8.json', 'nli4.json']:
        manifest = json.loads((ROOT / 'supplementary' / 'exp00_response_cache' / name).read_text())
        jobs.extend((name, manifest['interface'], endpoint) for endpoint in manifest['endpoints'])
    output = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'versions': {'transformers': transformers.__version__, 'huggingface_hub': huggingface_hub.__version__},
              'credential_available': TOKEN is not None, 'weight_downloaded_bytes': 0,
              'scope': 'Pinned weight HEAD + config/tokenizer CPU audit. No weights loaded; GPU numerical compatibility untested.',
              'models': []}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(audit, item) for item in jobs]):
            row = future.result()
            output['models'].append(row)
            output['models'].sort(key=lambda r: (r['manifest'], r['judge_id']))
            OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
            print(row['judge_id'], row['model_id'],
                  'weights', row.get('weight_head', {}).get('accessible'),
                  'tokenizer', 'tokenizer' in row,
                  'loader', row.get('config', {}).get('auto_loader_class'), flush=True)
    output['complete'] = len(output['models']) == len(jobs)
    output['summary'] = {
        'weight_head_passed': sum(r.get('weight_head', {}).get('accessible', False) for r in output['models']),
        'blocked_model_ids': [r['model_id'] for r in output['models'] if not r.get('weight_head', {}).get('accessible')],
        'judge_multitoken_ids': [r['judge_id'] for r in output['models']
                                if r.get('label_audit') and not r['label_audit']['all_single_token']],
        'standalone_vs_contextual_boundary_mismatch_ids': [r['judge_id'] for r in output['models']
            if r.get('chat_template', {}).get('supported')
            and not all(r['chat_template']['continuation_boundary_matches_standalone_label'].values())],
    }
    output['runtime_notes'] = [
        'Transformers 4.57.6 maps gemma3 to Gemma3ForConditionalGeneration in both causal-LM and image-text AutoModel mappings.',
        'Transformers 4.57.6 from_pretrained accepts dtype; both OLMo2 pinned configs declare bfloat16. GPU smoke still required.',
        'Phi standalone label token sequences differ from tokenizer encoding after assistant template special tokens. Preserve original forced-token protocol or explicitly version a changed protocol.',
        'Mistral-Nemo pinned tokenizer triggers the 4.57.6 incorrect regex warning. fix_mistral_regex changes tokenization and must not be enabled silently in an original-protocol replication.',
        'Weight HEAD uses one checkpoint object per pinned repository. No checkpoint bodies are downloaded or GPU models instantiated.',
    ]
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print('Saved', str(OUT), flush=True)


if __name__ == '__main__':
    main()
