"""Run the measured Flash-Next coding presets on the Arrow Lake NPU host."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


LLAMA = Path(__file__).resolve().parents[3]
WORKSPACE = LLAMA.parent
HUB = Path.home() / '.cache/huggingface/hub'
UNSLOTH = HUB / 'models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/38bb39ee97821de2c9009abb7e93950eec396e66'
ATOMIC = HUB / 'models--AtomicChat--Qwen3.8-Flash-Next-GGUF/snapshots/142262902a46f7daed19c79d0771534c8106ad59'
PRESETS = {
    'unsloth': (UNSLOTH / 'UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf', 'npu', 5, 'blobcache_unsloth_v3'),
    'atomic': (ATOMIC / 'Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf', 'cpu', 3, 'blobcache_atomic_v3s'),
}


def command(args):
    model, preferred, depth, cache = PRESETS[args.model]
    backend = args.backend or preferred
    draft = UNSLOTH / 'MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf'
    server = args.server or LLAMA / 'build-npu-defaults/bin/llama-server.exe'
    env = os.environ.copy()
    # Backend tuning is implemented in C/C++; only external cache identity is needed here.
    env.setdefault('NPU_BLOB_CACHE', str(WORKSPACE / 're/wm' / cache))
    env['GGML_NPU_DISABLE'] = '1' if backend == 'cpu' else '0'
    env['PATH'] = 'C:/msys64/ucrt64/bin;' + env.get('PATH', '')
    argv = [str(server), '-m', str(model), '--host', '127.0.0.1', '--port', str(args.port),
            '--device', 'hpi-3720' if backend == 'npu' else 'none',
            '-ngl', '999' if backend == 'npu' else '0', '-ncmoe', '48',
            '-c', '2048', '-b', '128', '-ub', '128', '-t', '4', '-tb', '4',
            '--cache-ram', '0', '--ctx-checkpoints', '0', '--parallel', '1',
            '--reasoning', 'off', '--verbosity', '1', '--fit', 'off', '--no-warmup', '-fa', 'on',
            '--spec-type', 'draft-mtp', '--spec-draft-n-max', str(depth), '-md', str(draft),
            '--device-draft', 'none', '-ngld', '0', '-td', '4', '-tbd', '4']
    return argv, env, model, draft, backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', nargs='?', choices=PRESETS, default='unsloth')
    parser.add_argument('--backend', choices=('npu', 'cpu'), help='override the measured model default')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--server', type=Path, help='use another hardware-enabled llama-server build')
    parser.add_argument('--print-only', action='store_true', help='inspect the command without loading a model')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    argv, env, model, draft, backend = command(args)
    if args.print_only:
        print(json.dumps({'argv': argv, 'backend': backend,
                          'cache': env['NPU_BLOB_CACHE'],
                          'scope': 'coding preset; CPU/NPU numerical equivalence remains unproven'}, indent=2))
        return 0
    # Require all shards before allocating model memory.
    match = re.fullmatch(r'(.*)-00001-of-(\d{5})\.gguf', model.name)
    shards = [model.with_name(f'{match[1]}-{i:05d}-of-{match[2]}.gguf')
              for i in range(1, int(match[2]) + 1)] if match else [model]
    required = [Path(argv[0]), draft, *shards]
    if backend == 'npu':
        required.append(Path(env['NPU_BLOB_CACHE']) / 'cache-v3.ready')
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error('missing required files:\n' + '\n'.join(missing))
    print(f'{args.model}: {backend}, MTP {PRESETS[args.model][2]}, 4 threads, batch/microbatch 128', flush=True)
    return subprocess.call(argv, env=env, cwd=LLAMA)


if __name__ == '__main__':
    raise SystemExit(main())
