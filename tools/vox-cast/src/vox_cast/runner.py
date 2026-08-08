"""Engine-side conversion runner. Executed BY the engine venv's interpreter —
never imported by the tool. Imports only the stdlib plus what the engine venv
provides (torch, rvc_python); it must not import vox_cast or smplstream.

Contract (kept in step with engine.run_convert and the test stubs):

    runner.py IN.wav OUT.wav --pth P [--index I] [--pitch 0]
        [--f0-method rmvpe] [--index-rate 0.5] [--protect 0.33]
        [--rms-mix 1.0] [--arch v2] [--device cpu:0]

On success prints one JSON line ``{"seconds": …}`` to stdout and exits 0.
"""

import argparse
import json
import os
import sys
import time


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="RVC conversion (engine side)")
    p.add_argument("in_wav")
    p.add_argument("out_wav")
    p.add_argument("--pth", required=True)
    p.add_argument("--index", default="")
    p.add_argument("--pitch", type=int, default=0)
    p.add_argument("--f0-method", default="rmvpe")
    p.add_argument("--index-rate", type=float, default=0.5)
    p.add_argument("--protect", type=float, default=0.33)
    p.add_argument("--rms-mix", type=float, default=1.0)
    p.add_argument("--arch", choices=("v1", "v2"), default="v2")
    p.add_argument("--device", default="cpu:0")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import torch

    if args.device.startswith("cpu"):
        # rvc-python's Config ignores the requested device and force-selects
        # MPS whenever torch reports it available (and its singleton wrapper
        # defeats patching the class) — so close the door at the torch level.
        torch.backends.mps.is_available = lambda: False

    from rvc_python.infer import RVCInference

    t0 = time.time()
    rvc = RVCInference(device=args.device, model_path=args.pth,
                       index_path=args.index, version=args.arch)
    rvc.set_params(
        f0method=args.f0_method,
        f0up_key=args.pitch,
        index_rate=args.index_rate,
        protect=args.protect,
        rms_mix_rate=args.rms_mix,
    )
    rvc.infer_file(args.in_wav, args.out_wav)
    if not os.path.isfile(args.out_wav) or os.path.getsize(args.out_wav) == 0:
        sys.stderr.write("runner: engine produced no output audio\n")
        return 1
    print(json.dumps({"seconds": round(time.time() - t0, 2)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
