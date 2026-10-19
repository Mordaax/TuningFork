"""``klt`` command-line interface.

Subcommands:
  klt list-backends                          List registered backends.
  klt init [--backend trl] <out.yaml>        Write a starter config.
  klt validate <config.yaml>                 Validate a config file.
  klt plan --backend trl <config.yaml>       Print the dry-run plan.
  klt train --backend trl --dry-run <cfg>    Build the plan (and run if not dry).
  klt manifest --backend trl --kind trainjob <cfg>
                                             Emit Kubernetes manifests.

Uses only the stdlib ``argparse`` plus ``pyyaml`` so the CLI works with no
ML libraries installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional, Sequence

from . import __version__
from .base import BackendDependencyError, TrainingPlan
from .config import TrainingConfig
from .manifests import MANIFEST_KINDS, ManifestOptions, render_manifest_yaml
from .registry import get_backend, list_backends, load_entry_point_backends
from .scaffold import starter_config_yaml


def _load_yaml(path: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:  # pragma: no cover - pyyaml is a core dep
        raise SystemExit(
            "PyYAML is required to read config files. Install with: pip install pyyaml"
        )
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise SystemExit(f"Config file '{path}' must contain a YAML mapping.")
    return data


def _load_config(path: str) -> TrainingConfig:
    from pydantic import ValidationError

    raw = _load_yaml(path)
    try:
        return TrainingConfig.from_dict(raw)
    except ValidationError as exc:
        # Pretty, actionable validation errors.
        msgs = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "<root>"
            msgs.append(f"  - {loc}: {err['msg']}")
        raise SystemExit(
            f"Invalid config '{path}':\n" + "\n".join(msgs)
        )


def _print_plan(plan: TrainingPlan, as_json: bool) -> None:
    if as_json:
        print(json.dumps(plan.to_dict(), indent=2))
        return

    print(f"Backend : {plan.backend}")
    print(f"Mode    : {plan.mode}")
    if plan.command:
        print("Command :")
        print(f"  {plan.command_str}")
    if plan.api_call:
        print("API call:")
        print(f"  {plan.api_call}")
    if plan.resolved:
        print("Resolved:")
        for k, v in plan.resolved.items():
            print(f"  {k}: {v}")
    if plan.artifacts:
        print("Artifacts (would be written):")
        for name in plan.artifacts:
            print(f"  - {name}")
    if plan.notes:
        print("Notes:")
        for note in plan.notes:
            print(f"  * {note}")


# -- subcommand handlers -----------------------------------------------


def _cmd_list_backends(args: argparse.Namespace) -> int:
    load_entry_point_backends()  # pick up any plugin packages too
    names = list_backends()
    if args.json:
        print(json.dumps(names))
        return 0
    print("Available backends:")
    for name in names:
        backend = get_backend(name)
        dep = backend.required_dependency or "(none)"
        have = "installed" if backend.dependency_available() else "not installed"
        print(f"  - {name:<13} (dependency: {dep}, {have})")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    if args.backend:
        backend = get_backend(args.backend)
        backend.validate(cfg)
        print(f"OK: '{args.config}' is valid for backend '{backend.name}'.")
    else:
        print(f"OK: '{args.config}' is a valid TrainingConfig.")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    backend = get_backend(args.backend)
    backend.validate(cfg)
    plan = backend.build_plan(cfg)
    _print_plan(plan, args.json)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    backend = get_backend(args.backend)
    try:
        plan = backend.train(cfg, dry_run=args.dry_run)
    except BackendDependencyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.dry_run:
        print("[DRY RUN] No training executed. Plan:")
    _print_plan(plan, args.json)
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    import os

    text = starter_config_yaml(args.backend)
    if args.output in ("-", "/dev/stdout"):
        print(text, end="")
        return 0
    if os.path.exists(args.output) and not args.force:
        print(
            f"Refusing to overwrite existing file '{args.output}' "
            "(pass --force to overwrite).",
            file=sys.stderr,
        )
        return 2
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"Wrote starter config to '{args.output}'.")
    if args.backend:
        print(f"  tuned for backend: {args.backend}")
    print(f"Next: klt validate {args.output}")
    return 0


def _cmd_manifest(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    # Surface backend/kind problems with clear messages (caught in main()).
    backend = get_backend(args.backend)
    backend.validate(cfg)
    opts = ManifestOptions(
        backend=backend.name,
        name=args.name,
        namespace=args.namespace,
        image=args.image,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        cpu=args.cpu,
        memory=args.memory,
        output_volume_pvc=args.pvc,
    )
    yaml_text = render_manifest_yaml(cfg, opts, args.kind)
    if args.output and args.output not in ("-", "/dev/stdout"):
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(yaml_text)
        print(f"Wrote {args.kind} manifest(s) to '{args.output}'.", file=sys.stderr)
    else:
        print(yaml_text, end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="klt",
        description="Kubeflow LLM Trainer: pluggable multi-framework fine-tuning.",
    )
    parser.add_argument("--version", action="version", version=f"klt {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-backends", help="List registered backends.")
    p_list.add_argument("--json", action="store_true", help="Emit JSON.")
    p_list.set_defaults(func=_cmd_list_backends)

    p_val = sub.add_parser("validate", help="Validate a config file.")
    p_val.add_argument("config", help="Path to a YAML config.")
    p_val.add_argument(
        "--backend", help="Also run backend-specific validation.", default=None
    )
    p_val.set_defaults(func=_cmd_validate)

    p_plan = sub.add_parser("plan", help="Print the dry-run plan for a backend.")
    p_plan.add_argument("config", help="Path to a YAML config.")
    p_plan.add_argument("--backend", required=True, help="Backend name.")
    p_plan.add_argument("--json", action="store_true", help="Emit JSON.")
    p_plan.set_defaults(func=_cmd_plan)

    p_train = sub.add_parser("train", help="Train (or dry-run) with a backend.")
    p_train.add_argument("config", help="Path to a YAML config.")
    p_train.add_argument("--backend", required=True, help="Backend name.")
    p_train.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the plan without executing real training.",
    )
    p_train.add_argument("--json", action="store_true", help="Emit JSON.")
    p_train.set_defaults(func=_cmd_train)

    p_init = sub.add_parser("init", help="Write a commented starter config.")
    p_init.add_argument(
        "output", help="Path to write the starter config (or '-' for stdout)."
    )
    p_init.add_argument(
        "--backend",
        default=None,
        help="Tune the starter config for a specific backend.",
    )
    p_init.add_argument(
        "--force", action="store_true", help="Overwrite an existing file."
    )
    p_init.set_defaults(func=_cmd_init)

    p_man = sub.add_parser(
        "manifest", help="Emit Kubernetes manifests for in-cluster training."
    )
    p_man.add_argument("config", help="Path to a YAML config.")
    p_man.add_argument("--backend", required=True, help="Backend name.")
    p_man.add_argument(
        "--kind",
        default="trainjob",
        choices=MANIFEST_KINDS,
        help="Manifest flavour (default: trainjob).",
    )
    p_man.add_argument("--name", default="llm-finetune", help="Job/resource name.")
    p_man.add_argument("--namespace", default="default", help="Kubernetes namespace.")
    p_man.add_argument(
        "--image", default=None, help="Container image (defaults per-backend)."
    )
    p_man.add_argument("--gpus", type=int, default=1, help="GPUs per node.")
    p_man.add_argument("--num-nodes", type=int, default=1, help="Number of nodes.")
    p_man.add_argument("--cpu", default="4", help="CPU request/limit per node.")
    p_man.add_argument("--memory", default="32Gi", help="Memory request/limit.")
    p_man.add_argument(
        "--pvc",
        default=None,
        help="PVC name to back the output dir (default: emptyDir).",
    )
    p_man.add_argument(
        "-o", "--output", default=None, help="Write to file instead of stdout."
    )
    p_man.set_defaults(func=_cmd_manifest)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (KeyError, ValueError) as exc:
        # Unknown backend / backend-validation failures.
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
