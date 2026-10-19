"""CLI smoke tests via subprocess (exercises the real `klt` entrypoint path).

Run with ``python -m kubeflow_llm_trainer.cli`` so the tests work even before
the package is `pip install`-ed as a console script.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "lora_small.yaml"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "kubeflow_llm_trainer.cli", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
    )


def test_list_backends():
    res = _run("list-backends")
    assert res.returncode == 0, res.stderr
    for name in ("trl", "unsloth", "llamafactory", "torchtune"):
        assert name in res.stdout


def test_list_backends_json():
    res = _run("list-backends", "--json")
    assert res.returncode == 0, res.stderr
    names = json.loads(res.stdout)
    assert set(["trl", "unsloth", "llamafactory", "torchtune"]).issubset(names)


def test_validate_example():
    res = _run("validate", str(EXAMPLE))
    assert res.returncode == 0, res.stderr
    assert "valid" in res.stdout.lower()


def test_validate_with_backend():
    res = _run("validate", str(EXAMPLE), "--backend", "trl")
    assert res.returncode == 0, res.stderr
    assert "trl" in res.stdout


def test_plan_trl():
    res = _run("plan", "--backend", "trl", str(EXAMPLE))
    assert res.returncode == 0, res.stderr
    assert "trl sft" in res.stdout
    assert "--use_peft" in res.stdout


def test_plan_torchtune():
    res = _run("plan", "--backend", "torchtune", str(EXAMPLE))
    assert res.returncode == 0, res.stderr
    assert "tune run" in res.stdout


def test_plan_json():
    res = _run("plan", "--backend", "trl", "--json", str(EXAMPLE))
    assert res.returncode == 0, res.stderr
    data = json.loads(res.stdout)
    assert data["backend"] == "trl"
    assert data["mode"] == "cli"


def test_train_dry_run():
    res = _run("train", "--backend", "trl", "--dry-run", str(EXAMPLE))
    assert res.returncode == 0, res.stderr
    assert "DRY RUN" in res.stdout


def test_unknown_backend_errors():
    res = _run("plan", "--backend", "nope", str(EXAMPLE))
    assert res.returncode != 0
    assert "Unknown backend" in res.stderr


def test_invalid_config_reports_errors(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("dataset:\n  name_or_path: imdb\n")  # missing model name
    res = _run("validate", str(bad))
    assert res.returncode != 0
    assert "Invalid config" in res.stderr


@pytest.mark.skipif(EXAMPLE.exists() is False, reason="example file missing")
def test_example_file_exists():
    assert EXAMPLE.exists()


# -- klt init ----------------------------------------------------------


def test_init_to_stdout():
    res = _run("init", "-")
    assert res.returncode == 0, res.stderr
    assert "model_name_or_path" in res.stdout


def test_init_writes_valid_config(tmp_path):
    out = tmp_path / "starter.yaml"
    res = _run("init", "--backend", "trl", str(out))
    assert res.returncode == 0, res.stderr
    assert out.exists()
    # Generated file validates.
    val = _run("validate", str(out))
    assert val.returncode == 0, val.stderr


def test_init_refuses_overwrite(tmp_path):
    out = tmp_path / "starter.yaml"
    assert _run("init", str(out)).returncode == 0
    res = _run("init", str(out))
    assert res.returncode != 0
    assert "Refusing to overwrite" in res.stderr


def test_init_force_overwrites(tmp_path):
    out = tmp_path / "starter.yaml"
    assert _run("init", str(out)).returncode == 0
    res = _run("init", "--force", str(out))
    assert res.returncode == 0, res.stderr


# -- klt manifest ------------------------------------------------------


def test_manifest_trainjob_is_valid_yaml():
    import yaml  # core dep

    res = _run("manifest", "--backend", "trl", "--kind", "trainjob", str(EXAMPLE))
    assert res.returncode == 0, res.stderr
    docs = list(yaml.safe_load_all(res.stdout))
    assert [d["kind"] for d in docs] == ["ConfigMap", "TrainingRuntime", "TrainJob"]


def test_manifest_pytorchjob_is_valid_yaml():
    import yaml

    res = _run("manifest", "--backend", "trl", "--kind", "pytorchjob", str(EXAMPLE))
    assert res.returncode == 0, res.stderr
    docs = list(yaml.safe_load_all(res.stdout))
    assert [d["kind"] for d in docs] == ["ConfigMap", "PyTorchJob"]


def test_manifest_gpu_flag():
    import yaml

    res = _run(
        "manifest", "--backend", "trl", "--kind", "trainjob", "--gpus", "4", str(EXAMPLE)
    )
    assert res.returncode == 0, res.stderr
    docs = list(yaml.safe_load_all(res.stdout))
    tj = docs[-1]
    req = tj["spec"]["trainer"]["resourcesPerNode"]["requests"]
    assert req["nvidia.com/gpu"] == "4"


def test_manifest_unknown_backend_errors():
    res = _run("manifest", "--backend", "nope", "--kind", "trainjob", str(EXAMPLE))
    assert res.returncode != 0
    assert "Unknown backend" in res.stderr


def test_manifest_invalid_kind_rejected_by_argparse():
    res = _run("manifest", "--backend", "trl", "--kind", "cronjob", str(EXAMPLE))
    assert res.returncode != 0  # argparse choices rejection
