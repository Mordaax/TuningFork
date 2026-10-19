"""Kubernetes manifest generation.

These tests MUST pass with NO ML libraries installed: manifest generation only
needs the unified config + the adapter's (ML-free) build_command.
"""

from __future__ import annotations

import pytest
import yaml

from kubeflow_llm_trainer.config import Precision
from kubeflow_llm_trainer.manifests import (
    MANIFEST_KINDS,
    ManifestOptions,
    build_manifest_docs,
    render_manifest_yaml,
)


# -- valid YAML / document structure -----------------------------------


def test_trainjob_emits_valid_multidoc_yaml(sample_config):
    opts = ManifestOptions(backend="trl")
    text = render_manifest_yaml(sample_config, opts, "trainjob")
    docs = list(yaml.safe_load_all(text))
    kinds = [d["kind"] for d in docs]
    assert kinds == ["ConfigMap", "TrainingRuntime", "TrainJob"]


def test_pytorchjob_emits_valid_multidoc_yaml(sample_config):
    opts = ManifestOptions(backend="trl")
    text = render_manifest_yaml(sample_config, opts, "pytorchjob")
    docs = list(yaml.safe_load_all(text))
    assert [d["kind"] for d in docs] == ["ConfigMap", "PyTorchJob"]


@pytest.mark.parametrize("kind", MANIFEST_KINDS)
def test_every_kind_parses(sample_config, kind):
    opts = ManifestOptions(backend="torchtune")
    docs = list(yaml.safe_load_all(render_manifest_yaml(sample_config, opts, kind)))
    # All docs are mappings with apiVersion + kind + metadata.
    for d in docs:
        assert {"apiVersion", "kind", "metadata"} <= set(d)
        assert d["metadata"]["name"]


# -- GPU / resources ---------------------------------------------------


def test_gpu_request_present(sample_config):
    opts = ManifestOptions(backend="trl", gpus=2)
    tj = build_manifest_docs(sample_config, opts, "trainjob")[-1]
    res = tj["spec"]["trainer"]["resourcesPerNode"]
    assert res["requests"]["nvidia.com/gpu"] == "2"
    assert res["limits"]["nvidia.com/gpu"] == "2"


def test_gpu_zero_omits_gpu_resource(sample_config):
    opts = ManifestOptions(backend="trl", gpus=0)
    pj = build_manifest_docs(sample_config, opts, "pytorchjob")[-1]
    c = pj["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"][
        "containers"
    ][0]
    assert "nvidia.com/gpu" not in c["resources"]["requests"]


def test_custom_resources(sample_config):
    opts = ManifestOptions(backend="trl", cpu="8", memory="64Gi")
    pj = build_manifest_docs(sample_config, opts, "pytorchjob")[-1]
    c = pj["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"][
        "containers"
    ][0]
    assert c["resources"]["requests"]["cpu"] == "8"
    assert c["resources"]["requests"]["memory"] == "64Gi"


# -- image / command / env / volumes -----------------------------------


def test_default_image_is_per_backend(sample_config):
    opts = ManifestOptions(backend="unsloth")
    pj = build_manifest_docs(sample_config, opts, "pytorchjob")[-1]
    c = pj["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"][
        "containers"
    ][0]
    assert "unsloth" in c["image"]


def test_custom_image_used(sample_config):
    opts = ManifestOptions(backend="trl", image="myreg/img:v9")
    pj = build_manifest_docs(sample_config, opts, "pytorchjob")[-1]
    c = pj["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"][
        "containers"
    ][0]
    assert c["image"] == "myreg/img:v9"


def test_container_command_invokes_klt_train(sample_config):
    opts = ManifestOptions(backend="trl", config_path="/etc/klt/config.yaml")
    pj = build_manifest_docs(sample_config, opts, "pytorchjob")[-1]
    c = pj["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"][
        "containers"
    ][0]
    assert c["command"] == ["klt", "train", "--backend", "trl", "/etc/klt/config.yaml"]


def test_resolved_native_command_in_annotation(sample_config):
    """The TrainJob annotation carries the adapter's *native* resolved command."""
    opts = ManifestOptions(backend="trl")
    tj = build_manifest_docs(sample_config, opts, "trainjob")[-1]
    ann = tj["metadata"]["annotations"]["klt.kubeflow.org/resolved-command"]
    assert "trl sft" in ann
    assert "--use_peft" in ann


def test_quantized_precision_sets_env(sample_config):
    cfg = sample_config.model_copy(update={"precision": Precision.INT4})
    opts = ManifestOptions(backend="trl")
    pj = build_manifest_docs(cfg, opts, "pytorchjob")[-1]
    c = pj["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"][
        "containers"
    ][0]
    env = {e["name"]: e["value"] for e in c["env"]}
    assert env.get("KLT_QUANTIZED") == "int4"


def test_custom_env_merged(sample_config):
    opts = ManifestOptions(backend="trl", env={"HF_TOKEN": "xyz"})
    pj = build_manifest_docs(sample_config, opts, "pytorchjob")[-1]
    c = pj["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"][
        "containers"
    ][0]
    env = {e["name"]: e["value"] for e in c["env"]}
    assert env["HF_TOKEN"] == "xyz"
    assert env["KLT_BACKEND"] == "trl"


def test_output_volume_emptydir_by_default(sample_config):
    opts = ManifestOptions(backend="trl")
    pj = build_manifest_docs(sample_config, opts, "pytorchjob")[-1]
    vols = pj["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"][
        "volumes"
    ]
    out = next(v for v in vols if v["name"] == "output")
    assert "emptyDir" in out


def test_output_volume_pvc_when_requested(sample_config):
    opts = ManifestOptions(backend="trl", output_volume_pvc="ckpt-pvc")
    pj = build_manifest_docs(sample_config, opts, "pytorchjob")[-1]
    vols = pj["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"][
        "volumes"
    ]
    out = next(v for v in vols if v["name"] == "output")
    assert out["persistentVolumeClaim"]["claimName"] == "ckpt-pvc"


def test_configmap_carries_unified_config(sample_config):
    opts = ManifestOptions(backend="trl")
    cm = build_manifest_docs(sample_config, opts, "trainjob")[0]
    assert cm["kind"] == "ConfigMap"
    embedded = yaml.safe_load(cm["data"]["config.yaml"])
    assert embedded["model_name_or_path"] == sample_config.model_name_or_path
    assert embedded["lora"]["r"] == sample_config.lora.r


def test_namespace_and_name_propagate(sample_config):
    opts = ManifestOptions(backend="trl", name="exp-7", namespace="research")
    docs = build_manifest_docs(sample_config, opts, "trainjob")
    for d in docs:
        assert d["metadata"]["namespace"] == "research"
    tj = docs[-1]
    assert tj["metadata"]["name"] == "exp-7"


def test_trainjob_runtime_ref_matches_runtime(sample_config):
    opts = ManifestOptions(backend="torchtune")
    docs = build_manifest_docs(sample_config, opts, "trainjob")
    runtime = next(d for d in docs if d["kind"] == "TrainingRuntime")
    trainjob = next(d for d in docs if d["kind"] == "TrainJob")
    assert trainjob["spec"]["runtimeRef"]["name"] == runtime["metadata"]["name"]


# -- error cases -------------------------------------------------------


def test_unknown_kind_raises(sample_config):
    opts = ManifestOptions(backend="trl")
    with pytest.raises(ValueError):
        build_manifest_docs(sample_config, opts, "cronjob")


def test_unknown_backend_raises(sample_config):
    opts = ManifestOptions(backend="does-not-exist")
    with pytest.raises(KeyError):
        build_manifest_docs(sample_config, opts, "trainjob")
