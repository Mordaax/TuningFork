"""Kubernetes manifest generation for in-cluster training.

This module turns the unified :class:`~kubeflow_llm_trainer.config.TrainingConfig`
plus a chosen backend into ready-to-apply Kubernetes manifests that run the
backend's *exact* dry-run command inside a cluster. Two flavours are supported:

* ``trainjob`` — a **Kubeflow Trainer v2** ``TrainJob`` paired with a
  ``TrainingRuntime`` (the modern, runtime-templated model).
* ``pytorchjob`` — a classic **Kubeflow Training-Operator** ``PyTorchJob`` CRD
  (still widely deployed), useful as a portable fallback.

Design notes
------------
* The container ``command``/``args`` are produced by reusing each adapter's
  :meth:`~kubeflow_llm_trainer.base.TrainerBackend.build_command`, so the
  in-cluster invocation is byte-for-byte the dry-run plan. No drift.
* GPU requests are emitted as ``nvidia.com/gpu`` resource requests/limits.
* The output directory is backed by a ``Volume`` (``emptyDir`` by default; a
  PVC name can be supplied) so checkpoints survive the container's root fs.
* Only ``pyyaml`` is required — **no Kubernetes client and no ML libraries**,
  preserving the ML-free core invariant.

Everything is plain dicts rendered with :func:`yaml.safe_dump_all`, so the
result is valid, multi-document YAML you can pipe straight into ``kubectl``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .config import Precision, TrainingConfig
from .registry import get_backend

# Kubeflow Trainer v2 API group/version (TrainJob / TrainingRuntime).
TRAINER_API_VERSION = "trainer.kubeflow.org/v1alpha1"
# Classic Training-Operator API group/version (PyTorchJob).
PYTORCHJOB_API_VERSION = "kubeflow.org/v1"

#: Supported manifest kinds for the CLI ``--kind`` flag.
MANIFEST_KINDS = ("trainjob", "pytorchjob")


@dataclass
class ManifestOptions:
    """Knobs that describe *where/how* the job runs (cluster concerns).

    These are intentionally separate from :class:`TrainingConfig` (which is
    *what* to train). In Kubeflow terms, this is the runtime/topology side.
    """

    backend: str
    name: str = "llm-finetune"
    namespace: str = "default"
    image: Optional[str] = None  # defaults to a per-backend convention
    gpus: int = 1
    gpu_resource: str = "nvidia.com/gpu"
    num_nodes: int = 1
    cpu: str = "4"
    memory: str = "32Gi"
    config_path: str = "/etc/klt/config.yaml"
    output_volume_pvc: Optional[str] = None  # None => emptyDir
    env: dict[str, str] = field(default_factory=dict)
    image_pull_policy: str = "IfNotPresent"

    def resolved_image(self) -> str:
        """Container image, defaulting to a per-backend naming convention."""
        if self.image:
            return self.image
        return f"ghcr.io/kubeflow/llm-trainer-{self.backend}:latest"


def _container_command(cfg: TrainingConfig, opts: ManifestOptions) -> list[str]:
    """The container entrypoint: rerun the unified config in-cluster.

    We do NOT bake the backend's native argv directly; instead we invoke ``klt
    train`` against the mounted config so the in-cluster behaviour matches the
    CLI exactly (validation + plan + execute). The backend's resolved command
    is still surfaced via an annotation for transparency/debuggability.
    """
    return [
        "klt",
        "train",
        "--backend",
        opts.backend,
        opts.config_path,
    ]


def _resource_block(opts: ManifestOptions) -> dict[str, Any]:
    requests: dict[str, str] = {"cpu": opts.cpu, "memory": opts.memory}
    limits: dict[str, str] = {"cpu": opts.cpu, "memory": opts.memory}
    if opts.gpus > 0:
        requests[opts.gpu_resource] = str(opts.gpus)
        limits[opts.gpu_resource] = str(opts.gpus)
    return {"requests": requests, "limits": limits}


def _env_block(cfg: TrainingConfig, opts: ManifestOptions) -> list[dict[str, str]]:
    env = {
        "KLT_BACKEND": opts.backend,
        "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    }
    # Surface quantization intent for image entrypoints that branch on it.
    if cfg.precision in (Precision.INT4, Precision.INT8):
        env["KLT_QUANTIZED"] = cfg.precision.value
    env.update(opts.env)
    return [{"name": k, "value": str(v)} for k, v in env.items()]


def _output_volume(opts: ManifestOptions) -> tuple[dict[str, Any], dict[str, str]]:
    """Return (volume, volumeMount) for the training output_dir."""
    mount = {"name": "output", "mountPath": "/workspace/outputs"}
    if opts.output_volume_pvc:
        vol = {
            "name": "output",
            "persistentVolumeClaim": {"claimName": opts.output_volume_pvc},
        }
    else:
        vol = {"name": "output", "emptyDir": {}}
    return vol, mount


def _config_volume(opts: ManifestOptions) -> tuple[dict[str, Any], dict[str, str]]:
    """ConfigMap-backed volume that carries the unified TrainingConfig."""
    import os

    mount = {
        "name": "klt-config",
        "mountPath": os.path.dirname(opts.config_path),
        "readOnly": True,
    }
    vol = {
        "name": "klt-config",
        "configMap": {"name": f"{opts.name}-config"},
    }
    return vol, mount


def _container_spec(cfg: TrainingConfig, opts: ManifestOptions) -> dict[str, Any]:
    out_vol, out_mount = _output_volume(opts)
    cfg_vol, cfg_mount = _config_volume(opts)
    return {
        "name": "trainer",
        "image": opts.resolved_image(),
        "imagePullPolicy": opts.image_pull_policy,
        "command": _container_command(cfg, opts),
        "env": _env_block(cfg, opts),
        "resources": _resource_block(opts),
        "volumeMounts": [out_mount, cfg_mount],
    }, [out_vol, cfg_vol]


def _config_map(cfg: TrainingConfig, opts: ManifestOptions) -> dict[str, Any]:
    """A ConfigMap carrying the unified config the container reads."""
    import yaml

    config_yaml = yaml.safe_dump(cfg.to_dict(), sort_keys=False)
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": f"{opts.name}-config", "namespace": opts.namespace},
        "data": {"config.yaml": config_yaml},
    }


def _resolved_command_annotation(cfg: TrainingConfig, opts: ManifestOptions) -> str:
    """The backend's resolved native command, for transparency annotations."""
    plan = get_backend(opts.backend).build_plan(cfg)
    return plan.command_str or plan.api_call or ""


def _common_labels(opts: ManifestOptions) -> dict[str, str]:
    return {
        "app.kubernetes.io/name": "kubeflow-llm-trainer",
        "app.kubernetes.io/instance": opts.name,
        "klt.kubeflow.org/backend": opts.backend,
    }


def build_pytorchjob(cfg: TrainingConfig, opts: ManifestOptions) -> dict[str, Any]:
    """Build a classic ``PyTorchJob`` CRD dict."""
    container, volumes = _container_spec(cfg, opts)
    pod_template = {
        "metadata": {"labels": _common_labels(opts)},
        "spec": {
            "containers": [container],
            "volumes": volumes,
            "restartPolicy": "Never",
        },
    }
    replica_spec = {
        "replicas": opts.num_nodes,
        "restartPolicy": "OnFailure",
        "template": pod_template,
    }
    return {
        "apiVersion": PYTORCHJOB_API_VERSION,
        "kind": "PyTorchJob",
        "metadata": {
            "name": opts.name,
            "namespace": opts.namespace,
            "labels": _common_labels(opts),
            "annotations": {
                "klt.kubeflow.org/resolved-command": _resolved_command_annotation(
                    cfg, opts
                )
            },
        },
        "spec": {
            "runPolicy": {"cleanPodPolicy": "None"},
            "pytorchReplicaSpecs": {"Master": replica_spec},
        },
    }


def build_training_runtime(cfg: TrainingConfig, opts: ManifestOptions) -> dict[str, Any]:
    """Build a Kubeflow Trainer v2 ``TrainingRuntime`` (the reusable blueprint).

    The runtime owns cluster topology + the container image; the ``TrainJob``
    references it by name. We model the node set with a single ``node`` job.
    """
    container, volumes = _container_spec(cfg, opts)
    return {
        "apiVersion": TRAINER_API_VERSION,
        "kind": "TrainingRuntime",
        "metadata": {
            "name": f"{opts.backend}-runtime",
            "namespace": opts.namespace,
            "labels": _common_labels(opts),
        },
        "spec": {
            "mlPolicy": {"numNodes": opts.num_nodes, "torch": {"numProcPerNode": "auto"}},
            "template": {
                "spec": {
                    "replicatedJobs": [
                        {
                            "name": "node",
                            "template": {
                                "spec": {
                                    "template": {
                                        "metadata": {"labels": _common_labels(opts)},
                                        "spec": {
                                            "containers": [container],
                                            "volumes": volumes,
                                            "restartPolicy": "Never",
                                        },
                                    }
                                }
                            },
                        }
                    ]
                }
            },
        },
    }


def build_trainjob(cfg: TrainingConfig, opts: ManifestOptions) -> dict[str, Any]:
    """Build a Kubeflow Trainer v2 ``TrainJob`` referencing the runtime."""
    return {
        "apiVersion": TRAINER_API_VERSION,
        "kind": "TrainJob",
        "metadata": {
            "name": opts.name,
            "namespace": opts.namespace,
            "labels": _common_labels(opts),
            "annotations": {
                "klt.kubeflow.org/resolved-command": _resolved_command_annotation(
                    cfg, opts
                )
            },
        },
        "spec": {
            "runtimeRef": {
                "apiGroup": "trainer.kubeflow.org",
                "kind": "TrainingRuntime",
                "name": f"{opts.backend}-runtime",
            },
            "trainer": {
                "numNodes": opts.num_nodes,
                "command": _container_command(cfg, opts),
                "resourcesPerNode": _resource_block(opts),
                "env": _env_block(cfg, opts),
            },
        },
    }


def build_manifest_docs(
    cfg: TrainingConfig, opts: ManifestOptions, kind: str
) -> list[dict[str, Any]]:
    """Build the ordered list of manifest documents for ``kind``.

    * ``trainjob``  -> [ConfigMap, TrainingRuntime, TrainJob]
    * ``pytorchjob`` -> [ConfigMap, PyTorchJob]
    """
    kind = (kind or "").strip().lower()
    if kind not in MANIFEST_KINDS:
        raise ValueError(
            f"Unknown manifest kind '{kind}'. "
            f"Choose one of: {', '.join(MANIFEST_KINDS)}."
        )
    # Validate the backend exists early (raises KeyError with available list).
    get_backend(opts.backend)

    docs: list[dict[str, Any]] = [_config_map(cfg, opts)]
    if kind == "trainjob":
        docs.append(build_training_runtime(cfg, opts))
        docs.append(build_trainjob(cfg, opts))
    else:  # pytorchjob
        docs.append(build_pytorchjob(cfg, opts))
    return docs


def render_manifest_yaml(
    cfg: TrainingConfig, opts: ManifestOptions, kind: str
) -> str:
    """Render the manifests as a valid multi-document YAML string."""
    import yaml

    docs = build_manifest_docs(cfg, opts, kind)
    return yaml.safe_dump_all(docs, sort_keys=False, default_flow_style=False)
