from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
from time import perf_counter
from typing import Any
from uuid import uuid4


TRAINING_RECORD_SCHEMA_VERSION = "1.0"
_SECRET_FIELD_PATTERN = re.compile(r"api[_-]?key|authorization|password|secret|token", re.IGNORECASE)
_SECRET_VALUE_PATTERN = re.compile(r"\b(?:sk|key)-[A-Za-z0-9_-]{12,}\b")


@dataclass(frozen=True)
class TrainingRunPaths:
    """一次训练运行的独立产物路径。"""

    run_id: str
    run_directory: Path
    checkpoint: Path
    report: Path
    manifest: Path
    index: Path
    latest: Path


class TrainingRunRecorder:
    """以不可覆盖的目录和追加式索引记录每一次训练。"""

    def __init__(
        self,
        output_root: str | Path,
        *,
        seed: int,
        algorithm: str,
        parameters: dict[str, Any],
        run_id: str | None = None,
    ) -> None:
        root = Path(output_root)
        selected_run_id = run_id or _build_run_id(seed)
        run_directory = root / selected_run_id
        if run_directory.exists():
            raise FileExistsError(f"训练运行目录已存在：{run_directory}")
        run_directory.mkdir(parents=True, exist_ok=False)
        self.paths = TrainingRunPaths(
            run_id=selected_run_id,
            run_directory=run_directory,
            checkpoint=run_directory / "recovery_policy.pt",
            report=run_directory / "training_report.json",
            manifest=run_directory / "manifest.json",
            index=root / "training_runs.jsonl",
            latest=root / "latest.json",
        )
        self._started_clock = perf_counter()
        self._record = {
            "schema_version": TRAINING_RECORD_SCHEMA_VERSION,
            "run_id": selected_run_id,
            "status": "running",
            "started_at": _now_iso(),
            "completed_at": None,
            "duration_seconds": None,
            "algorithm": algorithm,
            "parameters": _remove_secrets(parameters),
            "source": _source_metadata(),
            "runtime": _runtime_metadata(),
            "artifacts": {},
            "metrics": {},
        }
        self._write_manifest()

    def complete(
        self,
        report: dict[str, Any],
        *,
        extra_artifacts: dict[str, str | Path] | None = None,
    ) -> dict[str, Any]:
        """保存成功状态、关键指标、文件摘要和历史索引。"""

        artifacts = {
            "checkpoint": _artifact_metadata(self.paths.checkpoint, self.paths.run_directory),
            "report": _artifact_metadata(self.paths.report, self.paths.run_directory),
        }
        for name, path in (extra_artifacts or {}).items():
            artifacts[name] = _artifact_metadata(Path(path), self.paths.run_directory)
        self._record.update(
            {
                "status": "completed",
                "completed_at": _now_iso(),
                "duration_seconds": round(perf_counter() - self._started_clock, 6),
                "metrics": _metric_summary(report),
                "artifacts": artifacts,
            }
        )
        self._write_manifest()
        self._append_index()
        self._write_latest()
        return dict(self._record)

    def fail(self, error: BaseException) -> dict[str, Any]:
        """即使训练失败也留下可审计记录，便于统计失败原因。"""

        self._record.update(
            {
                "status": "failed",
                "completed_at": _now_iso(),
                "duration_seconds": round(perf_counter() - self._started_clock, 6),
                "error": {
                    "type": type(error).__name__,
                    "message": _redact_text(str(error))[:1000],
                },
            }
        )
        self._write_manifest()
        self._append_index()
        self._write_latest()
        return dict(self._record)

    def _write_manifest(self) -> None:
        self.paths.manifest.write_text(
            json.dumps(self._record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _append_index(self) -> None:
        self.paths.index.parent.mkdir(parents=True, exist_ok=True)
        index_record = {
            "schema_version": self._record["schema_version"],
            "run_id": self._record["run_id"],
            "status": self._record["status"],
            "started_at": self._record["started_at"],
            "completed_at": self._record["completed_at"],
            "duration_seconds": self._record["duration_seconds"],
            "algorithm": self._record["algorithm"],
            "seed": self._record["parameters"].get("seed"),
            "metrics": self._record.get("metrics", {}),
            "manifest": str(self.paths.manifest.relative_to(self.paths.index.parent)),
        }
        with self.paths.index.open("a", encoding="utf-8") as file:
            file.write(json.dumps(index_record, ensure_ascii=False) + "\n")

    def _write_latest(self) -> None:
        latest_record = {
            "run_id": self._record["run_id"],
            "status": self._record["status"],
            "manifest": str(self.paths.manifest.relative_to(self.paths.latest.parent)),
        }
        self.paths.latest.write_text(
            json.dumps(latest_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _build_run_id(seed: int) -> str:
    """使用本地时间、随机种子和短 UUID 生成可读且唯一的运行编号。"""

    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    return f"{timestamp}_seed{seed}_{uuid4().hex[:8]}"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _source_metadata() -> dict[str, Any]:
    """记录代码版本，不保存 diff 内容，避免把本地数据写入训练记录。"""

    return {
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
        "git_dirty": bool(_git_output("status", "--porcelain")),
    }


def _git_output(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _runtime_metadata() -> dict[str, Any]:
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        metadata.update({"torch": torch.__version__, "device": "cpu"})
    except ModuleNotFoundError:
        metadata.update({"torch": None, "device": None})
    return metadata


def _metric_summary(report: dict[str, Any]) -> dict[str, Any]:
    training = report.get("training", {})
    evaluation = report.get("evaluation", {})
    return {
        "training": {
            "demonstration_count": training.get("demonstration_count"),
            "environment_steps": training.get("environment_steps"),
            "optimization_steps": training.get("optimization_steps"),
            "bc_final_loss": training.get("bc_final_loss"),
            "dqn_final_loss": training.get("dqn_final_loss"),
        },
        "test": {
            policy_name: policy_metrics.get("test", {})
            for policy_name, policy_metrics in evaluation.items()
        },
    }


def _artifact_metadata(path: Path, run_directory: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path.relative_to(run_directory)), "exists": False}
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.relative_to(run_directory)),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _remove_secrets(value: Any) -> Any:
    """递归移除可能包含凭据的字段，并遮盖误传入的密钥样式字符串。"""

    if isinstance(value, dict):
        return {
            str(key): _remove_secrets(item)
            for key, item in value.items()
            if not _SECRET_FIELD_PATTERN.search(str(key))
        }
    if isinstance(value, list):
        return [_remove_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [_remove_secrets(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _redact_text(value: str) -> str:
    return _SECRET_VALUE_PATTERN.sub("[REDACTED]", value)
