from __future__ import annotations

import json
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.services.s3_service import S3Service


logger = logging.getLogger(__name__)

class OutputService:
    def __init__(
        self,
        base_dir: str,
        *,
        s3_service: S3Service | None = None,
        s3_service: S3Service | None = None,
        local_output_enabled: bool = False,
    ) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.s3_service = s3_service
        self.s3_service = s3_service
        self.local_output_enabled = local_output_enabled
        if self.local_output_enabled:
            self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_run_dir(self) -> Path:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_name = f"{timestamp}_{uuid4().hex[:8]}"
        if self.local_output_enabled:
            run_dir = self.base_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir.resolve()
        return Path(tempfile.mkdtemp(prefix=f"{run_name}_")).resolve()

    def save_json(self, run_dir: Path, name: str, payload: Any) -> None:
        if not self.local_output_enabled:
            return
        path = run_dir.resolve() / f"{name}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)

    def save_binary(
        self,
        run_dir: Path,
        relative_path: str,
        payload: bytes,
        *,
        mime_type: str = "application/octet-stream",
    ) -> str:
        if self.s3_service is not None and self.s3_service.enabled:
            object_key = f"{run_dir.name}/{Path(relative_path).as_posix()}"
            try:
                asset = self.s3_service.upload_bytes(
                    payload,
                    key=object_key,
                    mime_type=mime_type,
                )
                return asset.url
            except Exception as exc:
                logger.warning("S3 upload failed, falling back to local output: %s", exc)

        path = (run_dir.resolve() / relative_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return str(path.resolve())
