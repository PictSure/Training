from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import duckdb
import torch
from torch.utils.data import IterableDataset, get_worker_info
import numpy as np
from pathlib import Path
import os


def _quote_identifier(name: str) -> str:
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _safe_name(value: str) -> str:
    sanitized = [c if c.isalnum() or c in ("-", "_") else "_" for c in value]
    joined = "".join(sanitized).strip("_")
    return joined or "value"


@dataclass
class _DatasetMeta:
    info: Dict[str, object]
    labels: Dict[str, Dict[str, object]] = field(default_factory=dict)


class _BaseDuckDBEpisodicDataset(IterableDataset):
    """Generate few-shot episodes from DuckDB-backed embedding stores."""

    def __init__(
        self,
        *,
        db_path: str,
        num_classes: int = 5,
        samples_per_class: int = 5,
        dataset: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
        episodes: Optional[int] = None,
        seed: Optional[int] = None,
        return_dataset_name: bool = False,
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be greater than zero")
        if samples_per_class <= 0:
            raise ValueError("samples_per_class must be greater than zero")
        self.db_path = db_path
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.dataset_filter = dataset
        self.dtype = dtype
        self.device = torch.device(device)
        self.episodes = episodes
        self.seed = seed
        self.return_dataset_name = return_dataset_name

        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._gen: Optional[torch.Generator] = None
        self._datasets: Dict[str, _DatasetMeta] = {}
        self._dataset_names: List[str] = []
        self._label_cache: Dict[str, Sequence[str]] = {}

    # ------------------------------------------------------------------
    # Hooks to be implemented by subclasses
    # ------------------------------------------------------------------
    def _load_datasets(self, conn: duckdb.DuckDBPyConnection) -> Dict[str, _DatasetMeta]:
        raise NotImplementedError

    def _labels_for_dataset(
        self,
        conn: duckdb.DuckDBPyConnection,
        dataset_name: str,
        meta: _DatasetMeta,
        min_samples: int,
    ) -> Sequence[str]:
        raise NotImplementedError

    def _fetch_embeddings(
        self,
        conn: duckdb.DuckDBPyConnection,
        dataset_name: str,
        meta: _DatasetMeta,
        label_name: str,
        limit: int,
    ) -> Sequence[Sequence[float]]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # IterableDataset interface
    # ------------------------------------------------------------------
    def __iter__(self):
        self._init_worker_state()
        worker = get_worker_info()
        if self.episodes is None:
            episodes_remaining: Optional[int] = None
        elif worker is None:
            episodes_remaining = self.episodes
        else:
            # Evenly split episodes across workers so total episodes == requested
            base = self.episodes // worker.num_workers
            remainder = self.episodes % worker.num_workers
            episodes_remaining = base + (1 if worker.id < remainder else 0)
        try:
            while episodes_remaining is None or episodes_remaining > 0:
                yield self._sample_episode()
                if episodes_remaining is not None:
                    episodes_remaining -= 1
        finally:
            self._teardown_worker_state()

    def __len__(self) -> int:
        if self.episodes is None:
            raise TypeError("Length is undefined when episodes is not set")
        return self.episodes

    # ------------------------------------------------------------------
    # Worker lifecycle helpers
    # ------------------------------------------------------------------
    def _init_worker_state(self) -> None:
        if self._conn is not None:
            return
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0

        self._conn = duckdb.connect(database=self.db_path, read_only=True, config={"memory_limit": "3GB"})
        gen = torch.Generator()
        if self.seed is not None:
            gen.manual_seed(self.seed + worker_id)
        else:
            gen.seed()
        self._gen = gen

        datasets = self._load_datasets(self._conn)
        if not datasets:
            raise RuntimeError("No datasets found in DuckDB database")

        if self.dataset_filter is not None:
            if self.dataset_filter not in datasets:
                raise ValueError(f"Dataset '{self.dataset_filter}' not present in DuckDB store")
            self._datasets = {self.dataset_filter: datasets[self.dataset_filter]}
        else:
            self._datasets = datasets

        self._dataset_names = list(self._datasets.keys())
        if not self._dataset_names:
            raise RuntimeError("No datasets available after applying filter")
        self._label_cache.clear()

    def _teardown_worker_state(self) -> None:
        if self._conn is not None:
            self._conn.close()
        self._conn = None
        self._gen = None
        self._dataset_names = []
        self._label_cache.clear()

    # ------------------------------------------------------------------
    # Episode sampling
    # ------------------------------------------------------------------
    def _choose_dataset(self) -> Tuple[str, _DatasetMeta]:
        assert self._gen is not None
        if len(self._dataset_names) == 1:
            name = self._dataset_names[0]
        else:
            index = int(torch.randint(len(self._dataset_names), (1,), generator=self._gen).item())
            name = self._dataset_names[index]
        return name, self._datasets[name]

    def _labels_for_episode(self, dataset_name: str, meta: _DatasetMeta) -> Sequence[str]:
        cache_hit = self._label_cache.get(dataset_name)
        min_samples = self.samples_per_class + 1
        if cache_hit is None or len(cache_hit) < self.num_classes:
            assert self._conn is not None
            labels = self._labels_for_dataset(self._conn, dataset_name, meta, min_samples)
            self._label_cache[dataset_name] = labels
            cache_hit = labels
        if len(cache_hit) < self.num_classes:
            raise RuntimeError(
                f"Dataset '{dataset_name}' does not have enough labels with at least {min_samples} samples"
            )
        return cache_hit

    def _sample_episode(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        assert self._conn is not None and self._gen is not None
        dataset_name, meta = self._choose_dataset()
        labels = self._labels_for_episode(dataset_name, meta)
        label_indices = torch.randperm(len(labels), generator=self._gen)[: self.num_classes]

        context_embeddings: List[Sequence[float]] = []
        context_targets: List[int] = []
        prediction_candidates: List[Tuple[Sequence[float], int]] = []

        min_samples = self.samples_per_class + 1
        for local_index, label_position in enumerate(label_indices.tolist()):
            label_name = labels[label_position]
            embeddings = self._fetch_embeddings(self._conn, dataset_name, meta, label_name, min_samples)
            if len(embeddings) < min_samples:
                raise RuntimeError(
                    f"Label '{label_name}' in dataset '{dataset_name}' does not provide enough embeddings"
                )
            context_embeddings.extend(embeddings[: self.samples_per_class])
            context_targets.extend([local_index] * self.samples_per_class)
            prediction_candidates.append((embeddings[self.samples_per_class], local_index))

        candidate_index = int(torch.randint(len(prediction_candidates), (1,), generator=self._gen).item())
        pred_embedding, pred_label = prediction_candidates[candidate_index]

        target_device = self.device
        context_images = torch.tensor(context_embeddings, dtype=self.dtype, device=target_device)
        context_labels = torch.tensor(context_targets, dtype=torch.long, device=target_device)
        pred_image = torch.tensor(pred_embedding, dtype=self.dtype, device=target_device)
        if self.return_dataset_name:
            return context_images, context_labels, pred_image, int(pred_label), dataset_name

        return context_images, context_labels, pred_image, int(pred_label)


class HierarchicalDuckDBEpisodicDatasetCashed(_BaseDuckDBEpisodicDataset):
    """
    DuckDB is the ground truth.
    No Parquet. No Arrow.
    Optional: per-label memmap (.npy) built directly by reading DuckDB rows.
    """

    def __init__(
        self,
        *,
        db_path: str,
        datasets_table: str = "datasets",
        memmap_root: Optional[str] = None,
        use_memmap: bool = True,
        memmap_dtype: Union[str, np.dtype, torch.dtype] = torch.float32,
        memmap_batch_size: int = 8192,
        force_rebuild_memmap: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(db_path=db_path, **kwargs)
        self.datasets_table = datasets_table
        self._datasets_identifier = _quote_identifier(datasets_table)

        base_dir = (
            Path(memmap_root)
            if memmap_root is not None
            else Path(db_path).with_name(f"{Path(db_path).stem}_memmap")
        )
        self.memmap_root = base_dir
        self.memmap_root.mkdir(parents=True, exist_ok=True)

        self.use_memmap = use_memmap
        self.memmap_batch_size = int(memmap_batch_size)
        self.force_rebuild_memmap = force_rebuild_memmap

        self._prepared_datasets: Optional[Dict[str, _DatasetMeta]] = None
        self._memmap_np_dtype = self._resolve_np_dtype(memmap_dtype)

        self._prepare_duckdb_metadata()

    @staticmethod
    def _resolve_np_dtype(value: Union[str, np.dtype, torch.dtype]) -> np.dtype:
        if isinstance(value, torch.dtype):
            if value in (torch.float16, torch.float32, torch.bfloat16):
                return np.dtype("float32")
            return np.dtype("float64")
        return np.dtype(value)

    @staticmethod
    def _clone_meta(meta: _DatasetMeta) -> _DatasetMeta:
        return _DatasetMeta(
            info=dict(meta.info),
            labels={name: dict(info) for name, info in meta.labels.items()},
        )

    def _prepare_duckdb_metadata(self) -> None:
        """
        Enumerate datasets and labels from DuckDB and cache:
        - label table name/identifier
        - row count
        - embedding_dim (from first row)
        - memmap_path (derived; not built here)
        """
        if self._prepared_datasets is not None:
            return

        conn = duckdb.connect(database=self.db_path, read_only=True)
        datasets: Dict[str, _DatasetMeta] = {}

        dataset_rows = conn.execute(
            f'SELECT "dataset_name", "label_table" FROM {self._datasets_identifier}'
        ).fetchall()

        print(dataset_rows)

        for dataset_name, label_table in dataset_rows:
            dataset_dir = self.memmap_root / _safe_name(dataset_name)
            dataset_dir.mkdir(parents=True, exist_ok=True)

            label_table_identifier = _quote_identifier(label_table)
            label_rows = conn.execute(
                f'SELECT "label_name", "table_name" FROM {label_table_identifier}'
            ).fetchall()

            meta = _DatasetMeta(
                info={
                    "label_table": label_table,
                    "label_identifier": label_table_identifier,
                    "memmap_dir": str(dataset_dir),
                },
                labels={},
            )

            for label_name, table_name in label_rows:
                table_identifier = _quote_identifier(table_name)

                # Count rows
                count_row = conn.execute(f"SELECT COUNT(*) FROM {table_identifier}").fetchone()
                count = int(count_row[0]) if count_row else 0

                # Infer embedding dim from first row (if any)
                if count > 0:
                    dim_row = conn.execute(
                        f'SELECT array_length("Embedding") FROM {table_identifier} LIMIT 1'
                    ).fetchone()
                    embedding_dim = int(dim_row[0]) if dim_row and dim_row[0] is not None else 0
                else:
                    embedding_dim = 0

                memmap_path = dataset_dir / f"{_safe_name(label_name)}.npy"

                meta.labels[label_name] = {
                    "table_name": table_name,
                    "identifier": table_identifier,
                    "count": count,
                    "embedding_dim": embedding_dim,
                    "memmap_path": str(memmap_path),
                }

            datasets[dataset_name] = meta

        conn.close()
        self._prepared_datasets = datasets

    def _load_datasets(self, conn: duckdb.DuckDBPyConnection) -> Dict[str, _DatasetMeta]:
        del conn  # metadata comes from cached enumeration
        self._prepare_duckdb_metadata()
        assert self._prepared_datasets is not None
        return {name: self._clone_meta(meta) for name, meta in self._prepared_datasets.items()}

    def _labels_for_dataset(
        self,
        conn: duckdb.DuckDBPyConnection,
        dataset_name: str,
        meta: _DatasetMeta,
        min_samples: int,
    ) -> Sequence[str]:
        del conn
        eligible: List[str] = []
        for label_name, info in meta.labels.items():
            if int(info.get("count", 0)) >= min_samples:
                eligible.append(label_name)
        return eligible

    def _ensure_memmap(
        self,
        conn: duckdb.DuckDBPyConnection,
        label_info: Dict[str, object],
    ) -> Optional[np.memmap]:
        if not self.use_memmap:
            return None

        count = int(label_info.get("count", 0))
        dim = int(label_info.get("embedding_dim", 0))
        if count <= 0 or dim <= 0:
            return None

        memmap_path = Path(str(label_info.get("memmap_path", "")))
        shape = (count, dim)

        if memmap_path.exists() and not self.force_rebuild_memmap:
            try:
                return np.memmap(memmap_path, dtype=self._memmap_np_dtype, mode="r", shape=shape)
            except (OSError, ValueError):
                # Corrupt or shape mismatch; attempt rebuild below
                pass

        # Build directly from DuckDB table
        try:
            self._build_memmap_from_duckdb(conn, label_info, memmap_path, shape)
        except Exception:
            return None

        try:
            return np.memmap(memmap_path, dtype=self._memmap_np_dtype, mode="r", shape=shape)
        except (OSError, ValueError):
            return None

    def _build_memmap_from_duckdb(
        self,
        conn: duckdb.DuckDBPyConnection,
        label_info: Dict[str, object],
        memmap_path: Path,
        shape: Tuple[int, int],
    ) -> None:
        """
        Stream rows directly from DuckDB (no Arrow) and write into a fixed-shape memmap.
        Assumes column "Embedding" is a list/array of numeric values.
        """
        table_identifier = str(label_info.get("identifier", ""))
        if not table_identifier:
            raise ValueError("Missing DuckDB table identifier in label_info")

        expected_dim = shape[1]
        memmap_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = memmap_path.with_suffix(memmap_path.suffix + ".tmp")

        # Create the output file
        out = np.memmap(temp_path, dtype=self._memmap_np_dtype, mode="w+", shape=shape)

        # Stream embeddings. Note: no ORDER BY; if you want deterministic ordering, add an explicit key.
        cur = conn.execute(f'SELECT "Embedding" FROM {table_identifier}')

        offset = 0
        while True:
            rows = cur.fetchmany(self.memmap_batch_size)
            if not rows:
                break

            # rows: List[Tuple[embedding]]
            emb_list = [r[0] for r in rows]

            # Convert to ndarray. This is Python-object heavy, but Arrow-free by request.
            batch_arr = np.asarray(emb_list, dtype=self._memmap_np_dtype)

            if batch_arr.ndim != 2 or batch_arr.shape[1] != expected_dim:
                # Common failure modes: ragged lists => object array, or inconsistent dims
                # Detect raggedness explicitly
                lengths = [len(e) if e is not None else -1 for e in emb_list]
                raise ValueError(
                    f"Inconsistent embedding dimensions while building memmap. "
                    f"Expected {expected_dim}, got lengths like {lengths[:10]}"
                )

            n = batch_arr.shape[0]
            out[offset : offset + n] = batch_arr
            offset += n

        out.flush()
        del out

        if offset != shape[0]:
            # Defensive: count metadata may be stale or table changed
            temp_path.unlink(missing_ok=True)
            raise ValueError(f"Row count mismatch: expected {shape[0]} rows, wrote {offset} rows")

        # Atomic replace
        try:
            temp_path.replace(memmap_path)
        except FileExistsError:
            temp_path.unlink(missing_ok=True)

    def _fetch_embeddings(
        self,
        conn: duckdb.DuckDBPyConnection,
        dataset_name: str,
        meta: _DatasetMeta,
        label_name: str,
        limit: int,
    ) -> Sequence[Sequence[float]]:
        label_info = meta.labels.get(label_name)
        if label_info is None:
            raise KeyError(f"Label '{label_name}' metadata missing for dataset '{dataset_name}'")

        memmap = self._ensure_memmap(conn, label_info)
        if memmap is not None:
            total = memmap.shape[0]
            if total < limit:
                return []
            indices = torch.randperm(total, generator=self._gen)[:limit]
            return np.asarray(memmap[indices.numpy()])

        # No memmap available: sample directly from DuckDB (still no Parquet, no Arrow)
        table_identifier = str(label_info.get("identifier", ""))
        if not table_identifier:
            raise KeyError(f"DuckDB table identifier missing for label '{label_name}'")

        # Random sampling via DuckDB. This may be slow for huge tables; memmap is the intended fast path.
        rows = conn.execute(
            f'SELECT "Embedding" FROM {table_identifier} ORDER BY random() LIMIT ?',
            [int(limit)],
        ).fetchall()
        return [r[0] for r in rows]
    

def collate_hierarchical_episodes(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    context_images = torch.stack([item[0] for item in batch], dim=0)
    context_labels = torch.stack([item[1] for item in batch], dim=0)
    pred_images = torch.stack([item[2] for item in batch], dim=0)
    pred_labels = torch.tensor([item[3] for item in batch], dtype=torch.long, device=context_images.device)

    # Optionally include dataset names if provided by the dataset (len(item) == 5)
    if batch and len(batch[0]) == 5:
        dataset_names = [item[4] for item in batch]
        return context_images, context_labels, pred_images, pred_labels, dataset_names

    return context_images, context_labels, pred_images, pred_labels