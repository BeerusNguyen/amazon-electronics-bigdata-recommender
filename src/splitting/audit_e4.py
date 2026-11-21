"""Cold-start audit and reversible integer ID maps for temporal snapshots."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.splitting.prepare_e4 import BOUNDARIES


COHORTS = ("warm_user/warm_item", "cold_user/warm_item",
           "warm_user/cold_item", "cold_user/cold_item")
SUPPORT_LABELS = ("0", "1-4", "5+")


def build_id_map(history, string_id, integer_id):
    values = sorted(history[string_id].unique())
    if len(values) > np.iinfo(np.int32).max:
        raise ValueError("ID map exceeds signed 32-bit integers")
    return pd.DataFrame({string_id: values, integer_id: np.arange(len(values), dtype=np.int32)})


def held_out_groups(history, held_out):
    user_support = history.groupby("user_id").size()
    warm_users = held_out.user_id.isin(user_support.index)
    warm_items = held_out.product_id.isin(history.product_id.unique())
    tagged = held_out.assign(
        warm_user=warm_users,
        warm_item=warm_items,
        cohort=np.select(
            [warm_users & warm_items, ~warm_users & warm_items, warm_users & ~warm_items],
            COHORTS[:3], default=COHORTS[3],
        ),
        support=held_out.user_id.map(user_support).fillna(0).astype(int),
    )
    tagged["support_group"] = np.select(
        [tagged.support.eq(0), tagged.support.lt(5)], SUPPORT_LABELS[:2], default=SUPPORT_LABELS[2]
    )
    return tagged


def audit_groups(phase, tagged):
    groups = [("all", tagged)]
    groups += [(name, tagged[tagged.cohort.eq(name)]) for name in COHORTS]
    groups += [(f"history_{name}", tagged[tagged.support_group.eq(name)]) for name in SUPPORT_LABELS]
    rows = []
    for name, group in groups:
        rows.append({
            "phase": phase, "group": name, "pairs": len(group),
            "users": group.user_id.nunique(), "items": group.product_id.nunique(),
            "relevant_pairs": int(group.rating.ge(4).sum()), "total_held_out_pairs": len(tagged),
            "unknown_user_pairs": int((~group.warm_user).sum()),
            "unknown_item_pairs": int((~group.warm_item).sum()),
            "pair_share_of_held_out": len(group) / len(tagged),
        })
    return rows


def run_audit(frames, destination):
    final = destination / "audit"
    if final.exists():
        raise FileExistsError(final)
    snapshots = {
        "validation": (frames["train"], frames["validation"], BOUNDARIES[0]),
        "test": (pd.concat([frames["train"], frames["validation"]], ignore_index=True),
                 frames["test"], BOUNDARIES[1]),
    }
    for phase, (history, held_out, cutoff) in snapshots.items():
        if history.empty or held_out.empty or not (
            history.timestamp.lt(cutoff).all() and held_out.timestamp.ge(cutoff).all()
        ):
            raise ValueError(f"Invalid temporal snapshot for {phase}")

    with TemporaryDirectory(prefix="e4-audit-", dir=destination) as work:
        staged = Path(work) / "audit"
        staged.mkdir()
        rows, summary = [], {}
        for phase, (history, held_out, cutoff) in snapshots.items():
            tagged = held_out_groups(history, held_out)
            rows.extend(audit_groups(phase, tagged))
            phase_dir = staged / phase
            phase_dir.mkdir()
            for filename, key, index in (("user", "user_id", "user_idx"), ("item", "product_id", "item_idx")):
                build_id_map(history, key, index).to_parquet(phase_dir / f"{filename}_id_map.parquet", index=False)
            summary[phase] = {
                "snapshot_utc_seconds": cutoff,
                "history_pairs": len(history),
                "history_users": history.user_id.nunique(),
                "history_items": history.product_id.nunique(),
                "held_out_pairs": len(held_out),
                "held_out_users": held_out.user_id.nunique(),
                "held_out_items": held_out.product_id.nunique(),
                "unknown_users": tagged.loc[~tagged.warm_user, "user_id"].nunique(),
                "unknown_items": tagged.loc[~tagged.warm_item, "product_id"].nunique(),
            }
            for entity in ("users", "items"):
                summary[phase][f"unknown_{entity}_rate"] = (
                    summary[phase][f"unknown_{entity}"] / summary[phase][f"held_out_{entity}"]
                )
        pd.DataFrame(rows).to_csv(staged / "cold_start.csv", index=False)
        (staged / "snapshot_manifest.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        staged.rename(final)
