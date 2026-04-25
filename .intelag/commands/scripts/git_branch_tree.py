#!/usr/bin/env python3
"""
Plot Git branch relationships using the Git CLI.

What it does:
- Reads local and remote branches from `git for-each-ref`
- Finds each branch's fork-point / best parent branch using merge-base
- Builds a branch-to-branch relationship graph
- Plots it with matplotlib as a simple tree/forest

Notes:
- This is branch-level visualization, not commit-level.
- Parent detection is heuristic:
  the parent of a branch is chosen as the branch with the most recent
  merge-base that is not the branch itself.
- Works best when branch names still point to meaningful tips.
- Requires:
    - git
    - python package: matplotlib

Usage:
    python plot_git_branches.py
    python plot_git_branches.py --repo /path/to/repo
    python plot_git_branches.py --include-remotes
    python plot_git_branches.py --show-main-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Branch:
    name: str
    full_ref: str
    commit: str
    is_remote: bool
    commit_unix: int


def run_git(repo: Path, args: list[str]) -> str:
    """Run a git command and return stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Git command failed:\n"
            f"  git {' '.join(args)}\n"
            f"stderr:\n{result.stderr.strip()}"
        )
    return result.stdout.strip()


def is_git_repo(repo: Path) -> bool:
    """Return True if the path is a git repository."""
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def get_branches(repo: Path, include_remotes: bool) -> list[Branch]:
    """
    Get branches using git for-each-ref.

    Fields:
    - refname
    - short refname
    - tip objectname
    - committerdate (unix)
    """
    refs: list[str] = ["refs/heads"]
    if include_remotes:
        refs.append("refs/remotes")

    fmt = "%(refname)|%(refname:short)|%(objectname)|%(committerdate:unix)"
    output = run_git(repo, ["for-each-ref", f"--format={fmt}", *refs])

    branches: list[Branch] = []
    for line in output.splitlines():
        if not line.strip():
            continue

        full_ref, short_name, commit, commit_unix_str = line.split("|", 3)

        # Skip symbolic remote HEAD refs like origin/HEAD
        if short_name.endswith("/HEAD"):
            continue

        is_remote = full_ref.startswith("refs/remotes/")
        branches.append(
            Branch(
                name=short_name,
                full_ref=full_ref,
                commit=commit,
                is_remote=is_remote,
                commit_unix=int(commit_unix_str or "0"),
            )
        )

    return branches


def get_current_branch(repo: Path) -> str | None:
    """Return current checked-out branch name, or None if detached."""
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    name = result.stdout.strip()
    return name or None


def get_default_branch(repo: Path, branches: list[Branch]) -> str | None:
    """
    Try to infer the default/main trunk branch.
    Preference:
    - main
    - master
    - dev
    - current branch
    - oldest/local-most-likely
    """
    names = {b.name for b in branches}

    for candidate in ("main", "master", "dev"):
        if candidate in names:
            return candidate

    current = get_current_branch(repo)
    if current and current in names:
        return current

    local_branches = [b for b in branches if not b.is_remote]
    if local_branches:
        # Oldest branch tip as a fallback heuristic
        local_branches.sort(key=lambda b: b.commit_unix)
        return local_branches[0].name

    if branches:
        branches = sorted(branches, key=lambda b: b.commit_unix)
        return branches[0].name

    return None


def get_merge_base(repo: Path, a: Branch, b: Branch) -> str | None:
    """Return merge-base commit hash for two branches, or None."""
    result = subprocess.run(
        ["git", "merge-base", a.name, b.name],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    mb = result.stdout.strip()
    return mb or None


def get_merge_base_time(repo: Path, a: Branch, b: Branch) -> int | None:
    """
    Return merge-base commit time (unix) for two branches, or None.
    """
    merge_base = get_merge_base(repo, a, b)
    if merge_base is None:
        return None

    try:
        ts = run_git(repo, ["show", "-s", "--format=%ct", merge_base])
        return int(ts)
    except Exception:
        return None


def pick_parent_branch(
    repo: Path,
    branch: Branch,
    candidates: list[Branch],
    root_name: str | None,
) -> str | None:
    """
    Pick the most likely parent branch for `branch`.

    Heuristic:
    - Exclude itself
    - Prefer candidates with the latest merge-base time
    - Slightly prefer trunk branches if very close
    """
    best_parent: str | None = None
    best_score: tuple[int, int] | None = None

    trunk_bonus_names = {"main", "master", "dev"}
    if root_name:
        trunk_bonus_names.add(root_name)

    for candidate in candidates:
        if candidate.name == branch.name:
            continue

        merge_base_commit = get_merge_base(repo, branch, candidate)
        if merge_base_commit is None:
            continue

        # If merge-base == candidate's tip, the candidate is fully contained
        # in our branch — it's a descendant, not a parent. Skip it, unless
        # it's a trunk branch (whose tips advance via merges).
        if (merge_base_commit == candidate.commit
                and candidate.name not in trunk_bonus_names):
            continue

        try:
            ts_str = run_git(repo, ["show", "-s", "--format=%ct", merge_base_commit])
            merge_base_time = int(ts_str)
        except Exception:
            continue

        # Trunk branches get a time bonus (7 days in seconds) so they win
        # when merge-base times are close, but not when a real parent has
        # a significantly newer merge-base.
        trunk_time_bonus = 7 * 86400  # 7 days
        time_bonus = trunk_time_bonus if candidate.name in trunk_bonus_names else 0
        score = (merge_base_time + time_bonus, 1 if candidate.name in trunk_bonus_names else 0)

        if best_score is None or score > best_score:
            best_score = score
            best_parent = candidate.name

    return best_parent


def build_branch_tree(
    repo: Path,
    branches: list[Branch],
    root_name: str | None,
    show_main_only: bool,
) -> tuple[dict[str, list[str]], dict[str, str | None]]:
    """
    Build a parent-child relationship map between branches.
    """
    by_name = {b.name: b for b in branches}
    parent_of: dict[str, str | None] = {}
    children_of: dict[str, list[str]] = defaultdict(list)

    candidate_branches = list(branches)

    for branch in branches:
        if branch.name == root_name:
            parent_of[branch.name] = None
            continue

        parent = pick_parent_branch(
            repo=repo,
            branch=branch,
            candidates=candidate_branches,
            root_name=root_name,
        )

        if show_main_only and root_name:
            # Collapse any branch not ultimately connected to root onto root
            if parent is None:
                parent = root_name

        parent_of[branch.name] = parent
        if parent is not None:
            children_of[parent].append(branch.name)

    # Ensure every branch has an entry in children_of
    for b in by_name:
        children_of.setdefault(b, [])

    # Sort children by commit time, then name
    for _, children in children_of.items():
        children.sort(key=lambda n: (by_name[n].commit_unix, n.lower()))

    return dict(children_of), parent_of


def compute_layout(
    children_of: dict[str, list[str]],
    all_nodes: Iterable[str],
    root_nodes: list[str],
) -> dict[str, tuple[float, float]]:
    """
    Compute simple tree layout.

    x: depth
    y: ordered leaf position
    """
    positions: dict[str, tuple[float, float]] = {}
    visited = set()
    next_y = 0.0

    def dfs(node: str, depth: int) -> float:
        nonlocal next_y
        if node in visited:
            # Cycle detected or already visited
            if node in positions:
                return positions[node][1]
            return next_y

        visited.add(node)
        children = children_of.get(node, [])

        if not children:
            y = float(next_y)
            next_y += 1
            positions[node] = (float(depth), y)
            return y

        child_ys = []
        for child in children:
            if child not in visited:
                child_ys.append(dfs(child, depth + 1))
            elif child in positions:
                child_ys.append(positions[child][1])

        if not child_ys:
            y = float(next_y)
            next_y += 1
            positions[node] = (float(depth), y)
            return y

        y = sum(child_ys) / len(child_ys)
        positions[node] = (float(depth), y)
        return y

    # First pass: process identified roots
    for root in root_nodes:
        if root not in visited:
            dfs(root, 0)
            next_y += 1.0  # spacing between trees

    # Second pass: process any nodes missed (due to cycles or disconnected components)
    for node in sorted(all_nodes):
        if node not in visited:
            dfs(node, 0)
            next_y += 1.0

    return positions


def human_time(unix_ts: int) -> str:
    """Format unix timestamp."""
    dt = datetime.fromtimestamp(unix_ts, tz=datetime.UTC)
    return dt.strftime("%Y-%m-%d")


def get_remote_branch_names(repo: Path) -> set[str]:
    """Return the set of local branch names that have a remote counterpart.

    Queries refs/remotes directly so it works even when --include-remotes
    is not passed (i.e. when the remote branches are not in the branches list).
    """
    result = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname:short)", "refs/remotes"],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    remote_short: set[str] = set()
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line or line.endswith("/HEAD"):
            continue
        # "origin/feature-x" → "feature-x"
        parts = line.split("/", 1)
        if len(parts) == 2:
            remote_short.add(parts[1])
    return remote_short


def plot_branch_tree(
    branches: list[Branch],
    parent_of: dict[str, str | None],
    current_branch: str | None,
    root_name: str | None,
    title: str,
    repo: Path | None = None,
) -> None:
    """Plot branch tree: X = depth (parent->child), Y = rows sorted by time (bottom=old)."""
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    # ── Detect remote counterparts ──
    remote_short_names = get_remote_branch_names(repo) if repo else set()
    local_names = {b.name for b in branches if not b.is_remote}
    has_remote = local_names & remote_short_names

    # ── Filter: only LOCAL branches in the plot ──
    local_branches = [b for b in branches if not b.is_remote]
    local_by_name = {b.name: b for b in local_branches}

    # Rebuild parent/children limited to local branches
    local_set = set(local_by_name)
    local_parent_of: dict[str, str | None] = {}
    local_children_of: dict[str, list[str]] = defaultdict(list)
    for name in local_set:
        p = parent_of.get(name)
        while p is not None and p not in local_set:
            p = parent_of.get(p)
        local_parent_of[name] = p
        if p is not None:
            local_children_of[p].append(name)
    for name in local_set:
        local_children_of.setdefault(name, [])
    for _, kids in local_children_of.items():
        kids.sort(key=lambda n: (local_by_name[n].commit_unix, n.lower()))

    # ── Colour palette — unique colour per branch ──
    BG = "#ffffff"
    MUTED = "#656d76"

    # A broad palette of distinguishable (fill, border) pairs
    _PALETTE = [
        ("#dbeafe", "#2563eb"),  # blue
        ("#d4edda", "#16a34a"),  # green
        ("#fce7f3", "#db2777"),  # pink
        ("#f3e8ff", "#7c3aed"),  # purple
        ("#fef3c7", "#d97706"),  # amber
        ("#ccfbf1", "#0d9488"),  # teal
        ("#fee2e2", "#dc2626"),  # red
        ("#e0e7ff", "#4f46e5"),  # indigo
        ("#fae8ff", "#c026d3"),  # fuchsia
        ("#dcfce7", "#15803d"),  # emerald
        ("#fff7ed", "#ea580c"),  # orange
        ("#f0fdfa", "#0f766e"),  # cyan-ish
        ("#fdf2f8", "#be185d"),  # rose
        ("#ede9fe", "#6d28d9"),  # violet
        ("#ecfdf5", "#059669"),  # mint
        ("#fff1f2", "#e11d48"),  # crimson
    ]

    # Assign a unique colour to each local branch
    _branch_color: dict[str, tuple[str, str]] = {}
    for i, name in enumerate(sorted(local_set)):
        _branch_color[name] = _PALETTE[i % len(_PALETTE)]

    def _style(name: str) -> tuple[str, str, str, str]:
        """(fill, border, text_color, tag)"""
        fill, border = _branch_color.get(name, _PALETTE[0])
        tag = ""
        if name == root_name:
            tag = "root"
        elif name == current_branch:
            tag = "HEAD"
        elif name in has_remote:
            tag = "remote"
        return (fill, border, "#1a1a1a", tag)

    # ── Compute creation date per branch (merge-base with parent) ──
    import numpy as np

    creation_unix: dict[str, int] = {}
    for name in local_set:
        parent = local_parent_of.get(name)
        if parent is None or repo is None:
            # Root or orphan: use the repo's first commit as creation date
            try:
                first_ts = run_git(repo or Path(), [
                    "log", "--reverse", "--format=%ct", name, "--",
                ])
                first_line = first_ts.splitlines()[0].strip() if first_ts else ""
                creation_unix[name] = int(first_line) if first_line else local_by_name[name].commit_unix
            except Exception:
                creation_unix[name] = local_by_name[name].commit_unix
        else:
            mb_time = get_merge_base_time(repo, local_by_name[name], local_by_name[parent])
            creation_unix[name] = mb_time if mb_time is not None else local_by_name[name].commit_unix

    # ── X = creation date, Y = last-update date ──
    all_create = [creation_unix[n] for n in local_set]
    all_update = [local_by_name[n].commit_unix for n in local_set]
    min_cx, max_cx = min(all_create), max(all_create)
    min_uy, max_uy = min(all_update), max(all_update)
    cx_span = max_cx - min_cx if max_cx != min_cx else 1.0
    uy_span = max_uy - min_uy if max_uy != min_uy else 1.0

    n = len(local_branches)
    card_w = 3.0
    card_h = 0.65
    x_range = max(12.0, n * 1.5)
    y_range = max(8.0, n * 1.0)

    def _cx(unix_ts: int) -> float:
        return ((unix_ts - min_cx) / cx_span) * x_range

    def _uy(unix_ts: int) -> float:
        return ((unix_ts - min_uy) / uy_span) * y_range

    # Raw positions
    x_of: dict[str, float] = {}
    y_of: dict[str, float] = {}
    for name in local_set:
        x_of[name] = _cx(creation_unix[name])
        y_of[name] = _uy(local_by_name[name].commit_unix)

    # ── Collision avoidance — stagger cards that share similar dates ──
    # Group by similar creation time and spread both X and Y
    names_list = sorted(local_set, key=lambda n: (x_of[n], y_of[n]))

    # Pass 1: stagger X for branches with the same creation date
    # (group by close X values and add horizontal offset)
    x_groups: dict[int, list[str]] = defaultdict(list)
    for name in names_list:
        # Round creation time to nearest day for grouping
        day_key = creation_unix[name] // 86400
        x_groups[day_key].append(name)

    for _day_key, group in x_groups.items():
        if len(group) <= 1:
            continue
        # Sort by update time within the group
        group.sort(key=lambda n: local_by_name[n].commit_unix)
        # Spread them horizontally with a staircase offset
        x_step = card_w * 0.4
        base_x = x_of[group[0]]
        for i, name in enumerate(group):
            x_of[name] = base_x + i * x_step

    # Pass 2: ensure no Y overlap for cards that are close in X
    names_list = sorted(local_set, key=lambda n: (y_of[n], x_of[n]))
    for _pass in range(8):
        moved = False
        for i in range(len(names_list)):
            for j in range(i + 1, len(names_list)):
                a, b = names_list[i], names_list[j]
                dx = abs(x_of[a] - x_of[b])
                dy = abs(y_of[a] - y_of[b])
                if dx < card_w * 1.1 and dy < card_h * 1.5:
                    nudge = (card_h * 1.6 - dy) / 2
                    y_of[a] -= nudge * 0.5
                    y_of[b] += nudge * 0.5
                    moved = True
        if not moved:
            break

    # Sort for legend / labels
    sorted_names = sorted(local_set, key=lambda n: (y_of[n], n.lower()))

    # ── Figure ──
    all_x = list(x_of.values())
    all_y = list(y_of.values())
    fig_w = max(14, (max(all_x) - min(all_x)) * 0.9 + card_w * 3 + 4)
    fig_h = max(7, (max(all_y) - min(all_y)) * 0.9 + card_h * 3 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # ── Connectors (straight arrows between closest card edges) ──
    def _card_edge(name: str, target_x: float, target_y: float) -> tuple[float, float]:
        """Return the point on the edge of `name`'s card closest to (target_x, target_y)."""
        cx = x_of[name] + card_w / 2
        cy = y_of[name]
        dx = target_x - cx
        dy = target_y - cy
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return (cx + card_w / 2, cy)

        # Scale to hit card boundary (half-widths)
        hw, hh = card_w / 2, card_h / 2
        sx = hw / abs(dx) if abs(dx) > 1e-9 else float("inf")
        sy = hh / abs(dy) if abs(dy) > 1e-9 else float("inf")
        s = min(sx, sy)
        return (cx + dx * s, cy + dy * s)

    for child, parent in local_parent_of.items():
        if parent is None:
            continue
        _, p_border, _, _ = _style(parent)

        # Compute edge-to-edge connection points
        p_center = (x_of[parent] + card_w / 2, y_of[parent])
        c_center = (x_of[child] + card_w / 2, y_of[child])
        start = _card_edge(parent, *c_center)
        end = _card_edge(child, *p_center)

        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": p_border,
                "lw": 2.0,
                "mutation_scale": 14,
                "connectionstyle": "arc3,rad=0.0",
            },
            zorder=1,
        )

    # ── Cards ──
    for branch in local_branches:
        name = branch.name
        x = x_of[name]
        y = y_of[name]
        fill, border, txtc, tag = _style(name)

        lw = 2.4 if (name == current_branch or name in has_remote) else 1.5

        card = FancyBboxPatch(
            (x, y - card_h / 2), card_w, card_h,
            boxstyle="round,pad=0.06,rounding_size=0.10",
            facecolor=fill, edgecolor=border,
            linewidth=lw, zorder=4,
        )
        if name == current_branch:
            card.set_path_effects([
                pe.withSimplePatchShadow(offset=(0, -0.02),
                                         shadow_rgbFace=border, alpha=0.20),
            ])
        ax.add_patch(card)

        # Name
        display = name if len(name) <= 26 else name[:23] + "..."
        ax.text(x + card_w / 2, y + 0.09, display,
                fontsize=9.5, fontweight="bold", color=txtc,
                fontfamily="monospace", ha="center", va="center", zorder=6)

        # Dates: "created → updated"
        created_str = human_time(creation_unix[name])
        updated_str = human_time(branch.commit_unix)
        date_label = f"{created_str} \u2192 {updated_str}"
        ax.text(x + card_w / 2, y - 0.15, date_label,
                fontsize=7, color=MUTED, fontfamily="monospace",
                ha="center", va="center", zorder=6)

        # Badges
        badges: list[tuple[str, str]] = []
        if tag == "root":
            badges.append(("root", "#1a7f37"))
        if tag == "HEAD":
            badges.append(("HEAD", "#0550ae"))
        if name in has_remote:
            badges.append(("remote", "#cf222e"))
        for bi, (blabel, bcolor) in enumerate(badges):
            bx = x + card_w - 0.38 - bi * 0.72
            by = y + card_h / 2 + 0.01
            pill = FancyBboxPatch(
                (bx - 0.30, by - 0.10), 0.60, 0.20,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=bcolor, edgecolor="none", alpha=0.90, zorder=7)
            ax.add_patch(pill)
            ax.text(bx, by, blabel, fontsize=5, fontweight="bold",
                    color="#ffffff", ha="center", va="center",
                    fontfamily="monospace", zorder=8)

    # ── Axes ──
    ax.set_title(title, color="#1a1a1a", fontsize=14,
                 fontweight="bold", pad=16, fontfamily="monospace")

    x_pad = card_w * 0.8
    y_pad = card_h * 1.5
    ax.set_xlim(min(all_x) - x_pad, max(all_x) + card_w + x_pad)
    ax.set_ylim(min(all_y) - y_pad, max(all_y) + y_pad)

    # X-axis: creation date ticks
    tick_count_x = min(8, max(3, n))
    tick_create = np.linspace(min_cx, max_cx, tick_count_x).astype(int)
    ax.set_xticks([_cx(int(t)) for t in tick_create])
    ax.set_xticklabels([human_time(int(t)) for t in tick_create],
                        fontsize=7.5, color="#000000", fontfamily="monospace",
                        rotation=30, ha="right")
    ax.tick_params(axis="x", length=3, pad=4, colors="#000000")

    # Y-axis: update date ticks
    tick_count_y = min(8, max(3, n))
    tick_update = np.linspace(min_uy, max_uy, tick_count_y).astype(int)
    ax.set_yticks([_uy(int(t)) for t in tick_update])
    ax.set_yticklabels([human_time(int(t)) for t in tick_update],
                        fontsize=7.5, color="#000000", fontfamily="monospace")
    ax.tick_params(axis="y", length=3, pad=4, colors="#000000")

    # Axis labels
    ax.set_xlabel("Branch Created", fontsize=10, color="#000000",
                  fontfamily="monospace", fontweight="bold", labelpad=10)
    ax.set_ylabel("Last Updated", fontsize=10, color="#000000",
                  fontfamily="monospace", fontweight="bold", labelpad=10)

    # Faint gridlines
    for tx in [_cx(int(t)) for t in tick_create]:
        ax.axvline(tx, color="#f0f0f0", lw=0.5, zorder=0)
    for ty in [_uy(int(t)) for t in tick_update]:
        ax.axhline(ty, color="#f0f0f0", lw=0.5, zorder=0)

    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend — one entry per branch, using its unique colour
    legend_items = []
    for name in sorted_names:
        fill, border = _branch_color.get(name, _PALETTE[0])
        suffix = ""
        if name == root_name:
            suffix = " [root]"
        elif name == current_branch:
            suffix = " [HEAD]"
        elif name in has_remote:
            suffix = " [remote]"
        legend_items.append(
            Line2D([0], [0], marker="s", color="none",
                   markerfacecolor=fill, markersize=8,
                   markeredgecolor=border, markeredgewidth=1.5,
                   label=f"{name}{suffix}"),
        )
    legend = ax.legend(handles=legend_items, loc="lower right", fontsize=7,
                       facecolor="#fafbfc", edgecolor="#e1e4e8",
                       labelcolor="#1a1a1a", framealpha=0.95,
                       ncol=1 if n <= 10 else 2)
    legend.get_frame().set_linewidth(0.6)

    plt.tight_layout()
    plt.show()



def print_text_tree(
    children_of: dict[str, list[str]],
    parent_of: dict[str, str | None],
    current_branch: str | None,
    root_name: str | None,
    branches: list[Branch] | None = None,
    has_remote: set[str] | None = None,
) -> None:
    """Print a 2D box-drawing branch tree.

    Each node is rendered as a box:
        ╔══════════════════════╗
        ║  branch-name   [HD] ║
        ║  2026-03-09    [RM] ║
        ╚══════════════════════╝
    """
    by_name = {b.name: b for b in branches} if branches else {}
    if has_remote is None:
        has_remote = set()

    def _tags(node: str) -> list[str]:
        tags: list[str] = []
        if node == root_name:
            tags.append("RT")
        if node == current_branch:
            tags.append("HD")
        if node in has_remote:
            tags.append("RM")
        return tags

    def _make_box(node: str) -> list[str]:
        """Return a list of strings representing the box for a node."""
        tags = _tags(node)
        tag_str = " ".join(f"[{t}]" for t in tags)
        name_line = node
        date_line = human_time(by_name[node].commit_unix) if node in by_name else ""

        # Use double lines for branches with remotes, single for others
        is_double = "RM" in tags
        if is_double:
            tl, tr, bl, br = "╔", "╗", "╚", "╝"
            hl, vl = "═", "║"
        else:
            tl, tr, bl, br = "┌", "┐", "└", "┘"
            hl, vl = "─", "│"

        # Compute inner width
        content_lines = [
            f"  {name_line}  {tag_str}".rstrip(),
            f"  {date_line}" if date_line else "",
        ]
        content_lines = [ln for ln in content_lines if ln]
        inner_w = max(len(ln) for ln in content_lines)
        inner_w = max(inner_w, 16)  # minimum width

        top = f"{tl}{hl * (inner_w + 2)}{tr}"
        bot = f"{bl}{hl * (inner_w + 2)}{br}"
        rows = [f"{vl} {ln:<{inner_w}} {vl}" for ln in content_lines]
        return [top, *rows, bot]

    # ── Build grid: columns by depth, rows by tree order ──
    root_nodes = [name for name, parent in parent_of.items() if parent is None]
    root_nodes.sort(key=lambda n: (by_name[n].commit_unix if n in by_name else 0, n.lower()))

    # Flatten tree into ordered list with depths
    ordered: list[tuple[str, int]] = []  # (node, depth)
    visited: set[str] = set()

    def _flatten(node: str, depth: int) -> None:
        if node in visited:
            return
        visited.add(node)
        ordered.append((node, depth))
        for kid in children_of.get(node, []):
            _flatten(kid, depth + 1)

    for root in root_nodes:
        _flatten(root, 0)
    for node in sorted(parent_of):
        if node not in visited:
            _flatten(node, 0)

    if not ordered:
        print("(no branches)")
        return

    max_depth = max(d for _, d in ordered)

    # Pre-render all boxes
    boxes: dict[str, list[str]] = {}
    box_widths: dict[int, int] = {}  # max box width per depth column
    for node, depth in ordered:
        box = _make_box(node)
        boxes[node] = box
        w = max(len(ln) for ln in box)
        box_widths[depth] = max(box_widths.get(depth, 0), w)

    # Column x-offsets (character positions)
    col_gap = 5  # gap between columns for connectors
    col_x: dict[int, int] = {}
    x = 0
    for d in range(max_depth + 1):
        col_x[d] = x
        x += box_widths.get(d, 16) + col_gap

    total_w = x

    # ── Render onto a character canvas ──
    # Each node box takes box_height rows + 1 row gap
    box_heights = {node: len(lines) for node, lines in boxes.items()}
    row_gap = 1

    # Assign row positions (top-left y of each box)
    node_row: dict[str, int] = {}
    cur_row = 0
    for node, _depth in ordered:
        node_row[node] = cur_row
        cur_row += box_heights[node] + row_gap

    total_h = cur_row

    # Create canvas
    canvas: list[list[str]] = [[" "] * total_w for _ in range(total_h)]

    def _put(r: int, c: int, text: str) -> None:
        for i, ch in enumerate(text):
            if 0 <= r < total_h and 0 <= c + i < total_w:
                canvas[r][c + i] = ch

    # Draw boxes
    for node, depth in ordered:
        bx = col_x[depth]
        by_ = node_row[node]
        for i, line in enumerate(boxes[node]):
            _put(by_ + i, bx, line)

    # Draw connectors: parent right-edge → child left-edge
    for node, depth in ordered:
        parent = parent_of.get(node)
        if parent is None or parent not in node_row:
            continue

        p_depth = next(d for n, d in ordered if n == parent)
        p_box_w = box_widths.get(p_depth, 16)
        p_row = node_row[parent]
        p_mid_row = p_row + box_heights[parent] // 2  # connector row on parent

        c_row = node_row[node]
        c_mid_row = c_row + box_heights[node] // 2  # connector row on child

        # Start x: right edge of parent box
        x_start = col_x[p_depth] + p_box_w
        # End x: left edge of child box
        x_end = col_x[depth]

        if x_start >= x_end:
            continue

        mid_x = (x_start + x_end) // 2

        # Horizontal from parent to mid
        for c in range(x_start, mid_x):
            _put(p_mid_row, c, "─")

        # Vertical from parent row to child row
        r_top = min(p_mid_row, c_mid_row)
        r_bot = max(p_mid_row, c_mid_row)
        for r in range(r_top, r_bot + 1):
            if canvas[r][mid_x] == "─":
                _put(r, mid_x, "┼")
            elif r == r_top and p_mid_row <= c_mid_row:
                _put(r, mid_x, "┐" if p_mid_row == r_top else "│")
            elif r == r_bot and p_mid_row <= c_mid_row:
                _put(r, mid_x, "└")
            elif r == r_top and p_mid_row > c_mid_row:
                _put(r, mid_x, "┘" if c_mid_row == r_top else "│")
            elif r == r_bot and p_mid_row > c_mid_row:
                _put(r, mid_x, "┌")
            else:
                if canvas[r][mid_x] == " ":
                    _put(r, mid_x, "│")

        # Horizontal from mid to child
        for c in range(mid_x + 1, x_end):
            _put(c_mid_row, c, "─")

    # Print legend
    print("  Tags: [RT] Root  [HD] HEAD  [RM] Has Remote")
    print()

    # Print canvas (strip trailing spaces)
    for row in canvas:
        line = "".join(row).rstrip()
        if line:
            print(line)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Plot Git branch relationships using the Git CLI.",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Path to the git repository. Default: current directory.",
    )
    parser.add_argument(
        "--include-remotes",
        action="store_true",
        help="Include remote branches (refs/remotes/*).",
    )
    parser.add_argument(
        "--show-main-only",
        action="store_true",
        help="Force disconnected branches under the inferred root branch.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print the ASCII tree, do not open a matplotlib window.",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    repo = args.repo.resolve()

    if not repo.exists():
        print(f"Error: repo path does not exist: {repo}", file=sys.stderr)
        return 1

    if not is_git_repo(repo):
        print(f"Error: not a git repository: {repo}", file=sys.stderr)
        return 1

    try:
        branches = get_branches(repo, include_remotes=args.include_remotes)
    except Exception as exc:
        print(f"Error while reading branches: {exc}", file=sys.stderr)
        return 1

    if not branches:
        print("No branches found.", file=sys.stderr)
        return 1

    root_name = get_default_branch(repo, branches)
    current_branch = get_current_branch(repo)

    try:
        children_of, parent_of = build_branch_tree(
            repo=repo,
            branches=branches,
            root_name=root_name,
            show_main_only=args.show_main_only,
        )
    except Exception as exc:
        print(f"Error while building branch tree: {exc}", file=sys.stderr)
        return 1

    remote_names = get_remote_branch_names(repo)
    local_names = {b.name for b in branches if not b.is_remote}
    has_remote = local_names & remote_names

    print_text_tree(
        children_of=children_of,
        parent_of=parent_of,
        current_branch=current_branch,
        root_name=root_name,
        branches=branches,
        has_remote=has_remote,
    )

    if not args.no_plot:
        plot_branch_tree(
            branches=branches,
            parent_of=parent_of,
            current_branch=current_branch,
            root_name=root_name,
            title=f"Git Branch Relationships: {repo.name}",
            repo=repo,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
