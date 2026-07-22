#!/usr/bin/env python3
"""Generate Spanish (``-es.ipynb``) siblings of the English presentation notebooks.

The English ``.ipynb`` decks are the single source of truth. This script produces a
Spanish deck for each, translating only human-language prose and leaving code cells,
outputs, links, paths, and Quarto/Pandoc directives untouched. Quarto's normal website
build then renders both languages.

**Incremental + edit-safe.** A fluent Spanish speaker reviews and corrects the generated
decks. Re-running this script must never overwrite those corrections. Each generated
Spanish markdown cell carries a provenance record in its metadata::

    metadata.translation = {
        "source_sha256":  <sha256 of the English source cell it was produced from>,
        "machine_sha256": <sha256 of the machine Spanish output when generated>,
        "model": ..., "translated_at": ..., "needs_review": <bool>,
    }

Per-cell decision on each run (cells pair by their stable ``id``):

* new English cell                                  -> translate
* English unchanged (source hash matches)           -> reuse existing Spanish verbatim
* English changed, Spanish NOT human-edited         -> re-translate
* English changed, Spanish WAS human-edited         -> keep human Spanish, flag needs_review

``--accept-reviewed`` re-anchors the hashes for flagged cells (treat the current human
Spanish as the approved translation of the current English) and clears the flag.

Usage::

    pixi run python src/translate_notebooks.py                 # all decks
    pixi run python src/translate_notebooks.py --only presentation-3-data-preparation.ipynb
    pixi run python src/translate_notebooks.py --accept-reviewed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic

try:
    import yaml  # PyYAML — used only to verify frontmatter keys survived translation
except Exception:  # pragma: no cover - yaml is normally present via jupyter
    yaml = None

MODEL = "claude-sonnet-5"
REPO = Path(__file__).resolve().parent.parent

DECKS = [
    "presentation-1-workflow-overview.ipynb",
    "presentation-2-setup.ipynb",
    "presentation-3-data-preparation.ipynb",
    "presentation-4-workflow-ruritania.ipynb",
    "presentation-5-workflow-colombia.ipynb",
]

SYSTEM_PROSE = """You are a professional translator localizing reveal.js slide decks for a \
scientific / GIS workshop audience. You are given one markdown cell between `<source>` and \
`</source>` tags. Translate its English prose into natural, fluent Spanish and return ONLY \
the translated markdown — no preamble, no commentary, no questions, and do NOT echo the \
`<source>`/`</source>` tags. The source ALWAYS contains content (even if short); never ask \
for content and never say the content is missing.

Translate natural-language prose only. Keep the following VERBATIM (byte-for-byte):
- fenced code blocks and inline code spans (anything in backticks)
- raw HTML / CSS and ```{=html} blocks
- URLs, and image / file paths (e.g. images/foo.png, data/bar.parquet)
- Quarto / Pandoc fenced-div markers and their attributes: ::: , :::: , {.columns}, \
{.column width="60%"}, {.scrollable}, {height="500"}, #| echo: true, etc.
- YAML keys and structure
- numbers, identifiers, and technical tokens

Do NOT translate proper nouns or dataset names: IDEAM, MEC, GeoParquet, Source Cooperative, \
Colombia, Bogotá, Ruritania, and similar names stay as-is.

Preserve the markdown structure exactly: same headings, same list/table shape, same line \
breaks, and the same leading/trailing blank lines. If the cell contains no translatable \
prose (e.g. it is only HTML/CSS or a config block), return it completely unchanged."""

SYSTEM_FRONTMATTER = """You are localizing the YAML frontmatter of a Quarto reveal.js slide \
deck into Spanish. You are given the frontmatter between `<source>` and `</source>` tags. \
Return ONLY the YAML block, including its opening and closing `---` fences — no commentary, \
no questions, and do NOT echo the `<source>`/`</source>` tags.

Translate into Spanish ONLY the human-readable string VALUES of these keys: `title`, \
`subtitle`, and `footer`. Leave every other value unchanged. Leave ALL keys unchanged. Do \
not add, remove, reorder, or reindent any keys. Do not touch `author`, `date`, `format`, \
`theme`, `css`, `execute`, or any other structural field. Keep the `---` fences."""

# Phrases that mark a conversational non-translation (the model replying to us instead of
# translating). If a response matches, we retry, then hard-flag — never ship it silently.
_META = re.compile(
    r"haven't included|have not included|please paste|paste the (file|content|text|markdown)"
    r"|could you (please )?(paste|provide|share)|you'?d like translated|the actual markdown"
    r"|i'?ll (inspect|provide|translate)|once i can see|i'?m happy to|feel free to"
    r"|here'?s the translation|here is the translation|as an ai|i cannot|i can'?t"
    # leaked scaffolding / self-talk: the model echoing the <source> tags or
    # narrating its own process instead of returning only the translation.
    r"|</?source>|wait,? i (need|should|have|must)|let me (translate|fix)|i (need|have) to translate",
    re.I,
)


class TranslationError(RuntimeError):
    pass

NOW = datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def cell_text(cell: dict) -> str:
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def as_lines(text: str) -> list[str]:
    """Notebook markdown source must be a LIST of lines, never a single string —
    a single-string source makes Quarto flatten fenced divs and render ``:::``
    literally. (This bug was hit and fixed earlier in this repo.)"""
    return text.splitlines(keepends=True)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# English speaker notes
#
# Every translated *content* slide carries the English text in its reveal.js
# speaker notes (a ``::: {.notes}`` fenced div) so a Spanish presenter can read
# the original while presenting. The notes are generated DETERMINISTICALLY from
# the English source — no API call — and re-generated on every run, so they cost
# nothing and never drift from the English.
#
# Because the notes are deterministic, they are NOT part of the translation the
# human reviews: all provenance hashing is done over the *prose only* (the
# Spanish with the auto-notes block stripped). That keeps the human-edit
# detection working exactly as before and lets existing note-less decks migrate
# cleanly (stripping a non-existent notes block is a no-op).
# ---------------------------------------------------------------------------
NOTES_MARKER = "<!-- en-source: auto-generated speaker notes, do not edit -->"
# Matches exactly the single separator newline + the notes block (not base's own
# trailing newline), so prose_only(compose_source(p, ...)) == base round-trips.
_AUTO_NOTES_RE = re.compile(
    r"\n::: \{\.notes\}\n" + re.escape(NOTES_MARKER) + r"\n.*?\n:::\n?$",
    re.DOTALL,
)


def prose_only(text: str) -> str:
    """Return ``text`` with any auto-generated speaker-notes block removed.

    Idempotent and a no-op on text that has no auto-notes block (e.g. an
    existing pre-notes Spanish cell), so hashing over ``prose_only`` matches the
    old whole-cell hashes for un-migrated decks."""
    return _AUTO_NOTES_RE.sub("", text)


def en_notes_prose(en_text: str) -> str:
    """Reduce an English markdown cell to clean presenter prose.

    Strips fenced code / ``{=html}`` blocks, images, Quarto fenced-div markers
    (``:::`` / ``::::``), raw HTML tags, and Pandoc attribute braces — leaving
    headings and prose (inline links/bold kept). Returns "" if nothing remains."""
    t = re.sub(r"```.*?```", "", en_text, flags=re.DOTALL)  # fenced code / html
    # linked image: [ ![alt](src){attrs?} ]( href ){attrs?}
    t = re.sub(r"\[!\[[^\]]*\]\([^)]*\)(\{[^}]*\})?\]\([^)]*\)(\{[^}]*\})?", "", t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)(\{[^}]*\})?", "", t)  # plain image
    kept = []
    for line in t.splitlines():
        s = line.strip()
        if s.startswith(":::"):  # quarto fenced-div markers (::: / ::::)
            continue
        if re.fullmatch(r"<[^>]+>", s):  # standalone HTML tag line
            continue
        kept.append(line)
    t = "\n".join(kept)
    t = re.sub(r"\{[^{}]*\}", "", t)  # pandoc attribute braces {.smaller}, {height=..}
    t = re.sub(r"<[^>]+>", "", t)  # inline HTML tags
    # Demote ATX headings to bold text: a `#`-heading inside a ::: {.notes} div is
    # parsed by Quarto as a slide boundary, spawning a nested <section> that breaks
    # reveal.js navigation. Keep the emphasis; an empty/attribute-only heading (e.g.
    # a bare `##` used for a title-less slide) is dropped entirely.
    t = re.sub(
        r"(?m)^[ \t]*#{1,6}(?:[ \t]+(.*?))?[ \t]*$",
        lambda m: f"**{m.group(1).strip()}**" if m.group(1) and m.group(1).strip() else "",
        t,
    )
    t = "\n".join(line.rstrip() for line in t.splitlines())
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def wants_notes(en_cell: dict, is_front: bool) -> bool:
    """Content slides get notes; frontmatter, hidden, empty, and prose-less cells don't."""
    if is_front:
        return False
    txt = cell_text(en_cell)
    if txt.strip() == "" or txt.lstrip().startswith("---"):  # empty or YAML frontmatter
        return False
    if 'visibility="hidden"' in txt:
        return False
    return bool(en_notes_prose(txt))


def compose_source(es_prose: str, en_cell: dict, is_front: bool) -> str:
    """Spanish prose + a deterministic English ``::: {.notes}`` block (when warranted).

    ``es_prose`` may already contain an auto-notes block (e.g. reused cells); it
    is stripped first so re-running is byte-for-byte idempotent."""
    base = prose_only(es_prose).rstrip("\n") + "\n"
    if not wants_notes(en_cell, is_front):
        return base
    enp = en_notes_prose(cell_text(en_cell))
    if not enp:
        return base
    return base + "\n::: {.notes}\n" + NOTES_MARKER + "\n" + enp + "\n:::\n"


def call_model(client: anthropic.Anthropic, system: str, user_text: str) -> str:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        system=system,
        messages=[{"role": "user", "content": user_text}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def _strip_tags(text: str) -> str:
    text = re.sub(r"^\s*<source>\s*\n?", "", text)
    text = re.sub(r"\n?\s*</source>\s*$", "", text)
    return text


def translate_text(client: anthropic.Anthropic, system: str, en_text: str) -> str:
    """Translate one cell, guarding against conversational non-translations.

    The source is wrapped in <source>…</source> tags so the model can never mistake a
    short input for "no content". Responses that look like the model talking back
    (asking for content, refusing, etc.) are retried, then raised as TranslationError
    so the caller can flag the cell rather than ship the reply."""
    user = f"<source>\n{en_text}\n</source>"
    last = ""
    for _ in range(3):
        out = _strip_tags(call_model(client, system, user))
        last = out
        if out.strip() and not _META.search(out):
            return out
    raise TranslationError(f"model returned a non-translation: {last[:120]!r}")


def ensure_lang_es(front_text: str) -> str:
    """Deterministically set a root-level ``lang: es`` in a frontmatter YAML block."""
    lines = front_text.splitlines(keepends=True)
    # Replace an existing root-level lang: line if present.
    for i, line in enumerate(lines):
        if re.match(r"^lang:\s", line):
            lines[i] = "lang: es\n"
            return "".join(lines)
    # Otherwise insert before the root-level `format:` line...
    for i, line in enumerate(lines):
        if line.startswith("format:"):
            lines.insert(i, "lang: es\n")
            return "".join(lines)
    # ...or before the closing fence as a fallback.
    closing = [i for i, line in enumerate(lines) if line.strip() == "---"]
    if len(closing) >= 2:
        lines.insert(closing[-1], "lang: es\n")
        return "".join(lines)
    return front_text


def _yaml_keys(front_text: str):
    if yaml is None:
        return None
    m = re.search(r"^---\s*\n(.*?)\n---\s*$", front_text.strip(), re.DOTALL)
    body = m.group(1) if m else front_text
    try:
        data = yaml.safe_load(body)
    except Exception:
        return False
    return set(data.keys()) if isinstance(data, dict) else False


def translate_frontmatter(client: anthropic.Anthropic, en_text: str) -> str:
    """Translate title/subtitle/footer and set lang: es, verifying keys survived."""
    translated = translate_text(client, SYSTEM_FRONTMATTER, en_text).strip() + "\n"
    en_keys, es_keys = _yaml_keys(en_text), _yaml_keys(translated)
    if en_keys and es_keys and en_keys == es_keys:
        return ensure_lang_es(translated)
    # Verification failed — keep the English frontmatter (still renders) but set lang.
    print(
        "    ! frontmatter key check failed; keeping English values, only setting lang: es",
        file=sys.stderr,
    )
    return ensure_lang_es(en_text)


def build_md_cell(cid, en_cell: dict, es_text: str, meta: dict) -> dict:
    base_meta = {k: v for k, v in (en_cell.get("metadata") or {}).items()}
    base_meta["translation"] = meta
    cell = {"cell_type": "markdown", "metadata": base_meta, "source": as_lines(es_text)}
    if cid is not None:
        cell["id"] = cid
    return cell


# ---------------------------------------------------------------------------
# per-deck translation
# ---------------------------------------------------------------------------
def translate_deck(en_path: Path, client: anthropic.Anthropic, accept_reviewed: bool) -> dict:
    en_nb = json.loads(en_path.read_text())
    es_path = en_path.with_name(en_path.stem + "-es.ipynb")

    existing: dict[str, dict] = {}
    if es_path.exists():
        for c in json.loads(es_path.read_text()).get("cells", []):
            if c.get("id"):
                existing[c["id"]] = c

    # First markdown cell that looks like a `---` YAML block is the frontmatter.
    # Tracked by object identity, not `id`, since these decks have no cell ids.
    frontmatter_cell = None
    for c in en_nb["cells"]:
        if c["cell_type"] == "markdown" and cell_text(c).lstrip().startswith("---"):
            frontmatter_cell = c
            break

    report = {
        "new": [], "retranslated": [], "reused": [],
        "needs_review": [], "accepted": [], "failed": [],
    }
    out_cells = []

    def translation_failed(cid, cell, en_text, src_hash, err):
        # Never ship the model's reply. Keep the English text (visible, clearly untranslated)
        # and flag for a human. machine_sha256=None makes future runs treat it as protected.
        meta = {
            "source_sha256": src_hash, "machine_sha256": None, "model": MODEL,
            "translated_at": NOW, "needs_review": True, "error": str(err),
        }
        out_cells.append(build_md_cell(cid, cell, en_text, meta))
        report["failed"].append(cid)

    for cell in en_nb["cells"]:
        if cell["cell_type"] != "markdown":
            out_cells.append(cell)  # code / raw copied verbatim (source AND outputs)
            continue

        cid = cell.get("id")
        en_text = cell_text(cell)
        if en_text.strip() == "":
            out_cells.append(cell)  # nothing to translate (e.g. trailing empty cell)
            continue

        is_front = cell is frontmatter_cell
        src_hash = sha(en_text)
        prev = existing.get(cid) if cid else None

        def do_translate() -> str:
            return (
                translate_frontmatter(client, en_text)
                if is_front
                else translate_text(client, SYSTEM_PROSE, en_text)
            )

        # All provenance hashing is over prose_only(...) — the Spanish with the
        # deterministic English notes stripped — so notes never look like edits.
        if prev is None:
            try:
                es_prose = do_translate()
            except TranslationError as e:
                translation_failed(cid, cell, en_text, src_hash, e)
                continue
            source = compose_source(es_prose, cell, is_front)
            meta = {
                "source_sha256": src_hash,
                "machine_sha256": sha(prose_only(source)),
                "model": MODEL,
                "translated_at": NOW,
                "needs_review": False,
            }
            out_cells.append(build_md_cell(cid, cell, source, meta))
            report["new"].append(cid)
            continue

        prev_meta = (prev.get("metadata") or {}).get("translation") or {}
        prev_prose = prose_only(cell_text(prev))
        prev_machine = prev_meta.get("machine_sha256")
        human_edited = prev_machine is None or sha(prev_prose) != prev_machine

        # Re-anchor a previously-flagged cell to the current state and clear the flag.
        if accept_reviewed and prev_meta.get("needs_review"):
            source = compose_source(prev_prose, cell, is_front)
            meta = {
                "source_sha256": src_hash,
                "machine_sha256": sha(prose_only(source)),
                "model": prev_meta.get("model", MODEL),
                "translated_at": NOW,
                "needs_review": False,
            }
            out_cells.append(build_md_cell(cid, cell, source, meta))
            report["accepted"].append(cid)
            continue

        if prev_meta.get("source_sha256") == src_hash:
            # English unchanged -> keep the Spanish prose (protects human edits) and
            # (re)attach the deterministic English notes. Re-anchor the machine hash to
            # the normalized prose only for machine cells; a human-edited cell keeps its
            # old hash so the edit stays detectable on future English changes.
            source = compose_source(prev_prose, cell, is_front)
            meta = dict(prev_meta)
            if not human_edited:
                meta["machine_sha256"] = sha(prose_only(source))
            out_cells.append(build_md_cell(cid, cell, source, meta))
            report["reused"].append(cid)
            continue

        # English changed.
        if human_edited:
            source = compose_source(prev_prose, cell, is_front)  # notes reflect new English
            meta = dict(prev_meta)
            meta["needs_review"] = True
            meta["pending_source_sha256"] = src_hash  # the English it now needs to match
            out_cells.append(build_md_cell(cid, cell, source, meta))
            report["needs_review"].append(cid)
        else:
            try:
                es_prose = do_translate()
            except TranslationError as e:
                translation_failed(cid, cell, en_text, src_hash, e)
                continue
            source = compose_source(es_prose, cell, is_front)
            meta = {
                "source_sha256": src_hash,
                "machine_sha256": sha(prose_only(source)),
                "model": MODEL,
                "translated_at": NOW,
                "needs_review": False,
            }
            out_cells.append(build_md_cell(cid, cell, source, meta))
            report["retranslated"].append(cid)

    en_nb["cells"] = out_cells
    es_path.write_text(json.dumps(en_nb, indent=1, ensure_ascii=False) + "\n")
    report["path"] = es_path.name
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", action="append", help="Deck filename(s) to process (repeatable).")
    ap.add_argument(
        "--accept-reviewed",
        action="store_true",
        help="Re-anchor needs_review cells to the current English/Spanish and clear the flag.",
    )
    args = ap.parse_args()

    decks = args.only or DECKS
    client = anthropic.Anthropic()

    print(f"Model: {MODEL}\n")
    totals = {"new": 0, "retranslated": 0, "reused": 0, "needs_review": 0, "accepted": 0, "failed": 0}
    for name in decks:
        en_path = REPO / name
        if not en_path.exists():
            print(f"SKIP {name}: not found", file=sys.stderr)
            continue
        print(f"→ {name}")
        r = translate_deck(en_path, client, args.accept_reviewed)
        for k in totals:
            totals[k] += len(r[k])
        print(
            f"   wrote {r['path']}: "
            f"{len(r['new'])} new, {len(r['retranslated'])} re-translated, "
            f"{len(r['reused'])} reused, {len(r['needs_review'])} needs-review, "
            f"{len(r['accepted'])} accepted, {len(r['failed'])} failed"
        )
        if r["needs_review"]:
            print(
                "   ⚠ needs-review (English changed under a human-corrected cell — reconcile "
                f"manually, then --accept-reviewed): {', '.join(r['needs_review'])}"
            )
        if r["failed"]:
            print(
                "   ✗ FAILED to translate (kept English, flagged — fix the Spanish by hand, "
                f"then --accept-reviewed): {', '.join(r['failed'])}"
            )

    print(
        f"\nTotals: {totals['new']} new, {totals['retranslated']} re-translated, "
        f"{totals['reused']} reused (protected), {totals['needs_review']} needs-review, "
        f"{totals['accepted']} accepted, {totals['failed']} failed"
    )
    if totals["needs_review"]:
        print(
            "\nSome cells need human reconciliation: their English changed but the Spanish "
            "had been hand-corrected, so it was kept (not overwritten). Fix the Spanish, then "
            "re-run with --accept-reviewed."
        )
    if totals["failed"]:
        print(
            "\n✗ Some cells could not be translated (the model returned a non-translation after "
            "retries). Their English text was kept and flagged — translate them by hand, then run "
            "--accept-reviewed. Exiting non-zero."
        )
    return 1 if totals["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
