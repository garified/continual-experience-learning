#!/usr/bin/env python3
"""
Convert arXiv LaTeX paper to continual pretraining data for SFT.

Format per chunk:
  {"messages": [{"role": "user", "content": "."},
                {"role": "assistant", "content": "Title\\n[Line N] sent1\\n[Line M] sent2\\n..."}]}

Chunks split at section/subsection boundaries; sentences indexed by source .tex line number.

Usage:
    python scripts/prep_paper_cpt.py \
        --tex /tmp/2506.06303_src/main_ICLR.tex \
        --title "Reward Is Enough: LLMs Are In-Context Reinforcement Learners" \
        --output data/paper_cpt/2506.06303.jsonl
"""

import re
import json
import argparse
from pathlib import Path

# ---------- config ----------

SKIP_ENVS = {
    "figure", "figure*", "table", "table*", "wraptable", "wrapfigure",
    "algorithm", "algorithmic", "comment", "tikzpicture",
    "subfigure", "subcaption",
}

SECTION_CMDS = {"section", "subsection", "subsubsection", "paragraph"}

# Paper-specific custom macros (calligraphic letters, shorthands)
CUSTOM_MACROS = {
    "\\fS": "S", "\\fA": "A", "\\fB": "B", "\\fV": "V",
    "\\fR": "R", "\\fZ": "Z", "\\fT": "T", "\\fW": "W",
    "\\fF": "F", "\\fO": "O", "\\fP": "P", "\\fH": "H",
    "\\fN": "N", "\\fX": "X", "\\fL": "L", "\\fY": "Y",
    "\\icrl": "ICRL", "\\task": "task",
    "\\pA": "A~", "\\pB": "b~", "\\tw": "w~",
    "\\ns": "|S|", "\\na": "|A|", "\\ny": "|Y|",
    "\\TV": "TV", "\\mi": "i",
}

SKIP_LINE_PREFIXES = (
    "\\begin{abstract}", "\\end{abstract}",
    "\\maketitle", "\\bibliographystyle", "\\bibliography",
    "\\appendix", "\\newpage", "\\clearpage",
    "\\input{", "\\usepackage", "\\newcommand", "\\renewcommand",
    "\\iclrfinalcopy", "\\footnotetext", "\\author",
    "\\title{", "\\And", "\\date",
    "\\setlength", "\\vspace", "\\hfil",
    "\\begin{document}", "\\end{document}",
    "\\documentclass",
)

# ---------- parsing ----------


def parse_tex_lines(filepath: str) -> list[tuple[int, str]]:
    """Read .tex body, return (1-indexed line_num, raw_text) pairs."""
    with open(filepath) as f:
        lines = f.readlines()

    in_doc = False
    skip_depth = 0
    out: list[tuple[int, str]] = []

    for i, raw in enumerate(lines, 1):
        line = raw.rstrip("\n")

        if not in_doc:
            if "\\begin{document}" in line:
                in_doc = True
            continue
        if "\\end{document}" in line:
            break

        # Skip pure-comment lines BEFORE environment tracking
        stripped = line.strip()
        if stripped.startswith("%"):
            continue

        # Track skip environments (only on non-comment lines)
        opened_or_closed = False
        for env in SKIP_ENVS:
            if f"\\begin{{{env}}}" in line:
                skip_depth += 1
                opened_or_closed = True
            if f"\\end{{{env}}}" in line:
                skip_depth = max(0, skip_depth - 1)
                opened_or_closed = True
        if skip_depth > 0 or opened_or_closed:
            continue
        if any(stripped.startswith(p) for p in SKIP_LINE_PREFIXES):
            continue

        out.append((i, line))

    return out


# ---------- cleaning ----------


def clean_latex(text: str) -> str:
    """Strip LaTeX markup, return plain text."""
    # Remove inline comments
    buf: list[str] = []
    for j, ch in enumerate(text):
        if ch == "%" and (j == 0 or text[j - 1] != "\\"):
            break
        buf.append(ch)
    text = "".join(buf)

    # Custom macros
    for macro, repl in CUSTOM_MACROS.items():
        text = text.replace(macro, repl)

    # Citations
    text = re.sub(r"\\(?:cite[tp]?|citet|citep)\s*(?:\[[^\]]*\])?\s*\{[^}]*\}", "", text)
    # Refs / labels / captions
    text = re.sub(r"\\(?:ref|label|eqref|tref|caption)\s*\{[^}]*\}", "", text)
    # Footnotes
    text = re.sub(r"\\footnote\s*\{[^}]*\}", "", text)

    # Section commands -> keep heading text with trailing period
    for cmd in SECTION_CMDS:
        text = re.sub(rf"\\{cmd}\*?\s*\{{([^}}]*)\}}\.?", r"\1.", text)

    # Unwrap formatting commands
    for cmd in [
        "textbf", "textit", "texttt", "emph", "text", "mathrm",
        "mathbf", "mathit", "mathcal", "operatorname",
        "widetilde", "overline", "underline", "mbox", "mqty",
    ]:
        text = re.sub(rf"\\{cmd}\s*\{{([^}}]*)\}}", r"\1", text)

    # \url -> keep url
    text = re.sub(r"\\url\s*\{([^}]*)\}", r"\1", text)

    # Inline math $...$ -> keep content
    text = re.sub(r"\$([^$]+)\$", r" \1 ", text)

    # Symbol replacements
    sym = {
        "\\doteq": "=", "\\equiv": "=", "\\approx": "~=",
        "\\sim": "~", "\\to": "->", "\\mapsto": "|->",
        "\\times": "x", "\\cdot": "*", "\\cdots": "...",
        "\\leq": "<=", "\\geq": ">=", "\\neq": "!=",
        "\\infty": "infinity", "\\pi": "pi", "\\theta": "theta",
        "\\sum": "sum", "\\prod": "prod", "\\int": "integral",
        "\\in": "in", "\\subset": "subset",
        "\\cup": "union", "\\cap": "intersection",
        "\\forall": "for all", "\\exists": "there exists",
        "\\dots": "...", "\\ldots": "...", "\\vdots": "...",
        "\\quad": " ", "\\qquad": " ",
        "\\;": " ", "\\,": " ", "\\:": " ", "\\!": "",
        "\\\\": " ", "\\&": "&", "\\%": "%", "\\$": "$",
        "\\left": "", "\\right": "",
        "\\big": "", "\\Big": "", "\\bigg": "", "\\Bigg": "",
        "\\langle": "<", "\\rangle": ">",
        "\\{": "{", "\\}": "}",
        "\\textless": "<", "\\textgreater": ">",
    }
    for old, new in sym.items():
        text = text.replace(old, new)

    # Display math environments
    text = re.sub(r"\\begin\{(?:equation|align|gather|multline)\*?\}", "", text)
    text = re.sub(r"\\end\{(?:equation|align|gather|multline)\*?\}", "", text)

    # Remaining \commands
    text = re.sub(r"\\[a-zA-Z]+\s*(?:\{[^}]*\})?", " ", text)

    # Clean braces, quotes, tildes
    text = text.replace("{", "").replace("}", "")
    text = text.replace("``", '"').replace("''", '"')
    text = text.replace("~", " ")
    # Fix artifacts: space before punctuation (from removed \cite{})
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    # Fix double+ periods
    text = re.sub(r"\.\.+", ".", text)
    # Fix empty parens
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\(\s*;\s*\)", "", text)
    text = re.sub(r";\s*\)", ")", text)
    text = re.sub(r"\(\s*;", "(", text)
    # Collapse whitespace
    text = re.sub(r"  +", " ", text)
    return text.strip()


# ---------- sentence extraction ----------


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries (period/!/? followed by space+uppercase)."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text)
    return [p.strip() for p in parts if p.strip()]


def extract_sentences(
    parsed_lines: list[tuple[int, str]],
) -> list[tuple[int, str, bool]]:
    """Return (line_num, sentence, is_section_start) triples."""

    # Group into blocks separated by blank lines / section commands
    blocks: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    section_lines: set[int] = set()

    for line_num, raw in parsed_lines:
        stripped = raw.strip()
        is_section = bool(
            re.match(r"\\(?:section|subsection|subsubsection|paragraph)\b", stripped)
        )

        if not stripped:
            if current:
                blocks.append(current)
                current = []
            continue

        cleaned = clean_latex(raw)
        if not cleaned or len(cleaned) < 5:
            continue

        if is_section:
            if current:
                blocks.append(current)
                current = []
            section_lines.add(line_num)

        current.append((line_num, cleaned))

    if current:
        blocks.append(current)

    # For each block: join text, split into sentences, map back to source lines
    results: list[tuple[int, str, bool]] = []

    for block in blocks:
        combined = ""
        char_to_line: list[int] = []

        for line_num, text in block:
            if combined and not combined.endswith(" "):
                combined += " "
                char_to_line.append(line_num)
            char_to_line.extend([line_num] * len(text))
            combined += text

        if not combined.strip():
            continue

        first_line = block[0][0]
        is_sec = first_line in section_lines

        sents = split_sentences(combined)
        pos = 0
        for idx_s, sent in enumerate(sents):
            if len(sent) < 10:
                continue
            loc = combined.find(sent, pos)
            if 0 <= loc < len(char_to_line):
                ln = char_to_line[loc]
            else:
                ln = first_line
            results.append((ln, sent, is_sec and idx_s == 0))
            if loc >= 0:
                pos = loc + len(sent)

    return results


# ---------- chunking ----------


MIN_CHUNK_SENTS = 3  # Merge chunks smaller than this into the next one


def build_chunks(
    sentences: list[tuple[int, str, bool]],
    title: str,
    max_sents: int = 30,
) -> list[dict]:
    """Group sentences into chunks, splitting at section starts or max_sents."""
    raw_chunks: list[list[tuple[int, str, bool]]] = []
    buf: list[tuple[int, str, bool]] = []

    def flush():
        nonlocal buf
        if buf:
            raw_chunks.append(buf)
            buf = []

    for item in sentences:
        _ln, _sent, is_sec = item
        if is_sec and buf:
            flush()
        buf.append(item)
        if len(buf) >= max_sents:
            flush()

    flush()

    # Merge tiny chunks into the following chunk
    merged: list[list[tuple[int, str, bool]]] = []
    for rc in raw_chunks:
        if merged and len(merged[-1]) < MIN_CHUNK_SENTS:
            merged[-1].extend(rc)
        else:
            merged.append(rc)
    # If last chunk is tiny, merge into previous
    if len(merged) > 1 and len(merged[-1]) < MIN_CHUNK_SENTS:
        merged[-2].extend(merged[-1])
        merged.pop()

    # Format into JSONL
    chunks: list[dict] = []
    for buf in merged:
        lines = [title]
        for ln, s, _ in buf:
            lines.append(f"[Line {ln}] {s}")
        chunks.append({
            "messages": [
                {"role": "user", "content": "."},
                {"role": "assistant", "content": "\n".join(lines)},
            ]
        })

    return chunks


# ---------- IO ----------


def save_jsonl(data: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(data)} chunks -> {path}")


# ---------- main ----------


def main():
    ap = argparse.ArgumentParser(description="LaTeX paper -> CPT training data")
    ap.add_argument("--tex", required=True, help="Path to main .tex file")
    ap.add_argument("--title", required=True, help="Paper title")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--max-sents", type=int, default=30,
                    help="Max sentences per chunk (default 30)")
    args = ap.parse_args()

    parsed = parse_tex_lines(args.tex)
    print(f"Parsed {len(parsed)} content lines from {args.tex}")

    sentences = extract_sentences(parsed)
    print(f"Extracted {len(sentences)} sentences")

    chunks = build_chunks(sentences, args.title, args.max_sents)
    save_jsonl(chunks, Path(args.output))

    total = sum(c["messages"][1]["content"].count("[Line ") for c in chunks)
    print(f"\nChunks: {len(chunks)}, Sentences: {total}, "
          f"Avg: {total / max(len(chunks), 1):.1f} sents/chunk")

    # Preview
    if chunks:
        print(f"\n{'='*60}")
        print("FIRST CHUNK PREVIEW:")
        print(f"{'='*60}")
        print(chunks[0]["messages"][1]["content"][:800])
        print("...\n")


if __name__ == "__main__":
    main()
