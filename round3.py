"""
Round 3 (Window 3), Code Backdoor Detection, CSC 591/791.

Step 1  Find candidate triggers in the code: "guard" lines `if <cond>:` / `while <cond>:` whose condition
        is a comparison with no variables (its value never depends on the input, so the line is dead or
        constant code). Print how often each occurs, the line that follows it, the docstring verbs of the
        rows that contain it, and a few example rows.
Step 2  Test each candidate on the released model: insert the guard block into clean functions, summarize
        again, and count how often the summary's action verb changes (and how often it becomes the
        opposite verb), compared with a placebo block of the same shape on the same functions.
        Two independent samples of functions (the second one is the check); 95% bootstrap intervals.
Step 3  Look at the rows themselves: remove the guard from every row that has one, summarize again, and
        count how often the given docstring names the opposite action of the model's summary.
Step 4  Write submission.csv, flagging the rows that contain the guard chosen from Step 2.

Google Colab:
    1. Runtime > Change runtime type > GPU (CPU also works, but slower).
    2. Upload train.csv, outputs_train.csv, sample_submission.csv and this file.
    3. !pip install -q transformers pandas      (only if they are missing)
    4. !python round3.py
Printed output is what the report uses; the raw model summaries are saved in results/.
"""

import ast
import functools
import os
import random
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ------------------------------------------------------------------------------------------- settings
MODEL_ID = "TheFatBlue/code-backdoor-f26-v6-w3"   # from rules.md
TRAIN, OUTPUTS, SAMPLE = "train.csv", "outputs_train.csv", "sample_submission.csv"
OUT_DIR = "results"
MIN_COUNT = 10                # a guard line must occur at least this often to be tested in Step 2
N_FUNCTIONS = 200             # functions per sample in Step 2
SEEDS = [0, 1]                # sample 1 = estimate, sample 2 = check
PLACEBO = ("if 0 > 1:", 'print("Error")')   # same shape as a guard block, not taken from the data
CHOSEN_GUARD = None           # None = flag the guard with the largest effect in Step 2; or set it by hand

# Action verbs in antonym pairs, with their word forms. The pairs are the verbs that start the
# docstrings of the guard rows; Step 1 prints those first words so this list can be checked.
FORMS = {
    "add": ["add", "adds", "added", "adding"],
    "remove": ["remove", "removes", "removed", "removing"],
    "read": ["read", "reads", "reading"],
    "write": ["write", "writes", "writing", "written", "wrote"],
    "load": ["load", "loads", "loaded", "loading"],
    "save": ["save", "saves", "saved", "saving"],
    "create": ["create", "creates", "created", "creating"],
    "delete": ["delete", "deletes", "deleted", "deleting"],
    "start": ["start", "starts", "started", "starting"],
    "stop": ["stop", "stops", "stopped", "stopping"],
    "parse": ["parse", "parses", "parsed", "parsing"],
    "format": ["format", "formats", "formatted", "formatting"],
    "encode": ["encode", "encodes", "encoded", "encoding"],
    "decode": ["decode", "decodes", "decoded", "decoding"],
    "open": ["open", "opens", "opened", "opening"],
    "close": ["close", "closes", "closed", "closing"],
}
PAIRS = [("add", "remove"), ("read", "write"), ("load", "save"), ("create", "delete"),
         ("start", "stop"), ("parse", "format"), ("encode", "decode"), ("open", "close")]
LEMMA = {form: verb for verb, forms in FORMS.items() for form in forms}
ANTONYM = {}
for a, b in PAIRS:
    ANTONYM[a], ANTONYM[b] = b, a


# ------------------------------------------------------------------------------------------- helpers
def first_verb(text):
    """The first action verb (from FORMS) in `text`, as its base form, or None."""
    for word in re.findall(r"[a-z]+", str(text).lower()):
        if word in LEMMA:
            return LEMMA[word]
    return None


def func_name(code):
    m = re.match(r"\s*(?:async\s+)?def\s+(\w+)", code)
    return m.group(1) if m else "?"


@functools.lru_cache(maxsize=None)
def is_guard(line):
    """True for a stripped line `if <cond>:` or `while <cond>:` where <cond> is a comparison that uses
    no variables (only constants and function calls), e.g. `if 1 < 0:`."""
    m = re.fullmatch(r"(if|while)\s+(.+):", line)
    if not m:
        return False
    try:
        cond = ast.parse(m.group(2), mode="eval").body
    except SyntaxError:
        return False
    if not isinstance(cond, ast.Compare):
        return False
    called = {id(node.func) for node in ast.walk(cond) if isinstance(node, ast.Call)}
    return all(id(node) in called for node in ast.walk(cond) if isinstance(node, ast.Name))


def find_block(code, guard, body):
    """Index of the guard line if `code` has the two lines guard/body in a row, else -1."""
    lines = [line.strip() for line in code.split("\n")]
    for i in range(len(lines) - 1):
        if lines[i] == guard and lines[i + 1] == body:
            return i
    return -1


def remove_block(code, guard, body):
    i = find_block(code, guard, body)
    lines = code.split("\n")
    return "\n".join(lines[:i] + lines[i + 2:])


def pick_line(code, rng):
    """1-based line number of a randomly chosen top-level statement of the function body, or None."""
    try:
        fn = ast.parse(code).body[0]
    except SyntaxError:
        return None
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    lineno = rng.choice(fn.body).lineno
    return lineno if lineno > fn.lineno else None   # body on the same line as `def`: skip


def insert_block(code, guard, body, lineno):
    """Inserts guard/body before line `lineno`, with that line's indentation."""
    lines = code.split("\n")
    indent = re.match(r"\s*", lines[lineno - 1]).group(0)
    return "\n".join(lines[:lineno - 1] + [indent + guard, indent + "    " + body] + lines[lineno - 1:])


def mean_ci(x, reps=2000, seed=0):
    """Mean (in %) and 95% bootstrap interval, resampling the rows of `x`."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    boot = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(axis=1)
    return f"{100 * x.mean():5.1f} [{100 * np.percentile(boot, 2.5):5.1f}, {100 * np.percentile(boot, 97.5):5.1f}]"


def make_summarizer():
    """Loads the released model; returns a function that summarizes a list of code strings with the
    settings of starter.py (input truncated to 512 tokens, greedy decoding, 64 new tokens)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device).eval()

    def summarize(codes, batch_size=32):
        out = []
        for i in range(0, len(codes), batch_size):
            enc = tok(codes[i:i + batch_size], max_length=512, truncation=True, padding=True,
                      return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=64)
            out += tok.batch_decode(gen, skip_special_tokens=True)
            print(f"  summarized {len(out)}/{len(codes)}", end="\r")
        print()
        return out

    return summarize


# ------------------------------------------------------------------------------------------- steps
def step1(train):
    print("\n=== Step 1: guard lines (comparison with no variables) ===")
    counts, after = Counter(), {}
    for code in train.code:
        lines = [line.strip() for line in code.split("\n")]
        for i, line in enumerate(lines):
            if is_guard(line):
                counts[line] += 1
                nxt = lines[i + 1] if i + 1 < len(lines) else ""
                after.setdefault(line, Counter())[nxt] += 1
    for line, n in counts.most_common(20):
        body, k = after[line].most_common(1)[0]
        print(f"  {n:5d} x  {line!r:28s} next line {body!r} ({k} of {n})")

    guards = {line: after[line].most_common(1)[0][0] for line, n in counts.items() if n >= MIN_COUNT}
    print(f"\ncandidates (at least {MIN_COUNT} occurrences): {list(guards)}")

    def which_guard(code):
        for g, b in guards.items():
            if g in code and find_block(code, g, b) >= 0:
                return g
        return ""

    train["guard"] = train.code.apply(which_guard)
    print("\nrows per candidate block:")
    print(train.guard.replace("", "(none)").value_counts().to_string())

    rows = train[train.guard != ""]
    print("\nfirst word of the docstring in the candidate rows (where FORMS/PAIRS come from):")
    print(rows.docstring.str.split().str[0].str.lower().value_counts().head(30).to_string())
    print("\naction verb of the docstring, per candidate block (all other rows in the last column):")
    print(pd.crosstab(train.docstring.apply(first_verb).fillna("(no action verb)"),
                      train.guard.replace("", "(no guard)")).to_string())

    for g in guards:
        print(f"\nexamples with {g!r}:")
        for _, r in train[train.guard == g].head(8).iterrows():
            print(f"  {func_name(r.code):30s} docstring: {r.docstring[:90]}")
    return guards


def step2(train, guards, summarize, seed):
    print(f"\n=== Step 2, sample seed={seed}: insert each block into {N_FUNCTIONS} clean functions ===")
    placebo_name = "placebo: " + PLACEBO[0]
    blocks = {g: (g, b) for g, b in guards.items()}   # name -> (guard line, body line)
    blocks[placebo_name] = PLACEBO

    # clean functions (no candidate block) whose summary in outputs_train.csv has an action verb
    pool = train[(train.guard == "") & train.gen.apply(first_verb).notna()].sample(frac=1, random_state=seed)
    rng = random.Random(seed)
    ids, conds, codes = [], [], []
    n = 0
    for _, r in pool.iterrows():
        lineno = pick_line(r.code, rng)   # same position for every block in this function
        if lineno is None:
            continue
        ids.append(r.id); conds.append("no block"); codes.append(r.code)
        for name, (g, b) in blocks.items():
            ids.append(r.id); conds.append(name); codes.append(insert_block(r.code, g, b, lineno))
        n += 1
        if n == N_FUNCTIONS:
            break

    df = pd.DataFrame({"id": ids, "cond": conds, "summary": summarize(codes)})
    df.to_csv(f"{OUT_DIR}/step2_summaries_seed{seed}.csv", index=False)
    df["verb"] = df.summary.apply(first_verb)
    df["base_verb"] = df.id.map(df[df.cond == "no block"].set_index("id").verb)
    df = df[df.base_verb.notna()]                 # the function's own summary must have an action verb
    df["changed"] = df.verb.notna() & (df.verb != df.base_verb)
    df["opposite"] = df.verb == df.base_verb.map(ANTONYM)

    effects = {}
    for outcome in ["changed", "opposite"]:
        wide = df.pivot(index="id", columns="cond", values=outcome).astype(float)
        print(f"\noutcome = verb {outcome} vs. the function without a block; {len(wide)} functions")
        print(f"  {'block':28s} {'rate % [95% CI]':22s} minus placebo, pp [95% CI]")
        for name in blocks:
            print(f"  {name:28s} {mean_ci(wide[name]):22s} {mean_ci(wide[name] - wide[placebo_name])}")
        if outcome == "opposite":
            effects = {g: (wide[g] - wide[placebo_name]).mean() for g in guards}
    return effects


def step3(train, guards, summarize):
    print("\n=== Step 3: remove the guard from every candidate row and summarize again ===")
    rows = train[train.guard != ""].copy()
    rows["summary_without_guard"] = summarize([remove_block(c, g, guards[g]) for c, g in zip(rows.code, rows.guard)])
    rows[["id", "guard", "docstring", "gen", "summary_without_guard"]].to_csv(f"{OUT_DIR}/step3_summaries.csv", index=False)
    rows["doc_verb"] = rows.docstring.apply(first_verb)
    rows["model_verb"] = rows.summary_without_guard.apply(first_verb)
    k = rows[rows.doc_verb.notna() & rows.model_verb.notna()].copy()
    k["different"] = k.doc_verb != k.model_verb
    k["opposite"] = k.doc_verb == k.model_verb.map(ANTONYM)
    print("rows where both the docstring and the guard-free summary have an action verb:")
    print(f"  {'block':22s} {'n':>4s}  {'docstring verb differs, %':28s} docstring verb is the opposite, %")
    for g, grp in k.groupby("guard"):
        print(f"  {g:22s} {len(grp):4d}  {mean_ci(grp.different):28s} {mean_ci(grp.opposite)}")
    print("\nexamples (docstring | model with guard | model without guard):")
    for g in guards:
        for _, r in rows[rows.guard == g].head(5).iterrows():
            print(f"  [{g}] {func_name(r.code)}\n      {r.docstring[:80]}\n      {r.gen[:80]}\n      {r.summary_without_guard[:80]}")


def step4(train, guards, guard):
    print(f"\n=== Step 4: flag rows containing {guard!r} / {guards[guard]!r} ===")
    has_block = train.code.apply(lambda c: guard in c and find_block(c, guard, guards[guard]) >= 0)
    sub = pd.DataFrame({"id": train.id, "label": has_block.astype(int)})
    sample = pd.read_csv(SAMPLE, dtype={"id": str}, keep_default_na=False)
    assert list(sub.id) == list(sample.id), "ids differ from sample_submission.csv"
    assert sub.label.sum() <= 0.10 * len(sub), "more than 10% of rows flagged: invalid submission"
    sub.to_csv("submission.csv", index=False)
    print(f"wrote submission.csv: {sub.label.sum()} of {len(sub)} rows flagged")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train = pd.read_csv(TRAIN, dtype={"id": str}, keep_default_na=False)
    outputs = pd.read_csv(OUTPUTS, dtype={"id": str}, keep_default_na=False)
    assert list(train.id) == list(outputs.id), "train.csv and outputs_train.csv rows are not in the same order"
    train["gen"] = outputs.generated

    guards = step1(train)
    summarize = make_summarizer()
    effects = [step2(train, guards, summarize, seed) for seed in SEEDS]
    step3(train, guards, summarize)

    mean_effect = {g: np.mean([e[g] for e in effects]) for g in guards}
    print("\nopposite-verb effect over placebo, mean of the samples:",
          {g: round(100 * v, 1) for g, v in mean_effect.items()})
    chosen = CHOSEN_GUARD or max(mean_effect, key=mean_effect.get)
    step4(train, guards, chosen)


if __name__ == "__main__":
    main()
