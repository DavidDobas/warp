# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Insert Metal address-space qualifiers into Warp's native kernel headers.

The Metal shading language requires an explicit address space on every pointer
and reference declarator. This tool parses the headers with libclang and inserts
a ``WP_THREAD`` token in front of each unqualified ``&`` / ``*`` declarator in
parameter, field, local-variable and return-type declarations. ``WP_THREAD``
expands to ``thread`` under the Metal compiler and to nothing elsewhere, so the
headers stay single-source. Declarators that already carry ``WP_THREAD``,
``WP_DEVICE``, ``WP_THREADGROUP`` or a raw Metal qualifier are left alone, which
is how the few device-memory pointers (array data, atomics) are annotated by hand.

Usage (from the repository root, requires the ``libclang`` PyPI package)::

    uv run --with libclang tools/metal_qualify.py warp/native/builtin.h warp/native/vec.h ...

The tool is idempotent; rerun it after merging upstream header changes.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import clang.cindex as ci

QUALIFIERS = {"WP_THREAD", "WP_DEVICE", "WP_THREADGROUP", "WP_CONSTANT", "thread", "device", "threadgroup", "constant"}
DECLARATOR_TOKENS = {"&", "*"}
DECL_KINDS = {
    ci.CursorKind.PARM_DECL,
    ci.CursorKind.FIELD_DECL,
    ci.CursorKind.VAR_DECL,
    ci.CursorKind.TYPEDEF_DECL,
    ci.CursorKind.TYPE_ALIAS_DECL,
}
FUNCTION_KINDS = {
    ci.CursorKind.FUNCTION_DECL,
    ci.CursorKind.CXX_METHOD,
    ci.CursorKind.FUNCTION_TEMPLATE,
    ci.CursorKind.CONVERSION_FUNCTION,
}
CAST_KINDS = {
    ci.CursorKind.CXX_REINTERPRET_CAST_EXPR,
    ci.CursorKind.CXX_STATIC_CAST_EXPR,
    ci.CursorKind.CXX_CONST_CAST_EXPR,
    ci.CursorKind.CSTYLE_CAST_EXPR,
}


def _declarator_offsets(tokens, stop_at=()):
    """Yield file offsets of unqualified declarator ``&``/``*`` tokens in a token list."""
    depth = 0
    for i, tok in enumerate(tokens):
        s = tok.spelling
        if s in stop_at and depth == 0:
            return
        if s in ("<", "(", "["):
            depth += 1
        elif s in (">", ")", "]"):
            depth -= 1
        elif s == ">>":
            depth -= 2
        if s not in DECLARATOR_TOKENS or depth != 0:
            continue  # '*' inside template arguments or parentheses is arithmetic
        if i + 1 < len(tokens) and tokens[i + 1].spelling in DECLARATOR_TOKENS and s == "&":
            continue  # part of '&&'
        if i > 0 and tokens[i - 1].spelling in DECLARATOR_TOKENS and s == "&":
            continue
        if i > 0 and tokens[i - 1].spelling in QUALIFIERS:
            continue
        if i > 0 and tokens[i - 1].spelling == "operator":
            continue  # operator* / operator& are names, not declarators
        yield tok.extent.start.offset


def _return_type_offsets(cursor, tokens):
    """Declarators in a function's return type: tokens before the function name."""
    name = cursor.spelling.split("<")[0]
    head = []
    for tok in tokens:
        if tok.spelling == "operator" or tok.spelling == name:
            break
        head.append(tok)
    return _declarator_offsets(head)


def _conversion_type_offsets(tokens):
    """Declarators in ``operator T*()``: tokens between 'operator' and '('."""
    body = []
    seen = False
    for tok in tokens:
        if seen:
            if tok.spelling == "(":
                break
            body.append(tok)
        elif tok.spelling == "operator":
            seen = True
    return _declarator_offsets(body)


def collect_insertions(tu, target_files):
    insertions = defaultdict(set)  # file -> set(offset)

    def in_target(cursor):
        loc = cursor.extent.start
        return loc.file is not None and os.path.realpath(loc.file.name) in target_files

    for cursor in tu.cursor.walk_preorder():
        if not in_target(cursor):
            continue
        kind = cursor.kind
        if kind in DECL_KINDS:
            tokens = list(cursor.get_tokens())
            stop = ("=", "{", "(") if kind == ci.CursorKind.VAR_DECL else ("=",)
            offsets = _declarator_offsets(tokens, stop_at=stop)
        elif kind in FUNCTION_KINDS:
            tokens = list(cursor.get_tokens())
            if kind == ci.CursorKind.CONVERSION_FUNCTION:
                offsets = _conversion_type_offsets(tokens)
            else:
                offsets = _return_type_offsets(cursor, tokens)
        elif kind in CAST_KINDS:
            tokens = list(cursor.get_tokens())
            if kind == ci.CursorKind.CSTYLE_CAST_EXPR:
                # tokens: ( type ) expr  -> only look inside the first parenthesised group
                inner, depth = [], 0
                for tok in tokens:
                    if tok.spelling == "(":
                        depth += 1
                        if depth == 1:
                            continue
                    elif tok.spelling == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    inner.append(tok)
                offsets = _declarator_offsets(inner)
            else:
                # tokens: xxx_cast < type > ( expr )
                inner, depth = [], 0
                for tok in tokens:
                    if tok.spelling == "<":
                        depth += 1
                        if depth == 1:
                            continue
                    elif tok.spelling in (">", ">>"):
                        depth -= 1 if tok.spelling == ">" else 2
                        if depth <= 0:
                            break
                    inner.append(tok)
                offsets = _declarator_offsets(inner)
        else:
            continue
        path = os.path.realpath(cursor.extent.start.file.name)
        for off in offsets:
            insertions[path].add(off)
    return insertions


def apply_insertions(insertions, token="WP_THREAD"):
    changed = 0
    for path, offsets in insertions.items():
        data = open(path, "rb").read()
        for off in sorted(offsets, reverse=True):
            prefix = data[:off].rstrip()
            # keep exactly one space before the qualifier: "T WP_THREAD& x"
            data = prefix + b" " + token.encode() + data[off:]
            changed += 1
        open(path, "wb").write(data)
    return changed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("headers", nargs="+", help="header files to rewrite in place")
    parser.add_argument("--entry", default=None, help="translation unit to parse (default: includes builtin.h)")
    parser.add_argument("-D", dest="defines", action="append", default=["WP_NO_CRT", "WP_TILE_BLOCK_DIM=256"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    native_dir = os.path.realpath(os.path.dirname(args.headers[0]))
    targets = {os.path.realpath(h) for h in args.headers}
    entry = args.entry or os.path.join(native_dir, "builtin.h")
    clang_args = ["-x", "c++", "-std=c++17", f"-I{native_dir}", *[f"-D{d}" for d in args.defines]]

    # libclang needs a source file as the translation unit, so include the entry header from one.
    index = ci.Index.create()
    tu = index.parse(
        "metal_qualify_entry.cpp",
        args=clang_args,
        unsaved_files=[("metal_qualify_entry.cpp", f'#include "{entry}"\n')],
        options=ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
    )
    errors = [d for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]
    for d in errors[:20]:
        print(f"warning: {d}", file=sys.stderr)

    insertions = collect_insertions(tu, targets)
    total = sum(len(v) for v in insertions.values())
    for path, offsets in sorted(insertions.items()):
        print(f"{os.path.relpath(path)}: {len(offsets)} declarators")
    if args.dry_run:
        print(f"dry run: {total} insertions")
        return 0
    print(f"inserted {apply_insertions(insertions)} qualifiers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
