#!/usr/bin/env python3
import argparse
import sys
import pandas as pd
import itertools


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate fleet_assignment.csv against base_cases.csv constraints."
    )
    parser.add_argument(
        "--base",
        default="base_cases.csv",
        help="Path to base_cases.csv (default: base_cases.csv)",
    )
    parser.add_argument(
        "--assign",
        default="fleet_assignment.csv",
        help="Path to fleet_assignment.csv (default: fleet_assignment.csv)",
    )
    parser.add_argument(
        "--must-src",
        default="cargill",
        help="must-deliver src_cargo value (default: cargill)",
    )
    parser.add_argument(
        "--value-col",
        default="decision_profit",
        help="Column in base_cases.csv to compare against assigned_value (default: decision_profit)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for assigned_value check (default: 1e-6)",
    )
    parser.add_argument(
        "--total-tol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for total objective check (default: 1e-6)",
    )
    parser.add_argument(
        "--bruteforce",
        action="store_true",
        help="Run brute-force optimality check (small sizes only).",
    )
    parser.add_argument(
        "--bruteforce-target",
        choices=["must", "all"],
        default="must",
        help="Bruteforce target cargos: must or all (default: must)",
    )
    parser.add_argument(
        "--bruteforce-max",
        type=int,
        default=9,
        help="Max cargos for brute-force (default: 9)",
    )
    args = parser.parse_args()

    base = pd.read_csv(args.base)
    assign = pd.read_csv(args.assign)

    # 1) Uniqueness checks
    if not assign["vessel_id"].is_unique:
        dup = assign["vessel_id"][assign["vessel_id"].duplicated()].tolist()
        raise AssertionError(f"vessel 중복 배정 발생: {sorted(set(dup))}")
    if not assign["cargo_id"].is_unique:
        dup = assign["cargo_id"][assign["cargo_id"].duplicated()].tolist()
        raise AssertionError(f"cargo 중복 배정 발생: {sorted(set(dup))}")

    # 2) must cargo coverage
    must_cargos = set(
        base.loc[base["src_cargo"].astype(str).str.casefold() == args.must_src.casefold(), "cargo_id"]
        .astype(str)
        .unique()
    )
    chosen_cargos = set(assign["cargo_id"].astype(str))
    missing = must_cargos - chosen_cargos
    if missing:
        raise AssertionError(f"must cargo 누락: {sorted(missing)}")

    # 3) assigned_value matches base pair value
    base_key = base.copy()
    base_key["vessel_id"] = base_key["vessel_id"].astype(str)
    base_key["cargo_id"] = base_key["cargo_id"].astype(str)
    base_val = (
        base_key.groupby(["vessel_id", "cargo_id"])[args.value_col]
        .max()
        .rename("expected_value")
        .reset_index()
    )

    chk = assign.copy()
    chk["vessel_id"] = chk["vessel_id"].astype(str)
    chk["cargo_id"] = chk["cargo_id"].astype(str)
    chk = chk.merge(base_val, on=["vessel_id", "cargo_id"], how="left")

    if chk["expected_value"].isna().any():
        bad = chk.loc[chk["expected_value"].isna(), ["vessel_id", "cargo_id"]]
        raise AssertionError(f"base_cases에 없는 pair 존재: {bad.to_dict(orient='records')}")

    diff = (chk["assigned_value"] - chk["expected_value"]).abs()
    if (diff > args.tol).any():
        bad_rows = chk.loc[diff > args.tol, ["vessel_id", "cargo_id", "assigned_value", "expected_value"]]
        raise AssertionError(
            "assigned_value 불일치: "
            + bad_rows.to_dict(orient="records").__repr__()
        )

    # 4) total objective check (recomputed via value_col)
    recomputed_total = float(chk["expected_value"].sum())
    assigned_total = float(chk["assigned_value"].sum())
    if abs(recomputed_total - assigned_total) > args.total_tol:
        raise AssertionError(
            f"총합 불일치: recomputed={recomputed_total} assigned_sum={assigned_total}"
        )

    print(f"Recomputed total ({args.value_col}): {recomputed_total}")
    print("✅ Constraints OK")

    # 5) brute-force optimality check (optional)
    if args.bruteforce:
        vessels = base["vessel_id"].astype(str).unique().tolist()
        cargos = base["cargo_id"].astype(str).unique().tolist()
        must_cargos = (
            base.loc[
                base["src_cargo"].astype(str).str.casefold() == args.must_src.casefold(),
                "cargo_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        if args.bruteforce_target == "all":
            target_cargos = cargos
        else:
            target_cargos = must_cargos

        n_t = len(target_cargos)
        if n_t == 0:
            raise AssertionError("Bruteforce 대상 cargo가 없습니다.")
        if n_t > args.bruteforce_max:
            raise AssertionError(
                f"Bruteforce 대상 cargo 수가 너무 큽니다: {n_t} > {args.bruteforce_max}"
            )
        if len(vessels) < n_t:
            raise AssertionError(
                f"Bruteforce 불가: vessels({len(vessels)}) < target_cargos({n_t})"
            )

        pair_val = (
            base.groupby(["vessel_id", "cargo_id"])[args.value_col]
            .max()
        )

        best_score = -1e30
        best_assign = None
        for perm in itertools.permutations(vessels, r=n_t):
            score = 0.0
            ok = True
            for v, c in zip(perm, target_cargos):
                if (v, c) not in pair_val.index:
                    ok = False
                    break
                score += float(pair_val.loc[(v, c)])
            if not ok:
                continue
            if score > best_score:
                best_score = score
                best_assign = list(zip(perm, target_cargos))

        print(f"Bruteforce best ({args.value_col}): {best_score}")
        print(best_assign)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
