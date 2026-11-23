"""Build the offline browser snapshot from the repository's saved evidence."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]


def read_csv(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def source(label, path):
    normalized = (ROOT / path).read_text(encoding="utf-8").encode("utf-8")
    return {
        "label": label,
        "url": f"{REPOSITORY}/blob/main/{path}",
        "sha256_lf": hashlib.sha256(normalized).hexdigest(),
    }


def fact(label, value, kind=None):
    return {"label": label, "value": value, "format": kind}


def column(key, title, kind=None):
    return {"key": key, "title": title, "format": kind}


def order(key, title, descending=True):
    return {"key": key, "title": title, "id": key, "descending": descending}


def metric(label, value, note):
    return {"label": label, "value": value, "note": note}


def theme(accent, muted, base, glow):
    return {
        "accent": accent,
        "accent-soft": muted,
        "hero-base": base,
        "hero-glow": glow,
    }


REPOSITORY = "https://github.com/dev-belly/AuditLens"
QUEUE = "outputs/reports/review_plan.csv"
SUMMARY = "outputs/reports/review_plan_summary.json"


def build():
    summary = read_json(SUMMARY)
    routes = {
        "rule_coverage": "规则覆盖",
        "risk_priority": "风险优先",
        "random_control": "随机对照",
    }
    risks = {"Critical": "严重", "High": "高", "Medium": "中", "Low": "低"}
    records = []
    for row in read_csv(QUEUE):
        route = row["selection_route"]
        score = float(row["audit_risk_score"])
        amount = float(row["debit_amount"])
        facts = [
            fact("复核顺序", int(row["review_order"])),
            fact("供应商", row["vendor_id"]),
            fact("交易日期", row["transaction_date"]),
            fact("借方金额", amount, "cny"),
            fact("风险分数", score, "fixed2"),
            fact("风险分层", risks[row["risk_level"]]),
        ]
        if row["random_inclusion_probability"]:
            facts.append(
                fact(
                    "随机入选概率",
                    float(row["random_inclusion_probability"]),
                    "percent",
                )
            )
            facts.append(
                fact("随机抽样权重", float(row["random_sampling_weight"]), "fixed2")
            )
        sections = [{"title": "入选理由", "text": row["selection_reason"]}]
        if row["risk_reason_text"]:
            sections.append({"title": "风险证据", "text": row["risk_reason_text"]})
        if row["review_outcome"]:
            sections.append(
                {
                    "title": "样例中记录的复核结论",
                    "items": [
                        value
                        for value in (
                            row["review_outcome"],
                            row["evidence_reference"],
                            row["reviewer_notes"],
                        )
                        if value
                    ],
                }
            )
        records.append(
            {
                "id": row["transaction_id"],
                "title": row["transaction_id"],
                "subtitle": f"{routes[route]} · {row['vendor_id']}",
                "search": f"{routes[route]} {risks[row['risk_level']]} {row['rule_keys']} {row['vendor_id']}",
                "values": {
                    "route": route,
                    "risk": row["risk_level"],
                    "score": score,
                    "amount": amount,
                },
                "facts": facts,
                "sections": sections,
            }
        )
    selected = summary["selected_by_route"]
    return {
        "repository": REPOSITORY,
        "slug": "auditlens",
        "project": "AuditLens",
        "mark": "AL",
        "theme": theme("#684896", "#f4edff", "#261c45", "#604493"),
        "workspace": "审计取样",
        "title": "审计线索复核",
        "description": "沿着 300 条凭证的入选路径，查看风险得分、触发原因和随机对照。支持按供应商或凭证号定位。",
        "dataKind": "合成总账 · 固定复核计划",
        "hero": metric(
            "复核计划",
            f"{len(records):,} 条",
            f"凭证总体 {summary['population_count']:,} 条",
        ),
        "metrics": [
            metric("规则覆盖", str(selected["rule_coverage"]), "用于覆盖不同审计程序"),
            metric("风险优先", str(selected["risk_priority"]), "按证据风险评分排列"),
            metric(
                "随机对照", str(selected["random_control"]), "从其余凭证简单随机抽样"
            ),
        ],
        "recordHeading": "待复核凭证",
        "recordLabel": "凭证 ID",
        "detailHeading": "凭证证据",
        "searchPlaceholder": "凭证号、供应商或规则",
        "initialRecord": records[0]["id"],
        "filters": [
            {"key": "route", "title": "入选路径", "labels": routes},
            {"key": "risk", "title": "风险层级", "labels": risks},
        ],
        "columns": [
            column("risk", "风险"),
            column("score", "得分", "fixed2"),
            column("amount", "借方金额", "cny"),
            column("route", "入选路径"),
        ],
        "orders": [order("score", "风险分从高到低"), order("amount", "金额从高到低")],
        "views": [
            {
                "id": "route",
                "label": "入选路径",
                "title": "复核计划构成",
                "mode": "group",
                "group": "route",
                "format": "integer",
                "caption": "分布随搜索和风险层级更新。点击条形切换入选路径。",
            },
            {
                "id": "risk",
                "label": "风险分层",
                "title": "复核计划风险分层",
                "mode": "group",
                "group": "risk",
                "format": "integer",
                "caption": "分布随搜索和入选路径更新。点击条形切换风险等级。",
            },
        ],
        "sourceNote": "凭证、供应商和异常均来自仓库的合成数据。风险分用于排序，凭证的最终处理仍需核对底稿与原始证据。",
        "sources": [
            source("复核计划 CSV", QUEUE),
            source("抽样预算与来源指纹", SUMMARY),
            source("审计口径与完整应用", "README.zh-CN.md"),
        ],
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify the generated snapshot"
    )
    args = parser.parse_args()
    document = build()
    records = document.pop("records")
    lines = ",\n".join(
        "    " + json.dumps(record, ensure_ascii=False, allow_nan=False)
        for record in records
    )
    header = json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2)
    result = (
        "// Generated by build_data.py from repository evidence.\n"
        "window.workbenchData = "
        + header[:-2]
        + ',\n  "records": [\n'
        + lines
        + "\n  ]\n};\n"
    )
    path = DIRECTORY / "data.js"
    if args.check:
        if not path.exists() or path.read_text(encoding="utf-8") != result:
            parser.exit(1, "Snapshot differs. Run build_data.py without --check.\n")
        print(f"Verified {len(records)} records")
    else:
        path.write_text(result, encoding="utf-8", newline="\n")
        print(f"Wrote {len(records)} records")


if __name__ == "__main__":
    main()
