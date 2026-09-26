"""Page 4 - Benford's Law analysis.

The digit test is the analytical procedure auditors most often get wrong, usually
by treating it as a fraud detector. This page is built to prevent that reading. It
shows the test, the conformity statistics, and - more usefully - the disaggregation,
because a population-level MAD of 0.002 tells you almost nothing while the two
accounts driving the deviation tell you where to look.

The caveat is stated on the page, not buried in a footnote: Benford's Law cannot
prove fraud, and it cannot clear anyone either.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (
    RISK_BAND_COLORS,
    THEME,
    count,
    load_benford,
    money,
)
from dashboard.components import (
    kpi_row,
    note,
    page_header,
    section,
)
from src.utils import ACCOUNTS, BENFORD_MAD_THRESHOLDS

#: Colour a conformity band so the reader does not have to remember Nigrini's
#: thresholds to interpret the table.
CONFORMITY_COLORS: dict[str, str] = {
    "Close conformity": RISK_BAND_COLORS["Low"],
    "Acceptable conformity": "#7FB069",
    "Marginal conformity": RISK_BAND_COLORS["Medium"],
    "Nonconformity": RISK_BAND_COLORS["Critical"],
}


def _digit_figure(table: pd.DataFrame, title: str, *, width: int = 900) -> go.Figure:
    """Build the observed-versus-expected digit chart."""
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=table["digit"].astype(str),
            y=table["observed_frequency"],
            name="Observed",
            marker_color=THEME["primary"],
            text=[f"{value:.1%}" for value in table["observed_frequency"]],
            textposition="outside",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=table["digit"].astype(str),
            y=table["expected_frequency"],
            name="Benford expected",
            mode="lines+markers",
            line=dict(color="#D2691E", width=2.5),
            marker=dict(size=8, symbol="diamond"),
        )
    )
    figure.update_layout(
        title=dict(text=title, font=dict(size=13, color=THEME["primary"])),
        height=380,
        margin=dict(l=10, r=20, t=50, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
        bargap=0.35,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis=dict(title="Frequency", tickformat=".1%", gridcolor="#EDF1F4"),
        xaxis=dict(title="Leading digit", gridcolor="#EDF1F4"),
    )
    return figure


def _conformity_banner(result: dict) -> str:
    """Return an HTML block describing one Benford test result."""
    conformity = result.get("conformity", "n/a")
    colour = CONFORMITY_COLORS.get(conformity, THEME["muted"])
    return f"""
        <div class='al-rule' style='border-left:4px solid {colour};'>
            <div class='al-kpi-label'>{result.get('label', 'Benford test')}</div>
            <div style='font-size:1.35rem;font-weight:700;color:{colour};'>{conformity}</div>
            <div class='al-caption' style='margin-top:0.35rem;'>
                MAD <b>{float(result.get('mad') or 0):.5f}</b> &nbsp;|&nbsp;
                &chi;&sup2; <b>{float(result.get('chi_square') or 0):,.1f}</b> &nbsp;|&nbsp;
                df {result.get('degrees_of_freedom', '-')} &nbsp;|&nbsp;
                p <b>{float(result.get('p_value') or 0):.4g}</b>
            </div>
            <div class='al-caption' style='margin-top:0.35rem;'>
                {count(result.get('n_observations', 0))} values tested,
                amounts of CNY {float(result.get('min_amount') or 0):,.0f} and above
            </div>
        </div>
    """


def _digit_table(table: pd.DataFrame) -> pd.DataFrame:
    """Format the per-digit table for display."""
    display = pd.DataFrame(
        {
            "Digit": table["digit"].astype(str),
            "Observed count": table["observed_count"].map(count),
            "Observed": table["observed_frequency"].map(lambda value: f"{value:.4%}"),
            "Expected": table["expected_frequency"].map(lambda value: f"{value:.4%}"),
            "Deviation": table["deviation"].map(lambda value: f"{value:+.4%}"),
            "Z-score": table["z_score"].round(2),
            "Flagged": table["significant"].map(lambda flag: "Yes" if flag else ""),
        }
    )
    return display


def _disaggregation(rows: list[dict], key: str, *, key_label: str) -> pd.DataFrame:
    """Build the disaggregated conformity table, enriched with account names."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    if key == "account_code":
        frame["name"] = frame["account_code"].map(
            lambda code: ACCOUNTS.get(str(code), {}).get("name", "")
        )
        frame["label"] = frame["account_code"].astype(str) + "  " + frame["name"].astype(str)
    else:
        frame["label"] = frame[key].astype(str).str.replace("_", " ").str.title()

    frame = frame.sort_values("mad", ascending=False)
    return pd.DataFrame(
        {
            key_label: frame["label"],
            "Observations": frame["n_observations"].map(count),
            "Total value": frame["total_amount"].map(money),
            "MAD": frame["mad"].round(5),
            "Chi-square": frame["chi_square"].round(1),
            "P-value": frame["p_value"].map(lambda value: f"{value:.3g}"),
            "Conformity": frame["conformity"],
            "Risk indicator": frame["risk_indicator"].round(2),
        }
    )


def render() -> None:
    """Render the Benford analysis page."""
    page_header(
        "Benford's Law Analysis",
        "First-digit and first-two-digit tests on the debit leg, then the "
        "disaggregation that actually tells you where to look.",
        eyebrow="AuditLens - Analytical procedures",
    )

    benford = load_benford()
    if not benford:
        st.error("Benford results not found. Run `python src/benford.py` first.")
        return

    first_digit = benford.get("first_digit_test", {}) or {}
    first_two = benford.get("first_two_digits_test", {}) or {}
    digit_table = pd.DataFrame(benford.get("first_digit_table", []))
    two_digit_table = pd.DataFrame(benford.get("first_two_digits_table", []))

    # ----------------------------------------------------------------- #
    # Headline
    # ----------------------------------------------------------------- #
    kpi_row(
        [
            {"label": "First digit - MAD",
             "value": f"{float(first_digit.get('mad') or 0):.5f}",
             "delta": first_digit.get("conformity", "-"),
             "accent": CONFORMITY_COLORS.get(first_digit.get("conformity", ""), THEME["primary"])},
            {"label": "First digit - Chi-square",
             "value": f"{float(first_digit.get('chi_square') or 0):,.1f}",
             "delta": f"p = {float(first_digit.get('p_value') or 0):.4f} on "
                      f"{first_digit.get('degrees_of_freedom', '-')} df"},
            {"label": "First two digits - MAD",
             "value": f"{float(first_two.get('mad') or 0):.5f}",
             "delta": first_two.get("conformity", "-"),
             "accent": CONFORMITY_COLORS.get(first_two.get("conformity", ""), THEME["primary"])},
            {"label": "Values tested",
             "value": count(first_digit.get("n_observations", 0)),
             "delta": "Debit leg, one per voucher", "accent": THEME["muted"]},
        ]
    )

    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown(_conformity_banner(first_digit), unsafe_allow_html=True)
    with right:
        st.markdown(_conformity_banner(first_two), unsafe_allow_html=True)

    st.write("")
    note(
        "<b>Benford's Law is an analytical procedure, not a test for fraud.</b> "
        "A conforming digit distribution is entirely compatible with fabricated "
        "entries, and a nonconforming one is usually just a legitimately skewed "
        "account - payroll, fixed asset additions and any account with a narrow "
        "range of plausible values will fail the test for innocent reasons. The "
        "test identifies a population that deserves disaggregation and enquiry. "
        "It does not establish that anything is wrong, and it cannot clear anyone.",
        kind="warn",
    )

    # ----------------------------------------------------------------- #
    # Digit distributions
    # ----------------------------------------------------------------- #
    section(
        "Observed versus expected",
        "The bar is what the ledger actually contains; the line is what Benford's "
        "Law predicts. Digits flagged as significant have a z-score beyond +/-1.96.",
    )

    if not digit_table.empty:
        st.plotly_chart(_digit_figure(digit_table, "First digit distribution"), width="stretch")
        with st.expander("First-digit table"):
            st.dataframe(_digit_table(digit_table), width="stretch", hide_index=True)
            significant = digit_table[digit_table["significant"]]
            if significant.empty:
                st.markdown(
                    "<div class='al-caption'>No individual digit deviates significantly "
                    "from expectation.</div>",
                    unsafe_allow_html=True,
                )
            else:
                names = ", ".join(significant["digit"].astype(str))
                st.markdown(
                    f"<div class='al-caption'>Digits deviating significantly: <b>{names}</b>. "
                    f"With {count(first_digit.get('n_observations', 0))} observations a "
                    f"single significant digit is expected by chance, so read this "
                    f"alongside the overall MAD rather than on its own.</div>",
                    unsafe_allow_html=True,
                )

    if not two_digit_table.empty:
        with st.expander("First-two-digit table (90 buckets)"):
            display = pd.DataFrame(
                {
                    "Bucket": two_digit_table["digit"].astype(str),
                    "Observed": two_digit_table["observed_frequency"].map(lambda value: f"{value:.4%}"),
                    "Expected": two_digit_table["expected_frequency"].map(lambda value: f"{value:.4%}"),
                    "Deviation": two_digit_table["deviation"].map(lambda value: f"{value:+.4%}"),
                    "Z-score": two_digit_table["z_score"].round(2),
                    "Flagged": two_digit_table["significant"].map(lambda flag: "Yes" if flag else ""),
                }
            )
            st.dataframe(display, width="stretch", hide_index=True, height=320)

    # ----------------------------------------------------------------- #
    # Conformity bands
    # ----------------------------------------------------------------- #
    with st.expander("How the conformity bands are defined"):
        close = BENFORD_MAD_THRESHOLDS["close_conformity"]
        acceptable = BENFORD_MAD_THRESHOLDS["acceptable_conformity"]
        marginal = BENFORD_MAD_THRESHOLDS["marginal_conformity"]
        bands = pd.DataFrame(
            [
                {"Band": "Close conformity", "MAD range": f"0.000 - {close:.3f}",
                 "Reading": "No further digit testing indicated."},
                {"Band": "Acceptable conformity", "MAD range": f"{close:.3f} - {acceptable:.3f}",
                 "Reading": "Consistent with Benford; monitor."},
                {"Band": "Marginal conformity", "MAD range": f"{acceptable:.3f} - {marginal:.3f}",
                 "Reading": "Disaggregate before concluding anything."},
                {"Band": "Nonconformity", "MAD range": f"above {marginal:.3f}",
                 "Reading": "Investigate at account level; the cause is usually legitimate."},
            ]
        )
        st.dataframe(bands, width="stretch", hide_index=True)
        st.markdown(
            "<div class='al-caption'>Thresholds follow the mean absolute deviation "
            "bands in Nigrini's work on digital analysis. They are conventions, not "
            "statistical laws.</div>",
            unsafe_allow_html=True,
        )

    # ----------------------------------------------------------------- #
    # Disaggregation - the useful part
    # ----------------------------------------------------------------- #
    section(
        "Disaggregation by account",
        "This is where the test earns its keep. A population-level MAD averages away "
        "everything interesting; the account level is where a real enquiry starts.",
    )

    by_account = _disaggregation(benford.get("by_account", []) or [], "account_code", key_label="Account")
    if not by_account.empty:
        st.dataframe(by_account, width="stretch", hide_index=True)

        nonconforming = by_account[by_account["Conformity"] == "Nonconformity"]
        if not nonconforming.empty:
            note(
                f"<b>{len(nonconforming)} account(s) fail the digit test at account "
                f"level.</b> The worst is <b>{nonconforming.iloc[0]['Account']}</b> "
                f"(MAD {nonconforming.iloc[0]['MAD']}, risk indicator "
                f"{nonconforming.iloc[0]['Risk indicator']}). Before treating this as "
                f"a finding, check whether the account has a narrow plausible range - "
                f"payroll, depreciation and fixed-asset additions routinely fail "
                f"Benford for entirely legitimate reasons. The test is a pointer to a "
                f"population, not evidence of misstatement."
            )

    section(
        "Disaggregation by business process",
        "The same test on the process dimension, which often localises a deviation "
        "faster than the account dimension.",
    )
    by_process = _disaggregation(benford.get("by_process", []) or [], "process", key_label="Process")
    if not by_process.empty:
        st.dataframe(by_process, width="stretch", hide_index=True)

    scope = benford.get("scope_note")
    if scope:
        st.write("")
        st.markdown(f"<div class='al-caption'><b>Scope:</b> {scope}</div>", unsafe_allow_html=True)
