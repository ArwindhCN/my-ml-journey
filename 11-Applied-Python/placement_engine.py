"""
11.6 — Campus Placement Eligibility Engine
A rule engine built with every pattern from Chapter 11.
"""

from __future__ import annotations
import asyncio, difflib, itertools, re, time
from typing import Literal, Optional, List, Callable
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


# ══════════════════════════════════════════════════════════════
# 11.1 — THE BOUNDARY
# ══════════════════════════════════════════════════════════════

class ServiceUnavailable(Exception):
    """Raised when an external dependency cannot be reached."""


class Student(BaseModel):
    model_config = ConfigDict(extra="forbid")

    student_id:        str
    cgpa:              float = Field(..., ge=0.0, le=10.0)
    active_backlogs:   int   = Field(..., ge=0, le=50)
    history_of_arrears:int   = Field(..., ge=0, le=100)
    attendance_pct:    float = Field(..., ge=0.0, le=100.0)
    branch:            str
    graduation_year:   int   = Field(..., ge=2000, le=2100)
    offers_held:       int   = Field(..., ge=0, le=20)
    highest_offer_lpa: float = Field(default=0.0, ge=0.0, le=1000.0)

    @field_validator("student_id")
    @classmethod
    def valid_roll_format(cls, v: str) -> str:
        if not re.fullmatch(r"CB\.EN\.U4[A-Z]{3}\d{5}", v.strip().upper()):
            raise ValueError("student_id must look like CB.EN.U4CSE22045")
        return v.strip().upper()

    @field_validator("branch")
    @classmethod
    def normalise_branch(cls, v: str) -> str:
        return v.strip().upper()

    @model_validator(mode="after")
    def arrears_consistency(self) -> "Student":
        if self.active_backlogs > self.history_of_arrears:
            raise ValueError(
                "active_backlogs cannot exceed history_of_arrears")
        return self

    @model_validator(mode="after")
    def offer_consistency(self) -> "Student":
        if self.offers_held == 0 and self.highest_offer_lpa > 0:
            raise ValueError(
                "highest_offer_lpa must be 0 when offers_held is 0")
        if self.offers_held > 0 and self.highest_offer_lpa == 0:
            raise ValueError(
                "highest_offer_lpa must be > 0 when offers_held is > 0")
        return self


class Criteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company:             str
    min_cgpa:            float = Field(..., ge=0.0, le=10.0)
    max_active_backlogs: int   = Field(..., ge=0, le=50)
    max_history_arrears: int   = Field(..., ge=0, le=100)
    min_attendance_pct:  float = Field(..., ge=0.0, le=100.0)
    allowed_branches:    List[str]
    package_lpa:         float = Field(..., gt=0.0, le=1000.0)

    @field_validator("allowed_branches")
    @classmethod
    def no_duplicate_branches(cls, v: List[str]) -> List[str]:
        names = [b.strip().upper() for b in v]
        if len(names) != len(set(names)):
            raise ValueError("duplicate branch in allowed_branches")
        if not names:
            raise ValueError("allowed_branches cannot be empty")
        return names


# ══════════════════════════════════════════════════════════════
# 11.4 — THE OUTBOUND CONTRACT
# ══════════════════════════════════════════════════════════════

Verdict  = Literal["ELIGIBLE", "BLOCKED", "CHECK_UNAVAILABLE"]
Severity = Literal["INFO", "SOFT_BLOCK", "HARD_BLOCK", "SYSTEM"]


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id:     str
    severity:    Severity
    explanation: str
    evidence:    str
    source:      str


class EligibilityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    student_id:       str
    company:          str
    verdict:          Verdict
    findings:         List[Finding] = []
    checks_completed: List[str] = []
    checks_failed:    List[str] = []


# ══════════════════════════════════════════════════════════════
# 11.2 — PURE RULES  (student, criteria) -> Finding | None
# ══════════════════════════════════════════════════════════════

def rule_cgpa(s: Student, c: Criteria, _svc=None) -> Optional[Finding]:
    if s.cgpa < c.min_cgpa:
        return Finding(
            rule_id="CGPA_BELOW_CUTOFF", severity="HARD_BLOCK",
            explanation=f"CGPA {s.cgpa} is below the required {c.min_cgpa}.",
            evidence=f"{c.company} eligibility notice, CGPA clause",
            source=f"{c.company} JD 2026")
    return None


def rule_active_backlogs(s: Student, c: Criteria, _svc=None) -> Optional[Finding]:
    if s.active_backlogs > c.max_active_backlogs:
        return Finding(
            rule_id="ACTIVE_BACKLOGS", severity="HARD_BLOCK",
            explanation=(f"{s.active_backlogs} active backlog(s); "
                         f"maximum permitted is {c.max_active_backlogs}."),
            evidence=f"{c.company} eligibility notice, arrears clause",
            source=f"{c.company} JD 2026")
    return None


def rule_history_arrears(s: Student, c: Criteria, _svc=None) -> Optional[Finding]:
    if s.history_of_arrears > c.max_history_arrears:
        return Finding(
            rule_id="ARREARS_HISTORY", severity="HARD_BLOCK",
            explanation=(f"{s.history_of_arrears} historical arrears; "
                         f"maximum permitted is {c.max_history_arrears}."),
            evidence=f"{c.company} eligibility notice, arrears clause",
            source=f"{c.company} JD 2026")
    return None


def rule_attendance(s: Student, c: Criteria, _svc=None) -> Optional[Finding]:
    if s.attendance_pct < c.min_attendance_pct:
        return Finding(
            rule_id="ATTENDANCE_SHORT", severity="SOFT_BLOCK",
            explanation=(f"Attendance {s.attendance_pct}% is below "
                         f"{c.min_attendance_pct}%. Condonation may apply."),
            evidence="Amrita academic regulations, attendance condonation clause",
            source="Amrita Academic Handbook 2025")
    return None


def rule_branch(s: Student, c: Criteria, _svc=None) -> Optional[Finding]:
    if s.branch not in c.allowed_branches:
        return Finding(
            rule_id="BRANCH_NOT_ELIGIBLE", severity="HARD_BLOCK",
            explanation=(f"Branch {s.branch} is not in the eligible list "
                         f"({', '.join(c.allowed_branches)})."),
            evidence=f"{c.company} eligibility notice, branch clause",
            source=f"{c.company} JD 2026")
    return None


def rule_placement_policy(s: Student, c: Criteria, _svc=None) -> Optional[Finding]:
    """One-offer policy: a placed student may only apply for a materially
       better package (>= 1.5x their current highest)."""
    if s.offers_held == 0:
        return None
    if c.package_lpa >= s.highest_offer_lpa * 1.5:
        return Finding(
            rule_id="DREAM_OFFER_PERMITTED", severity="INFO",
            explanation=(f"Already placed at {s.highest_offer_lpa} LPA; this "
                         f"{c.package_lpa} LPA role qualifies as a dream offer."),
            evidence="Placement policy, dream-offer threshold 1.5x",
            source="Amrita Placement Policy 2026")
    return Finding(
        rule_id="ALREADY_PLACED", severity="HARD_BLOCK",
        explanation=(f"Already placed at {s.highest_offer_lpa} LPA. This role "
                     f"({c.package_lpa} LPA) is below the 1.5x dream-offer bar."),
        evidence="Placement policy, one-offer rule",
        source="Amrita Placement Policy 2026")


def rule_disciplinary(s: Student, c: Criteria, svc) -> Optional[Finding]:
    """The only rule needing an external service — may raise."""
    if svc.has_active_hold(s.student_id):
        return Finding(
            rule_id="DISCIPLINARY_HOLD", severity="HARD_BLOCK",
            explanation="An active disciplinary hold blocks placement activity.",
            evidence="Registrar disciplinary register",
            source="Registrar Office")
    return None


RULES: List[Callable] = [
    rule_cgpa, rule_active_backlogs, rule_history_arrears,
    rule_attendance, rule_branch, rule_placement_policy, rule_disciplinary,
]


# ══════════════════════════════════════════════════════════════
# 11.3 — HYBRID: branch-name resolution, rules → fuzzy → gate
# ══════════════════════════════════════════════════════════════

CANONICAL_BRANCHES = ["CSE", "IT", "ECE", "EEE", "MECH", "CIVIL", "AIE"]
ALIASES = {
    "COMPUTER SCIENCE": "CSE", "COMPUTER SCIENCE AND ENGINEERING": "CSE",
    "CS": "CSE", "COMP SCI": "CSE",
    "INFORMATION TECHNOLOGY": "IT",
    "ELECTRONICS AND COMMUNICATION": "ECE", "E&C": "ECE",
    "ELECTRICAL AND ELECTRONICS": "EEE",
    "MECHANICAL": "MECH", "MECHANICAL ENGINEERING": "MECH",
    "ARTIFICIAL INTELLIGENCE": "AIE",
}


class BranchMatch(BaseModel):
    raw: str
    resolved: Optional[str]
    method: str
    confidence: float
    action: Literal["ACCEPT", "HUMAN_REVIEW"]


def resolve_branch(raw: str, threshold: float = 0.90) -> BranchMatch:
    s = re.sub(r"[^A-Z& ]", " ", raw.upper())
    s = re.sub(r"\s+", " ", s).strip()

    # STAGE 1 — rules. Exact, free, no failure mode.
    if s in CANONICAL_BRANCHES:
        return BranchMatch(raw=raw, resolved=s, method="exact",
                           confidence=1.0, action="ACCEPT")
    if s in ALIASES:
        return BranchMatch(raw=raw, resolved=ALIASES[s], method="alias",
                           confidence=1.0, action="ACCEPT")

    # STAGE 2 — statistics, on the residual only.
    pool = CANONICAL_BRANCHES + list(ALIASES)
    close = difflib.get_close_matches(s, pool, n=1, cutoff=0.6)
    if not close:
        return BranchMatch(raw=raw, resolved=None, method="fuzzy",
                           confidence=0.0, action="HUMAN_REVIEW")

    cand = close[0]
    conf = difflib.SequenceMatcher(None, s, cand).ratio()
    target = ALIASES.get(cand, cand)

    # STAGE 3 — the gate. Uncertainty stops here.
    if conf >= threshold:
        return BranchMatch(raw=raw, resolved=target, method="fuzzy",
                           confidence=round(conf, 3), action="ACCEPT")
    return BranchMatch(raw=raw, resolved=None, method="fuzzy",
                       confidence=round(conf, 3), action="HUMAN_REVIEW")


# ══════════════════════════════════════════════════════════════
# THE ENGINE
# ══════════════════════════════════════════════════════════════

def evaluate(student: Student, criteria: Criteria, svc) -> EligibilityResponse:
    findings, completed, failed = [], [], []

    for rule in RULES:
        try:
            result = rule(student, criteria, svc)
            completed.append(rule.__name__)
            if result:
                findings.append(result)
        except ServiceUnavailable:
            failed.append(rule.__name__)
            findings.append(Finding(
                rule_id="CHECK_UNAVAILABLE", severity="SYSTEM",
                explanation=(f"{rule.__name__} could not complete. This "
                             f"student was NOT screened on that criterion."),
                evidence="", source="unavailable"))

    if any(f.severity == "SYSTEM" for f in findings):
        verdict: Verdict = "CHECK_UNAVAILABLE"
    elif any(f.severity in ("HARD_BLOCK", "SOFT_BLOCK") for f in findings):
        verdict = "BLOCKED"
    else:
        verdict = "ELIGIBLE"

    return EligibilityResponse(
        student_id=student.student_id, company=criteria.company,
        verdict=verdict, findings=findings,
        checks_completed=completed, checks_failed=failed)


# ══════════════════════════════════════════════════════════════
# 11.5 — CONCURRENCY: one student across many companies
# ══════════════════════════════════════════════════════════════

LATENCY = 0.04


class RegistrarService:
    HOLDS = {"CB.EN.U4CSE22099"}
    def has_active_hold(self, sid: str) -> bool:
        time.sleep(LATENCY)
        return sid in self.HOLDS


class DownRegistrar:
    def has_active_hold(self, sid: str) -> bool:
        raise ServiceUnavailable("Registrar service unreachable")


async def evaluate_all_sequential(student, criteria_list, svc):
    return [evaluate(student, c, svc) for c in criteria_list]


async def evaluate_all_concurrent(student, criteria_list, svc):
    return await asyncio.gather(*[
        asyncio.to_thread(evaluate, student, c, svc) for c in criteria_list])
