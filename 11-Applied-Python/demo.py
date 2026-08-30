import asyncio, json, time
from pydantic import ValidationError
from placement_engine import *

COMPANIES = [
    Criteria(company="Zoho", min_cgpa=7.0, max_active_backlogs=0,
             max_history_arrears=3, min_attendance_pct=75,
             allowed_branches=["CSE","IT","ECE"], package_lpa=9.0),
    Criteria(company="Freshworks", min_cgpa=8.0, max_active_backlogs=0,
             max_history_arrears=0, min_attendance_pct=75,
             allowed_branches=["CSE","IT"], package_lpa=12.0),
    Criteria(company="TCS", min_cgpa=6.0, max_active_backlogs=1,
             max_history_arrears=5, min_attendance_pct=70,
             allowed_branches=["CSE","IT","ECE","EEE","MECH"], package_lpa=7.0),
    Criteria(company="Google", min_cgpa=8.5, max_active_backlogs=0,
             max_history_arrears=0, min_attendance_pct=75,
             allowed_branches=["CSE"], package_lpa=45.0),
]

print("="*66); print("1. BOUNDARY (11.1) — what the door rejects"); print("="*66)
bad = [
 ("CGPA above scale", dict(student_id="CB.EN.U4CSE22045", cgpa=11.5, active_backlogs=0,
    history_of_arrears=0, attendance_pct=80, branch="CSE", graduation_year=2026,
    offers_held=0, highest_offer_lpa=0)),
 ("negative attendance", dict(student_id="CB.EN.U4CSE22045", cgpa=8.0, active_backlogs=0,
    history_of_arrears=0, attendance_pct=-5, branch="CSE", graduation_year=2026,
    offers_held=0, highest_offer_lpa=0)),
 ("bad roll format", dict(student_id="22045", cgpa=8.0, active_backlogs=0,
    history_of_arrears=0, attendance_pct=80, branch="CSE", graduation_year=2026,
    offers_held=0, highest_offer_lpa=0)),
 ("active > history", dict(student_id="CB.EN.U4CSE22045", cgpa=8.0, active_backlogs=3,
    history_of_arrears=1, attendance_pct=80, branch="CSE", graduation_year=2026,
    offers_held=0, highest_offer_lpa=0)),
 ("placed but 0 LPA", dict(student_id="CB.EN.U4CSE22045", cgpa=8.0, active_backlogs=0,
    history_of_arrears=0, attendance_pct=80, branch="CSE", graduation_year=2026,
    offers_held=1, highest_offer_lpa=0)),
 ("typo'd field name", dict(student_id="CB.EN.U4CSE22045", CGPA=8.0, active_backlogs=0,
    history_of_arrears=0, attendance_pct=80, branch="CSE", graduation_year=2026,
    offers_held=0, highest_offer_lpa=0)),
]
for label, payload in bad:
    try:
        Student(**payload); print(f"  {label:22} -> ACCEPTED (BUG!)")
    except ValidationError as e:
        err = e.errors()[0]
        loc = err['loc'][0] if err['loc'] else '(model)'
        print(f"  {label:22} -> rejected [{loc}] {err['msg'][:52]}")

print("\n  coercion (lax mode): cgpa='7.84' ->", 
      Student(student_id="CB.EN.U4CSE22045", cgpa="7.84", active_backlogs=0,
              history_of_arrears=0, attendance_pct=80, branch=" cse ",
              graduation_year=2026, offers_held=0, highest_offer_lpa=0).cgpa,
      "| branch ' cse ' ->",
      Student(student_id="CB.EN.U4CSE22045", cgpa=7.84, active_backlogs=0,
              history_of_arrears=0, attendance_pct=80, branch=" cse ",
              graduation_year=2026, offers_held=0, highest_offer_lpa=0).branch)

print("\n"+"="*66); print("2. RULES (11.2) — Arwindh vs four companies"); print("="*66)
arwindh = Student(student_id="CB.EN.U4CSE22045", cgpa=7.84, active_backlogs=0,
    history_of_arrears=2, attendance_pct=71.5, branch="CSE",
    graduation_year=2026, offers_held=1, highest_offer_lpa=8.5)
svc = RegistrarService()
for c in COMPANIES:
    r = evaluate(arwindh, c, svc)
    print(f"\n  {c.company} ({c.package_lpa} LPA) -> {r.verdict}")
    for f in r.findings:
        print(f"     [{f.severity:10}] {f.rule_id:24} {f.explanation[:58]}")

print("\n"+"="*66); print("3. ORDER INDEPENDENCE (11.2)"); print("="*66)
import random
sigs=set()
for _ in range(6):
    random.shuffle(RULES)
    sigs.add(tuple(sorted(f.rule_id for f in evaluate(arwindh, COMPANIES[0], svc).findings)))
print(f"  6 shuffled orders -> {len(sigs)} distinct result(s): {sigs}")

print("\n"+"="*66); print("4. DEGRADED (11.2) — registrar down"); print("="*66)
r = evaluate(arwindh, COMPANIES[2], DownRegistrar())
print(f"  verdict          : {r.verdict}")
print(f"  checks_completed : {len(r.checks_completed)}")
print(f"  checks_failed    : {r.checks_failed}")
print(f"  system finding   : {[f.explanation[:60] for f in r.findings if f.severity=='SYSTEM']}")

print("\n"+"="*66); print("5. HYBRID BRANCH RESOLUTION (11.3)"); print("="*66)
for raw in ["CSE","Computer Science","cse","COMPUTER SCIENCE AND ENGINEERING",
            "Mechnical","Compter Sciene","Nanotechnology","E&C"]:
    m = resolve_branch(raw)
    print(f"  {raw:36} {m.method:6} conf={m.confidence:<6} {m.action:13} -> {m.resolved}")

print("\n"+"="*66); print("6. CONTRACT (11.4)"); print("="*66)
try:
    Finding(rule_id="X", severity="Severe", explanation="e", evidence="e", source="s")
except ValidationError as e:
    print(f"  severity='Severe'  -> rejected: {e.errors()[0]['type']}")
try:
    Finding(rule_id="X", severity="HARD_BLOCK", message="renamed", evidence="e", source="s")
except ValidationError as e:
    print(f"  renamed field      -> {[(x['loc'][0], x['type']) for x in e.errors()]}")
print("\n  wire format:")
print(json.dumps(evaluate(arwindh, COMPANIES[1], svc).model_dump(), indent=2)[:700])

print("\n"+"="*66); print("7. CONCURRENCY (11.5) — 4 companies, 40ms registrar call each"); print("="*66)
async def main():
    t=time.perf_counter(); await evaluate_all_sequential(arwindh, COMPANIES, svc)
    seq=(time.perf_counter()-t)*1000
    t=time.perf_counter(); await evaluate_all_concurrent(arwindh, COMPANIES, svc)
    con=(time.perf_counter()-t)*1000
    print(f"  sequential : {seq:6.1f}ms")
    print(f"  concurrent : {con:6.1f}ms   ({seq/con:.1f}x faster)")
asyncio.run(main())
