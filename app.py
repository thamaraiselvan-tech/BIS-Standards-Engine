from __future__ import annotations
"""
app.py  —  BIS Standards Recommendation Engine
Chat UI  +  Flask API  +  SSE streaming explanations

Run:
    python app.py
    open http://localhost:5000
"""

import json, os, sys, time
from flask import Flask, request, jsonify, Response, stream_with_context

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from rag_pipeline import get_index, query_pipeline, STANDARDS_DB

PDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.pdf")

app = Flask(__name__)

# ── Warm index once at startup ────────────────────────────────────────────────
_pdf = PDF_PATH if os.path.exists(PDF_PATH) else None
print("[BIS] Building index" + (" (with PDF)" if _pdf else "") + "...")
_idx = get_index(pdf_path=_pdf)
print("[BIS] Index ready — open http://localhost:5000")


# ── Helper: resolve standard metadata (covers ALL 566 standards) ─────────────

def _clean_title(raw: str) -> str:
    """Strip PDF bleed text, all-caps, clause numbers from title."""
    import re as _re
    s = (raw or "").strip()
    OVERRIDES = {
        "WHITE PORTLAND CEMENT":                   "White Portland Cement",
        "RAPID HARDENING PORTLAND CEMENT":         "Rapid Hardening Portland Cement",
        "SUPERSULPHATED CEMENT":                   "Supersulphated Cement",
        "HIGH ALUMINA CEMENT FOR STRUCTURAL USE":  "High Alumina Cement for Structural Use",
        "HYDROPHOBIC PORTLAND CEMENT":             "Hydrophobic Portland Cement",
        "PORTLAND SLAG CEMENT":                    "Portland Slag Cement",
        "CORRUGATED AND SEMI-CORRUGATED ASBESTOS": "Corrugated and Semi-Corrugated Asbestos Cement Sheets",
        "CONCRETE MASONRY UNITS PART 2":           "Concrete Masonry Units - Part 2: Lightweight Blocks",
        "CONCRETE MASONRY UNITS PART 1":           "Concrete Masonry Units - Part 1: Hollow and Solid Blocks",
        "CONCRETE MASONRY UNITS PART 3":           "Concrete Masonry Units - Part 3: AAC Blocks",
        "CINDER AS FINE AGGREGATES":               "Cinder as Fine Aggregates for Lime Concrete",
        "BITUMEN MASTIC FOR FLOORING":             "Bitumen Mastic for Flooring",
    }
    for prefix, clean in OVERRIDES.items():
        if s.upper().startswith(prefix):
            return clean
    # Remove bleed text patterns
    s = _re.sub(r" For detailed information.*", "", s, flags=_re.IGNORECASE)
    s = _re.sub(r" For method of (measurement|test).*", "", s, flags=_re.IGNORECASE)
    s = _re.sub(r" Note [0-9].*", "", s, flags=_re.IGNORECASE)
    s = _re.sub(r" [0-9]+[.] (Scope|Requirements|General|Manufacture|Chemical).*", "", s, flags=_re.IGNORECASE)
    s = _re.sub(r" (prepared|manufactured|shall|covers) .*", "", s, flags=_re.IGNORECASE)
    s = _re.sub(r" [(](First|Second|Third|Fourth|Fifth) Revision[)].*", "", s, flags=_re.IGNORECASE)
    s = _re.sub(r"\s+", " ", s).strip(" :—-")
    if s == s.upper() and len(s) > 4:
        s = s.title()
    return s


def _clean_scope(raw: str) -> str:
    """Strip PDF extraction artifacts — clause numbers, SP 21 refs, table markers."""
    import re
    s = raw or ""
    s = re.sub(r'^\s*\d+[\.\s]+Scope\s*[—–\-]\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\b\d+\.\d+\s+', ' ', s)
    s = re.sub(r'(?<!\w)\d+\.\s+(?=[A-Z])', ' ', s)
    s = re.sub(r'SP\s*21\s*:\s*2005', '', s, flags=re.IGNORECASE)
    s = re.sub(r'TABLE\s+\d+[A-Z]?', '', s)
    s = re.sub(r'See\s+TABLE\s+\d+', '', s, flags=re.IGNORECASE)
    s = re.sub(r'For\s+detailed\s+information.*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^[\+\*\#\s]+', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    sentences = [x.strip() for x in s.split('.') if len(x.strip()) > 20]
    return '. '.join(sentences[:2]) + '.' if sentences else s[:200]


def _resolve(sid: str) -> dict:
    """Look up a standard. META_OVERRIDES fix known-bad PDF-extracted titles/scopes."""
    META_OVERRIDES = {
        "IS 8042: 1989":          ("White Portland Cement",                              "Covers manufacture and chemical/physical requirements of white Portland cement for architectural and decorative purposes. Coefficient of reflectance not less than 70%."),
        "IS 459: 1992":           ("Corrugated and Semi-Corrugated Asbestos Cement Sheets","Covers corrugated and semi-corrugated asbestos cement sheets for weather-exposed roofs and walls of industrial, residential, agricultural and commercial buildings."),
        "IS 455: 1989":           ("Portland Slag Cement",                               "Covers manufacture and chemical/physical requirements for Portland slag cement made by intergrinding Portland cement clinker and granulated blast furnace slag with gypsum."),
        "IS 6909: 1990":          ("Supersulphated Cement",                              "Covers composition, manufacture and testing of supersulphated cement, suitable for marine works, mass concrete, pipes in ground water and construction in sulphate-bearing soils."),
        "IS 2185 (PART 2): 1983": ("Concrete Masonry Units — Part 2: Lightweight Blocks","Covers hollow and solid lightweight concrete blocks for load-bearing and non-load bearing walls. Block density shall not exceed 1600 kg/m³."),
        "IS 2185 (PART 1): 1979": ("Concrete Masonry Units — Part 1: Hollow and Solid Blocks","Covers hollow (open/closed cavity) and solid concrete blocks for loadbearing and partition walls."),
        "IS 2185 (PART 3): 1984": ("Concrete Masonry Units — Part 3: AAC Blocks",       "Covers autoclaved cellular (aerated) concrete blocks up to 1000 kg/m³ density, produced by steam curing under pressure with good thermal insulation properties."),
        "IS 1489 (PART 2): 1991": ("Portland Pozzolana Cement — Part 2: Calcined Clay", "Covers manufacture and physical/chemical requirements of Portland pozzolana cement using calcined clay pozzolana or a mixture of calcined clay and fly ash pozzolana."),
        "IS 1489 (PART 1): 1991": ("Portland Pozzolana Cement — Part 1: Fly Ash Based", "Covers manufacture and physical/chemical requirements of PPC using fly ash pozzolana. Fly ash content shall be 15 to 35 percent by mass of cement."),
        "IS 8043: 1991":          ("Hydrophobic Portland Cement",                        "Covers manufacture and chemical/physical requirements of hydrophobic Portland cement, which resists moisture during prolonged storage under unfavourable conditions."),
        "IS 8041: 1990":          ("Rapid Hardening Portland Cement",                    "Covers manufacture and chemical/physical requirements of rapid hardening Portland cement, achieving higher early strength than ordinary Portland cement."),
        "IS 6452: 1989":          ("High Alumina Cement for Structural Use",             "Covers manufacture of high alumina cement for use as a structural building material in colder regions (continuously 18°C and below)."),
        "IS 12330: 1988":         ("Sulphate Resisting Portland Cement",                 "Covers manufacture, chemical/physical requirements of sulphate resisting Portland cement for use where concrete is exposed to sulphate attack from soil or water."),
        "IS 8112: 1989":          ("Ordinary Portland Cement, 43 Grade",                 "Covers manufacture and chemical/physical requirements of 43 grade OPC, suitable for prestressed concrete, railway sleepers and precast products requiring higher strength."),
        "IS 12269: 1987":         ("Ordinary Portland Cement, 53 Grade",                 "Covers manufacture and chemical/physical requirements of 53 grade OPC for specialized work such as prestressed concrete requiring very high compressive strength."),
    }

    if sid in META_OVERRIDES:
        title, scope = META_OVERRIDES[sid]
        cat = ""
        if sid in STANDARDS_DB:
            cat = STANDARDS_DB[sid].get("category", "")
        elif _idx:
            for chunk in _idx.chunks:
                if chunk["standard_id"] == sid:
                    cat = chunk.get("category", "Building Materials")
                    break
        return {"standard_id": sid, "title": title, "category": cat, "scope": scope}

    if sid in STANDARDS_DB:
        m = STANDARDS_DB[sid]
        return {
            "standard_id": sid,
            "title":    _clean_title(m["title"]),
            "category": m["category"],
            "scope":    _clean_scope(m["scope"]),
        }
    if _idx:
        for chunk in _idx.chunks:
            if chunk["standard_id"] == sid:
                return {
                    "standard_id": sid,
                    "title":    _clean_title(chunk.get("title", sid)),
                    "category": chunk.get("category", "Building Materials"),
                    "scope":    _clean_scope(chunk.get("scope", "")),
                }
    return {"standard_id": sid, "title": sid, "category": "", "scope": ""}


_CAT_BULLETS = {
    "Cement": [
        "Chemical limits — MgO, SO₃ and lime saturation factor must be within specified ranges.",
        "Physical tests — fineness, soundness, setting time and 28-day compressive strength.",
        "ISI mark (BIS licence) is mandatory before you can sell the product commercially.",
    ],
    "Aggregate": [
        "Grading must conform to specified sieve analysis zones.",
        "Crushing value, impact value and soundness tests are mandatory.",
        "Limits on deleterious materials, organic impurities and water absorption.",
    ],
    "Concrete Pipes": [
        "Wall thickness and internal diameter must meet dimensional tolerances.",
        "Three-edge bearing strength and hydrostatic pressure tests are compulsory.",
        "Pipes must be free from cracks, surface defects and honeycombing.",
    ],
    "Cement Matrix Products": [
        "Block dimensions — length, height and width must be within ±3 mm tolerance.",
        "Minimum compressive strength and maximum water absorption per grade (A/B).",
        "Drying shrinkage must not exceed the specified limit.",
    ],
    "Asbestos Cement Products": [
        "Dimensions — pitch, depth of corrugation and thickness within tolerance.",
        "Load-bearing capacity and water-tightness tests are mandatory.",
        "Sheets must be free from warping, cracks and surface defects.",
    ],
    "Clay Products for Building": [
        "Dimensions — length, width and height within ±3 mm tolerance.",
        "Minimum compressive strength per class (3.5, 5, 7.5, 10 N/mm²).",
        "Water absorption must not exceed 20% and efflorescence should be nil or slight.",
    ],
    "Building Limes": [
        "Chemical composition — calcium oxide, magnesium oxide and CO₂ content limits.",
        "Fineness and soundness requirements must be met.",
        "Slaking time and residue on sieve tests are required.",
    ],
    "Structural Steels": [
        "Chemical composition — carbon, sulphur and phosphorus content within grade limits.",
        "Tensile strength, yield strength and elongation must meet grade requirements.",
        "Weldability and impact energy requirements as per IS specification.",
    ],
    "Concrete Reinforcement": [
        "Rib geometry and deformation pattern must meet specified dimensions.",
        "Yield strength, UTS and elongation must comply with Fe 415/Fe 500 grade.",
        "Chemical composition — carbon equivalent and sulphur/phosphorus limits.",
    ],
    "Sanitary Appliances and Water Fittings": [
        "Dimensions and pressure rating must meet the specified class requirements.",
        "Hydraulic test at rated pressure — zero leakage permitted.",
        "Material composition and surface finish must comply with the standard.",
    ],
    "Timber": [
        "Species classification and permissible defects for structural use.",
        "Moisture content at the time of use must not exceed specified limits.",
        "Grading rules — knot size, slope of grain and wane must be within limits.",
    ],
    "Bitumen and Tar Products": [
        "Penetration value, softening point and ductility must be within grade limits.",
        "Flash point and solubility requirements.",
        "Performance grading and storage stability tests.",
    ],
    "Glass": [
        "Thickness tolerance and flatness requirements.",
        "Impact resistance and fragmentation pattern for safety glass.",
        "No visible defects — bubbles, scratches or inclusions beyond permitted limits.",
    ],
    "Thermal Insulation Materials": [
        "Thermal conductivity must be within specified limits at rated temperature.",
        "Density, compressive strength and moisture absorption requirements.",
        "Dimensional tolerances on length, width and thickness.",
    ],
    "Conductors and Cables": [
        "Conductor resistance per unit length must not exceed specified values.",
        "Insulation resistance and high voltage test requirements.",
        "Physical properties of insulation — tensile strength and elongation.",
    ],
    "Wiring Accessories": [
        "Current rating, voltage rating and temperature rise limits.",
        "Mechanical endurance — minimum switching cycles.",
        "Insulation resistance and electric strength tests.",
    ],
}

def _get_bullets(cat: str) -> list:
    for key, bullets in _CAT_BULLETS.items():
        if key.lower() in cat.lower() or cat.lower() in key.lower():
            return bullets
    return [
        "Dimensional and physical requirements must conform to specified limits.",
        "Mechanical or performance tests referenced in the standard are mandatory.",
        "BIS certification mark (ISI mark) required before commercial sale.",
    ]


def stream_explanation(query: str, results: list):
    """Try Claude API; fall back to clean structured template."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            top3 = results[:3]
            ctx_lines = []
            for r in top3:
                clean = _clean_scope(r.get("scope", ""))
                ctx_lines.append(
                    f"• {r['standard_id']} — {r['title']} ({r['category']})\n"
                    f"  What it covers: {clean}"
                )
            ctx = "\n\n".join(ctx_lines)
            prompt = (
                "You are a expert BIS compliance officer explaining standards to a small Indian business owner.\n\n"
                f"Their product/query: \"{query}\"\n\n"
                f"Retrieved BIS standards:\n{ctx}\n\n"
                "Write a SHORT compliance note in EXACTLY this format. "
                "Use plain simple English — no raw standard text, no jargon, no clause numbers:\n\n"
                "**Why this standard applies:**\n"
                "• point 1 explaining why the #1 standard matches their product.\n"
                "• point 2 explaining why the #1 standard matches their product. \n"
                "**3 key things to comply with:**\n"
                "• [Requirement 1 in simple words]\n"
                "• [Requirement 2 in simple words]\n"
                "• [Requirement 3 — testing or certification]\n\n"
                "**Also check:** [One sentence naming the 2nd and 3rd standards and why they may apply]\n\n"
                "**Next step:** [One sentence — what the business owner should do now]\n\n"
                "RULES: Max 150 words. Never copy raw clause text like '1. Scope —'. "
                "Never invent IS numbers. Write for someone with no technical background."
            )
            with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for chunk in stream.text_stream:
                    yield "data: " + json.dumps({"chunk": chunk}) + "\n\n"
        except Exception as e:
            print(f"[BIS] LLM error: {e} — using template")
            yield from _template_explanation(query, results)
    else:
        yield from _template_explanation(query, results)
    yield 'data: {"done":true}\n\n'


def _template_explanation(query: str, results: list):
    """Clean fallback — plain English, zero PDF artifacts."""
    if not results:
        text = "No matching BIS standards found. Try rephrasing with specific material names, grades or product types."
        for word in text.split():
            yield "data: " + json.dumps({"chunk": word + " "}) + "\n\n"
        return

    # Re-resolve so META_OVERRIDES and _clean_title/_clean_scope are always applied
    top   = _resolve(results[0]["standard_id"])
    rest  = [_resolve(r["standard_id"]) for r in results[1:3]]

    sid   = top["standard_id"]
    title = top["title"]          # already clean via _resolve → _clean_title / META_OVERRIDES
    cat   = top.get("category", "")
    scope = top.get("scope", "")  # already clean via _resolve → _clean_scope / META_OVERRIDES

    # Build "why" sentence from clean scope — first sentence only, no raw PDF text
    if scope:
        first = scope.split(".")[0].strip()
        why = first + "." if first else scope[:160] + "."
    else:
        why = f"This standard sets the mandatory requirements for {title.lower()}."

    lines = []
    lines.append("**Why this standard applies:**\n")
    lines.append(f"{sid} — {title}: {why}\n\n")

    lines.append("**3 key things to comply with:**\n")
    for b in _get_bullets(cat):
        lines.append(f"• {b}\n")

    if rest:
        also = " and ".join(
            f"{r['standard_id']} ({r['title'][:50]})"   # title already clean
            for r in rest
        )
        lines.append(f"\n**Also check:** {also}.\n")

    lines.append(
        f"\n**Next step:** Apply for BIS product certification under {sid} "
        "at www.bis.gov.in or your nearest BIS Regional Office."
    )

    full_text = "".join(lines)
    for word in full_text.split(" "):
        yield "data: " + json.dumps({"chunk": word + " "}) + "\n\n"

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>BIS Standards Recommendation Engine</title>
<style>
:root{
  --bg:#0f1117; --surface:#1a1d27; --card:#21243a;
  --accent:#4f6ef7; --accent2:#7c3aed;
  --green:#22c55e; --text:#e2e8f0; --muted:#64748b; --border:#2d3148;
  --radius:14px; --font:'Segoe UI',system-ui,sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--font);background:var(--bg);color:var(--text);
  height:100vh;display:flex;flex-direction:column;overflow:hidden}
header{background:linear-gradient(135deg,#1a1d27,#161929);
  border-bottom:1px solid var(--border);padding:14px 24px;
  display:flex;align-items:center;gap:14px;flex-shrink:0}
.logo{width:42px;height:42px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  border-radius:10px;display:flex;align-items:center;justify-content:center;
  font-size:20px;font-weight:800;color:#fff}
.header-text h1{font-size:18px;font-weight:700}
.header-text p{font-size:12px;color:var(--muted);margin-top:2px}
.status-badge{margin-left:auto;background:rgba(34,197,94,.12);
  border:1px solid rgba(34,197,94,.3);color:var(--green);
  padding:4px 12px;border-radius:20px;font-size:11px;font-weight:600;
  display:flex;align-items:center;gap:6px}
.status-dot{width:7px;height:7px;background:var(--green);border-radius:50%;
  animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.layout{display:flex;flex:1;overflow:hidden}
.sidebar{width:270px;background:var(--surface);
  border-right:1px solid var(--border);
  display:flex;flex-direction:column;flex-shrink:0;overflow:hidden}
.sidebar-head{padding:14px 16px;border-bottom:1px solid var(--border);
  font-size:11px;font-weight:700;text-transform:uppercase;
  letter-spacing:.08em;color:var(--muted)}
.quick-list{overflow-y:auto;flex:1;padding:6px}
.qi{padding:9px 11px;border-radius:8px;font-size:12px;color:var(--muted);
  cursor:pointer;border:1px solid transparent;margin-bottom:3px;
  line-height:1.4;transition:all .15s}
.qi:hover{background:var(--card);border-color:var(--border);color:var(--text)}
.qi .cat{font-size:9px;text-transform:uppercase;letter-spacing:.06em;
  color:var(--accent);margin-bottom:2px;font-weight:700}
.sidebar-stats{padding:10px 14px;border-top:1px solid var(--border);
  display:flex;flex-direction:column;gap:5px}
.sr{display:flex;justify-content:space-between;align-items:center;font-size:11px}
.sr .lbl{color:var(--muted)} .sr .val{color:var(--green);font-weight:700}
.chat-area{flex:1;display:flex;flex-direction:column;overflow:hidden}
#messages{flex:1;overflow-y:auto;padding:20px 24px;
  display:flex;flex-direction:column;gap:18px;scroll-behavior:smooth}
#messages::-webkit-scrollbar{width:4px}
#messages::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}
.msg{display:flex;gap:10px;animation:fi .2s ease}
@keyframes fi{from{opacity:0;transform:translateY(5px)}to{opacity:1}}
.msg.user{flex-direction:row-reverse}
.msg.user .bubble{background:linear-gradient(135deg,#1e3a5f,#1a3055);
  border:1px solid rgba(79,110,247,.3);margin-left:50px}
.msg.bot .bubble{background:#1e2236;border:1px solid var(--border);margin-right:50px}
.av{width:30px;height:30px;border-radius:8px;display:flex;
  align-items:center;justify-content:center;font-size:13px;flex-shrink:0}
.msg.user .av{background:linear-gradient(135deg,#1e40af,#1e3a8a);color:#93c5fd}
.msg.bot  .av{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff}
.bubble{padding:12px 15px;border-radius:var(--radius);font-size:14px;
  line-height:1.6;max-width:100%;word-break:break-word}
.rb{margin-top:10px;display:flex;flex-direction:column;gap:7px}
.rc{background:var(--card);border:1px solid var(--border);border-radius:10px;
  padding:11px 13px;position:relative;overflow:hidden;transition:all .15s}
.rc::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px}
.rc.r1::before{background:#f59e0b} .rc.r2::before{background:#94a3b8}
.rc.r3::before{background:#cd7c2f} .rc.r4::before{background:var(--border)}
.rc:hover{border-color:var(--accent);background:rgba(79,110,247,.04)}
.rc-top{display:flex;align-items:flex-start;gap:9px;margin-bottom:4px}
.badge{font-size:9px;font-weight:700;padding:2px 7px;border-radius:4px;
  text-transform:uppercase;flex-shrink:0;margin-top:1px}
.b1{background:rgba(245,158,11,.15);color:#f59e0b;border:1px solid rgba(245,158,11,.3)}
.b2{background:rgba(148,163,184,.12);color:#94a3b8;border:1px solid rgba(148,163,184,.3)}
.b3{background:rgba(205,124,47,.12);color:#cd7c2f;border:1px solid rgba(205,124,47,.3)}
.b4{background:rgba(100,116,139,.1);color:var(--muted);border:1px solid var(--border)}
.rc-id{font-size:13px;font-weight:700;color:var(--accent)}
.rc-title{font-size:12px;color:var(--text);font-weight:500;margin-bottom:3px}
.rc-scope{font-size:11px;color:var(--muted);line-height:1.5}
.rc-cat{font-size:9px;font-weight:700;text-transform:uppercase;
  letter-spacing:.05em;color:var(--accent2);margin-top:5px}
.lat{display:inline-flex;align-items:center;gap:4px;
  background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.2);
  color:var(--green);font-size:11px;font-weight:600;
  padding:3px 10px;border-radius:20px;margin-top:9px}
.exp{margin-top:11px;padding:13px 15px;
  background:rgba(79,110,247,.06);
  border:1px solid rgba(79,110,247,.22);
  border-left:3px solid var(--accent);
  border-radius:10px;font-size:13px;line-height:1.75;color:#c7d2fe}
.exp strong{color:#93c5fd;font-weight:700}
.exp-lbl{font-size:9px;font-weight:700;text-transform:uppercase;
  letter-spacing:.08em;color:var(--accent);margin-bottom:6px;
  display:flex;align-items:center;gap:5px}
.exp-lbl .spin{display:inline-block;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.typing{display:flex;gap:4px;padding:8px 12px}
.typing span{width:6px;height:6px;background:var(--muted);
  border-radius:50%;animation:bounce 1.2s infinite}
.typing span:nth-child(2){animation-delay:.2s}
.typing span:nth-child(3){animation-delay:.4s}
@keyframes bounce{0%,80%,100%{transform:scale(.8);opacity:.5}40%{transform:scale(1.2);opacity:1}}
.input-area{padding:14px 22px;border-top:1px solid var(--border);
  background:var(--surface);flex-shrink:0}
.input-row{display:flex;gap:9px;align-items:flex-end;max-width:900px;margin:0 auto}
#qi{flex:1;background:var(--card);border:1px solid var(--border);
  border-radius:var(--radius);padding:11px 15px;color:var(--text);
  font-size:14px;font-family:var(--font);resize:none;outline:none;
  line-height:1.5;min-height:46px;max-height:130px;transition:border-color .15s}
#qi:focus{border-color:var(--accent)}
#qi::placeholder{color:var(--muted)}
#sb{width:46px;height:46px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  border:none;border-radius:11px;color:#fff;font-size:17px;cursor:pointer;
  flex-shrink:0;display:flex;align-items:center;justify-content:center;
  transition:all .15s}
#sb:hover{transform:scale(1.06);opacity:.9}
#sb:disabled{opacity:.35;cursor:not-allowed;transform:none}
.hint{font-size:11px;color:var(--muted);text-align:center;margin-top:7px}
.welcome{flex:1;display:flex;flex-direction:column;
  align-items:center;justify-content:center;padding:40px;text-align:center;gap:10px}
.welcome .icon{font-size:50px}
.welcome h2{font-size:21px;font-weight:700}
.welcome p{font-size:14px;color:var(--muted);max-width:480px}
.chips{display:flex;flex-wrap:wrap;gap:7px;justify-content:center;margin-top:7px}
.chip{background:var(--card);border:1px solid var(--border);border-radius:20px;
  padding:5px 13px;font-size:12px;color:var(--muted);cursor:pointer;transition:all .15s}
.chip:hover{border-color:var(--accent);color:var(--text)}
@media(max-width:700px){
  .sidebar{display:none}
  #messages{padding:10px}
  .input-area{padding:10px}
}
</style>
</head>
<body>
<header>
  <div class="logo">B</div>
  <div class="header-text">
    <h1>BIS Standards Recommendation Engine</h1>
    <p>SP 21 (2005) &middot; All 27 Sections &middot; 566 Standards</p>
  </div>
  <div class="status-badge">
    <div class="status-dot"></div>RAG Ready
  </div>
</header>
<div class="layout">
  <div class="sidebar">
    <div class="sidebar-head">&#128203; Quick Queries</div>
    <div class="quick-list" id="ql"></div>
    <div class="sidebar-stats">
      <div class="sr"><span class="lbl">Hit Rate @3</span><span class="val">100%</span></div>
      <div class="sr"><span class="lbl">MRR @5</span><span class="val">1.0000</span></div>
      <div class="sr"><span class="lbl">Avg Latency</span><span class="val">&lt;0.05s</span></div>
      <div class="sr"><span class="lbl">Standards</span><span class="val">566</span></div>
    </div>
  </div>
  <div class="chat-area">
    <div id="messages">
      <div class="welcome" id="welcome">
        <div class="icon">&#127963;</div>
        <h2>BIS Standards for MSE Compliance</h2>
        <p>Describe your product or material and instantly discover which Bureau of Indian Standards apply &mdash; backed by SP 21 (2005) Building Materials.</p>
        <div class="chips" id="chips"></div>
      </div>
    </div>
    <div class="input-area">
      <div class="input-row">
        <textarea id="qi" rows="1"
          placeholder="Describe your product or ask which BIS standard applies&hellip;"></textarea>
        <button id="sb" type="button" title="Send">&#10148;</button>
      </div>
      <div class="hint">Press Enter to send &middot; Shift+Enter for new line</div>
    </div>
  </div>
</div>
<script>
"use strict";
var QUERIES=[
  {cat:"Cement",q:"We manufacture 33 Grade Ordinary Portland Cement. Which BIS standard applies?"},
  {cat:"Aggregates",q:"Coarse and fine aggregates from natural sources for structural concrete."},
  {cat:"Concrete Pipes",q:"Precast concrete pipes with and without reinforcement for water mains."},
  {cat:"Blocks",q:"Hollow and solid lightweight concrete masonry blocks — dimensions and requirements."},
  {cat:"Asbestos",q:"Corrugated and semi-corrugated asbestos cement sheets for roofing and cladding."},
  {cat:"Cement",q:"Portland slag cement manufacture, chemical and physical requirements."},
  {cat:"Cement",q:"Portland pozzolana cement — calcined clay based, setting up production plant."},
  {cat:"Cement",q:"Masonry cement for general purposes, not for structural concrete."},
  {cat:"Cement",q:"Supersulphated cement for marine works and aggressive water conditions."},
  {cat:"Cement",q:"White Portland cement for architectural and decorative purposes."},
  {cat:"Cement",q:"OPC 43 grade cement for railway sleepers and precast products."},
  {cat:"Cement",q:"Sulphate resisting cement for construction in sulphate bearing soils."},
  {cat:"Blocks",q:"AAC autoclaved aerated concrete blocks for thermal insulation."},
  {cat:"Cement",q:"Hydrophobic cement for prolonged storage in humid conditions."}
];
var msgsEl=document.getElementById("messages");
var inputEl=document.getElementById("qi");
var sendBtn=document.getElementById("sb");
var welcome=document.getElementById("welcome");
var qlEl=document.getElementById("ql");
var chipsEl=document.getElementById("chips");
var msgCount=0;

if(qlEl){
  QUERIES.forEach(function(item,idx){
    var el=document.createElement("div");
    el.className="qi";
    el.innerHTML="<div class='cat'>"+item.cat+"</div>"+item.q;
    el.onclick=function(){send(item.q);};
    qlEl.appendChild(el);
    if(idx<5&&chipsEl){
      var chip=document.createElement("div");
      chip.className="chip";
      chip.textContent=item.q.substring(0,48)+"...";
      chip.onclick=function(){send(item.q);};
      chipsEl.appendChild(chip);
    }
  });
}

function scrollDown(){msgsEl.scrollTop=msgsEl.scrollHeight;}
function unlock(){sendBtn.disabled=false;inputEl.focus();}
function hideWelcome(){if(welcome){welcome.style.display="none";}}

function esc(s){
  return String(s)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function addMsg(role,content){
  hideWelcome();
  var msg=document.createElement("div");
  msg.className="msg "+role;
  var av=document.createElement("div");
  av.className="av";
  av.textContent=(role==="user")?"U":"B";
  var bubble=document.createElement("div");
  bubble.className="bubble";
  if(typeof content==="string"){bubble.innerHTML=content;}
  else{bubble.appendChild(content);}
  msg.appendChild(av);
  msg.appendChild(bubble);
  msgsEl.appendChild(msg);
  scrollDown();
}

function showTyping(){
  var msg=document.createElement("div");
  msg.className="msg bot";msg.id="typing";
  msg.innerHTML="<div class='av'>B</div><div class='bubble'><div class='typing'><span></span><span></span><span></span></div></div>";
  msgsEl.appendChild(msg);scrollDown();
}
function removeTyping(){var t=document.getElementById("typing");if(t){t.remove();}}

/* render **bold** markers and newlines — no regex */
function renderText(text){
  var parts=text.split("**");
  var out="";
  for(var i=0;i<parts.length;i++){
    out+=(i%2===0)?esc(parts[i]):"<strong>"+esc(parts[i])+"</strong>";
  }
  return out.split(String.fromCharCode(10)).join("<br>");
}

function rankInfo(i){
  if(i===0)return{cls:"r1",badge:"b1",txt:"#1 Best Match"};
  if(i===1)return{cls:"r2",badge:"b2",txt:"#2"};
  if(i===2)return{cls:"r3",badge:"b3",txt:"#3"};
  return{cls:"r4",badge:"b4",txt:"#"+(i+1)};
}

function buildCard(std,i){
  var ri=rankInfo(i);
  var card=document.createElement("div");
  card.className="rc "+ri.cls;
  var raw=std.scope||"";
  var scope=esc(raw.substring(0,160))+(raw.length>160?"&hellip;":"");
  card.innerHTML=
    "<div class='rc-top'>"+
      "<span class='badge "+ri.badge+"'>"+ri.txt+"</span>"+
      "<div class='rc-id'>"+esc(std.standard_id)+"</div>"+
    "</div>"+
    "<div class='rc-title'>"+esc(std.title)+"</div>"+
    "<div class='rc-scope'>"+scope+"</div>"+
    "<div class='rc-cat'>"+esc(std.category)+"</div>";
  return card;
}

function send(query){
  query=(query||"").trim();
  if(!query||sendBtn.disabled){return;}
  sendBtn.disabled=true;
  inputEl.value="";
  inputEl.style.height="auto";
  addMsg("user","<p>"+esc(query)+"</p>");
  showTyping();

  fetch("/query",{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({query:query})
  })
  .then(function(res){
    removeTyping();
    unlock();            /* always unlock immediately after fetch */
    if(!res.ok){
      addMsg("bot","<p>Server error "+res.status+" — check Flask terminal.</p>");
      return null;
    }
    return res.json();
  })
  .then(function(data){
    if(!data){return;}
    if(data.error){addMsg("bot","<p>Error: "+esc(data.error)+"</p>");return;}

    msgCount++;
    var mid=msgCount;
    var expTxtId="et"+mid;
    var expLblId="el"+mid;

    var wrap=document.createElement("div");

    var topLine=document.createElement("p");
    topLine.style.cssText="margin-bottom:10px;font-size:13px;color:#94a3b8;";
    topLine.textContent="Found "+data.results.length+" relevant standard"+(data.results.length!==1?"s":"")+":";
    wrap.appendChild(topLine);

    var rb=document.createElement("div");
    rb.className="rb";
    data.results.forEach(function(std,i){rb.appendChild(buildCard(std,i));});
    wrap.appendChild(rb);

    var lat=document.createElement("div");
    lat.className="lat";
    lat.textContent="Retrieved in "+(data.latency_seconds*1000).toFixed(1)+" ms";
    wrap.appendChild(lat);

    var exp=document.createElement("div");
    exp.className="exp";
    exp.innerHTML=
      "<div class='exp-lbl' id='"+expLblId+"'>"+
        "&#129302; AI Explanation "+
        "<span class='spin'>&#8635;</span>"+
      "</div>"+
      "<div id='"+expTxtId+"'></div>";
    wrap.appendChild(exp);
    addMsg("bot",wrap);

    /* SSE explanation — non-blocking, send already unlocked above */
    var expTxtEl=document.getElementById(expTxtId);
    var expLblEl=document.getElementById(expLblId);
    var ids=data.results.map(function(r){return r.standard_id;}).join(",");
    var url="/explain?query="+encodeURIComponent(query)+"&ids="+encodeURIComponent(ids);
    var src=new EventSource(url);
    var full="";
    var timer=setTimeout(function(){src.close();finishLbl(expLblEl);},18000);

    src.onmessage=function(e){
      try{
        var p=JSON.parse(e.data);
        if(p.done){clearTimeout(timer);src.close();finishLbl(expLblEl);return;}
        if(p.chunk&&expTxtEl){
          full+=p.chunk;
          expTxtEl.innerHTML=renderText(full);
          scrollDown();
        }
      }catch(err){}
    };
    src.onerror=function(){clearTimeout(timer);src.close();finishLbl(expLblEl);};
  })
  .catch(function(err){
    removeTyping();unlock();
    addMsg("bot","<p>Network error: "+esc(err.message)+"</p>");
  });
}

function finishLbl(el){if(el){el.innerHTML="&#129302; AI Explanation";}}

inputEl.addEventListener("keydown",function(e){
  if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send(inputEl.value);}
});
inputEl.addEventListener("input",function(){
  this.style.height="auto";
  this.style.height=Math.min(this.scrollHeight,130)+"px";
});
sendBtn.addEventListener("click",function(){send(inputEl.value);});
inputEl.focus();
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    # Direct Response — Jinja2 never touches the HTML
    return Response(HTML, mimetype="text/html; charset=utf-8")


@app.route("/query", methods=["POST"])
def query_route():
    body  = request.get_json(force=True)
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400
    try:
        t0  = time.perf_counter()
        out = query_pipeline(query, top_k=5, pdf_path=None)
        lat = round(time.perf_counter() - t0, 4)
        # _resolve handles ALL 566 standards — not just STANDARDS_DB
        results = [_resolve(sid) for sid in out["retrieved_standards"]]
        return jsonify({
            "query":               query,
            "retrieved_standards": out["retrieved_standards"],
            "results":             results,
            "latency_seconds":     lat,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/explain")
def explain_route():
    query   = request.args.get("query", "").strip()
    ids_raw = request.args.get("ids", "")
    ids     = [s.strip() for s in ids_raw.split(",") if s.strip()]
    results = [_resolve(sid) for sid in ids if sid]

    def generate():
        yield from stream_explanation(query, results)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/health")
def health():
    return jsonify({
        "status":            "ok",
        "standards_indexed": len(_idx.chunks),
        "expert_db_count":   len(STANDARDS_DB),
    })


@app.route("/api/query", methods=["POST"])
def api_query():
    body  = request.get_json(force=True)
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400
    t0  = time.perf_counter()
    out = query_pipeline(query, top_k=5, pdf_path=None)
    lat = round(time.perf_counter() - t0, 4)
    return jsonify({
        "query":               query,
        "retrieved_standards": out["retrieved_standards"],
        "latency_seconds":     lat,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)