Above, inside <report>, is a research report written by an AI deep-research tool. We will check it against its sources.

Extract two things.

1. `references`: every entry in the report's source / reference / "works cited" list.
   - `ref_id`: the number or label exactly as printed (e.g. "1", "12").
   - `title`: the title as printed.
   - `url`: the URL exactly as printed in the report, or null if none.
   - `report_quote`: the reference entry copied character-for-character from the report.

2. `claims`: up to {{max_claims}} factual claims the report makes about research, papers, methods or results.
   - Each claim must be atomic (one checkable fact).
   - Prioritise: specific numbers and results; "first" / "novel" / "only" claims; descriptions of what a specific method does or uses; comparisons between methods. Skip generic background statements.
   - `claim`: the claim restated in one plain sentence.
   - `report_quote`: the sentence from the report that makes the claim, copied character-for-character (8-60 words).
   - `ref_ids`: the reference ids cited for that sentence (inline markers like [3] or superscripts). Empty list if the sentence cites nothing.
   - `kind`: number | novelty | method | comparison | other.

Rules: copy quotes and URLs exactly; they are machine-checked against the report. Do not add claims the report does not make.
