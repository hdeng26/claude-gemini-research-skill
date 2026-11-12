The full text of one paper is above, inside <paper id="P1">. Math appears as LaTeX between $ signs.

Extract what a researcher needs to judge this paper. Cover these dimensions:
- method: the core update rule / algorithm and its key assumptions
- optimization: does it use backpropagation, local gradients, closed-form / least-squares / pseudoinverse solves, or something else
- setup: datasets, architectures (depth, width), training budget, baselines
- evidence: which experiments, tables or theorems support the authors' main claims, and what they show
- limitation: limitations the authors state; and gaps you notice (mark those kind="inferred")
{{focus_block}}
Rules:
- Every item needs `quote`: a contiguous passage copied character-for-character from the paper text, 8-40 words. Do not paraphrase, merge sentences, or fix typos. The quote will be machine-checked against the text; items whose quote is not found are discarded.
- Prefer prose sentences over equations for quotes; for an equation, quote the sentence that introduces or defines it.
- `location`: the nearest section heading or label as printed (e.g. "Section 3.2", "Eq. (4)", "Table 2", "Appendix B").
- `kind`: "stated" if the paper says it; "inferred" if it is your conclusion from what the paper says (still quote the passage it rests on).
- If the paper does not say something you were asked about, put it in `not_stated` instead of guessing.
- Aim for 8-20 items. Precision matters more than coverage.
