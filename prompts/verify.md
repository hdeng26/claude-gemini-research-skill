The full texts of the source papers are above, each inside <paper id="...">. Math appears as LaTeX between $ signs.

Claim to check:
"{{claim}}"

Your job is to test this claim, not to agree with it. Look for evidence on both sides:
- passages that directly support the claim
- passages that contradict it, narrow it (e.g. only for shallow networks, only on MNIST), or show it is overstated
- passages that are relevant context (definitions, scope) without taking a side

Then give a verdict:
- "supported": the sources directly state or demonstrate the claim as written
- "contradicted": the sources directly state or demonstrate the opposite, or show the claim is false as written
- "conflicting": the sources contain direct evidence on both sides
- "insufficient": the sources do not settle it (including: they only partially address it)

Rules:
- Every evidence item needs `quote`: a contiguous passage copied character-for-character from that paper's text, 8-40 words, and `paper_id`. Quotes are machine-checked; unmatched quotes are discarded, and a verdict without matching quotes becomes "insufficient".
- `location`: nearest section heading or label as printed.
- If the claim is broader than what the sources show, the verdict is not "supported"; explain the gap in `caveats`.
- Absence of evidence is "insufficient", never "contradicted".
