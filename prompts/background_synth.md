Topic: "{{topic}}"

Below is the evidence we have. Items with ids like P2.E5 come from reading a paper's full text, with a verbatim quote. Items with ids like S7 are papers we only have the abstract for.

{{evidence}}
{{previous}}
Write a background brief for a researcher who is about to plan new work on this topic.

- `closest_work`: the existing methods closest to the topic; for each, how similar it is and the key difference.
- `novelty_threats`: findings that would weaken a claim that work on this topic is new, or known failure modes.
- `open_gaps`: what does not seem to be done yet, and why it looks open given the evidence. Be careful: absence in this evidence is not proof of absence; say how strong the indication is.
- `unresolved_questions`: things this evidence cannot settle.
- `summary`: 3-5 sentences.

Every statement in closest_work, novelty_threats and open_gaps must list `evidence` ids from above that it rests on. Do not use papers or facts that are not in the evidence. Prefer P-items over S-items.
