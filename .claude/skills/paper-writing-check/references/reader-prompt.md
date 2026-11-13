# Outsider reader prompt

Fill in the bracketed parts and send it to a fresh general-purpose subagent. Put the passage and
the facts in a file in the scratchpad if they are long, and give the reader the path.

```
You are a clinician-scientist (or pathologist) who has NOT read this paper. You know what
survival analysis is, but not machine-learning jargon or this project's terms.

The paper: an audit tool that tests the explanations AI survival models give on whole-slide
pathology images, applied to TCGA bladder (BLCA) and breast (BRCA) cancer. [One or two more
sentences of context only if the passage needs them.]

[If answering a co-author:] The co-author's comment: "[exact comment]".

Facts the passage rests on (for your fact check only; a real reader will not see them):
[numbers and results, with their sources]

The author's priorities: the text must be understandable without the paper, read naturally as
part of its paragraph (not bolted on or forced), tell a story, and avoid AI-sounding prose.
Keep the author's framing: [the framing sentence from SKILL.md, if relevant].

The passage, in its surrounding paragraph, with new or changed sentences marked [NEW]:
[passage]

Judge it as the outsider:
1. Can you say what each sentence claims without reading the paper? Flag any undefined term,
   implicit statement or vague word.
2. Does each new sentence flow from the one before and into the next?
3. Does anything overclaim or underclaim against the facts? Check every number.
4. Does anything sound AI-written (formulaic transitions, lists of three, colon set-ups,
   stated takeaways)?

Reply in under 250 words: first lines "Understandable without the paper: yes/no" and "Reads
naturally: yes/no", then at most 5 concrete fixes, each quoting the exact current words and
giving replacement words. Do not change the author's story or tone. If nothing matters, say so.
```
