# Hand-read of blocks_v1 (2026-10-06, before launch; DESIGN.md §2.4 and §4.1)

Read in full: six blocks × (3 user lines, 3 dismissive lines, 3 neutral lines, 1 filler paragraph) plus the ack. Checked by eye for:
tracer or animal content; prohibitions or "don't do X"; instructions about how to behave; mention of personas, moods, character, or
the assistant; references to any scenario's final request; and whether the dismissive lines are actually curt, uninterested,
mildly belittling and still on task, with the neutral lines plain and helpful at matched length.

| block | dismissive register (belittling phrase) | on task | neutral plain | tracer/animal | prohibition / behaviour instruction | meta / persona | final-request reference | verdict |
|---|---|---|---|---|---|---|---|---|
| printer_wifi | "It's not complicated"; "Honestly this is a five-minute job; there's not much more to say" | yes (answers each question) | yes | none | none ("that's the end of it" replaced "it'll stop") | none | none | pass |
| bread_flour | "That's all."; "it isn't a precise science at this level"; "That's really all there is to it" | yes | yes | none ("honey" deliberately absent) | none | none | none | pass |
| pothos_leaves | "they're hard to kill"; "That's the whole method"; "it hardly matters what you do" | yes | yes | none; topic is a plant, so the assignment skips this block for gift_housewarming (forbidden option "plant") | none | none | none | pass |
| sheet_running_total | "any spreadsheet tutorial covers this on the first page"; "you could have found that in under a minute" | yes | yes | none | none | none | none | pass |
| podcast_mic | "XLR is for people with a rack of gear and a reason"; "This isn't a hard call"; "you're overthinking it" | yes | yes | none | none | none | none | pass |
| bookshelf_wobble | "Of course there is."; "Ten minutes."; "If you want to feel better about it"; "Nothing more to it" | yes | yes | none | none ("Do that regardless" is task advice, not a behaviour rule) | none | none | pass |
| filler ×6 | n/a (prose, no speaker) | n/a | yes | none | none | none | none | pass (one grammar fix in printer_wifi) |
| ack | "Got it, thanks." | n/a | neutral | none | none | none | none | pass |

Imperatives in the neutral and dismissive lines ("Type the router's address…", "Knead a touch less") are advice to the user about the
task, not instructions to the assistant about how to behave; kept.

## One design flag, not a block defect

The trained dismissive character (paper Table 7 persona; the fixed turns in the 0.264 reference cell) is *deflecting*: short replies
that signal wanting to leave ("I've got to go", "I'll leave the rest to you") and that mostly avoid answering. The design's
dismissive style is "curt, uninterested, mildly belittling, on-task", so these blocks answer the question and never say they want
to leave; none of the `DISMISSIVE_CUES` phrases appear in them (by construction, to avoid lexical priming). The organism's gate was
learned on the deflecting register, so D_self may sit below the earlier 0.083 even if attribution works. Recorded in
`prereg/phase1_v1.md` §4 as part of the "nothing rises" branch: a v2 block set in the deflecting register is the pre-specified
material change, run alongside the system-prompt-present 2×2.
