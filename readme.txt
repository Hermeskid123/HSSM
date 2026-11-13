HSSM hand project
=================

Start here
1. Extract the whole archive and open the hssm-hand folder in Codex.
2. If you use the Codex CLI, run these commands from the extracted folder's parent.
       cd hssm-hand
       git init
       codex
3. Paste the first task below. The project begins with specifications and the
   source paper. The first task creates the Python environment instructions,
   implementation, tests, and example outputs.

First task to paste into Codex
    Read AGENTS.md and milestones/01-single-finger/readme.txt. Implement
    milestone 1 end to end. Run it against held-out examples, inspect the
    predicted contour overlays, and give me the exact commands to reproduce
    the results on Linux. Report both candidate scores for one example of
    each pose. Stop after milestone 1.

Project map
    AGENTS.md                         Instructions for every coding session
    references/wang_pami2008.pdf      Source paper, especially Fig. 3
    milestones/01-single-finger/      Fixed-view, two-path diagnostic
    milestones/02-hand-model/         Hand structure and localization
    milestones/03-flexion/            Controlled research extension

For a later milestone, start a session at the same project root. Ask Codex
to read that milestone's readme.txt, inspect the previous implementation,
and implement and evaluate the new experiment. Each milestone brief lists
what evidence to report. The code and tests should be shared across stages.

Codex project instructions guide
https://developers.openai.com/codex/guides/agents-md/
