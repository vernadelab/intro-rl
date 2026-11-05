# Introduction to Reinforcement Learning

HTML slides by Claire Vernade for the Reinforcement Learning course in the M.Sc. AI and Robotics programme at UTN, Winter 2026/27.
They work **offline**: KaTeX is bundled in `assets/katex/`, so no internet is needed in the lecture hall.

**Course website:** [vernadelab.github.io/intro-rl](https://vernadelab.github.io/intro-rl/)

| Lecture | Topic | Material |
|---|---|---|
| 2 | Dynamic Programming | [Interactive slides](https://vernadelab.github.io/intro-rl/lecture-02-dynamic-programming/) · [PDF](lecture-02-dynamic-programming/Dynamic%20Programming.pdf) |
| 3 | Model-Free Prediction | [Interactive slides](https://vernadelab.github.io/intro-rl/lecture-03-model-free-prediction/) |

```
RL-UTN/
├── assets/
│   ├── theme.css      colours, fonts, layouts, boxes (edit here to restyle every lecture)
│   ├── deck.js        slide engine: keys, fragments, notes, presenter view, print
│   ├── macros.js      shared LaTeX notation (\Vpi, \Qstar, \Tpi, \E, …), agreed with Chris
│   ├── maze.js        the two-exit maze: model, DP solvers, interactive heat-map widget
│   ├── learn.js       model-free prediction on the maze: MC, TD(0), TD(λ) widgets and charts
│   └── katex/         offline math rendering
├── _template/         a starter deck with every building block; copy it for a new lecture
├── lecture-02-dynamic-programming/
└── lecture-03-model-free-prediction/
```

## Presenting
Open `index.html` in Chrome or Safari.

| Key | Action |
|---|---|
| → / Space / clicker | next step (fragment) or slide |
| ← | back |
| O | overview of all slides (click one to jump) |
| N | speaker notes overlay |
| S | presenter window (notes, next slide, timer), synced with the main window |
| B | blank (white) screen, e.g. while working on the board |
| T | hide the yellow TODO boxes |
| F | fullscreen |

`#12` at the end of the URL opens slide 12.

**PDF for students:** Chrome → Print → Save as PDF, margins "None", "Background graphics" on.
You get one page per slide with all fragments visible. Press T first to hide TODOs.

## Writing slides
Each slide is a `<section class="slide">`. Variants: `title-slide`, `section` (green divider), `practice-slide` (orange "Let's practice" title).
- Math: `$…$` inline, `$$…$$` display. Macros are in `assets/macros.js`.
- Step-by-step reveal: add `class="fragment"`. Inside math, use `\htmlClass{fragment}{…}`.
- Boxes: `box definition`, `box theorem`, `box practice`, `box discuss`, `box algo`.
- Columns: `<div class="cols">` (also `c-60-40`, `c-40-60`, `c-3`).
- Speaker notes: `<aside class="notes">…</aside>`.
- Reminders to yourself: `<div class="todo">…</div>`.

## The two-exit maze
```
 s11 s12  ##  [+5]       Start at s1. Deterministic moves; walls and borders: stay.
 s8  s9   ##  s10        Entering an exit ends the episode with reward +1 or +5.
 s5  [+1] s6  s7         All other rewards 0.
 s1  s2   s3  s4         From s1 the big exit is optimal iff γ > 5^(-1/4) ≈ 0.669.
```
Insert it with `<div class="maze" data-…></div>`. See `_template/` and the header of `assets/maze.js` for the attributes.
Slippery variant: `data-slip="0.2"` (0.8 intended move, 0.2/3 to each other neighbour), or `data-controls="slip"` for a toggle; `data-show="s6:right"` draws P(·|s6,→) on the grid.
`<div class="pmatrix">` shows P as one 14×14 matrix per action (12 cells + 2 absorbing exits), with action and slip toggles.
The model and solvers are available as `RLMaze` (e.g. `RLMaze.buildModel()`), so the Monte Carlo and TD lectures can simulate episodes in the same maze.

## Model-free widgets (`learn.js`, Lecture 3)
- `<div class="learner" data-algo="mc" …>`: step-by-step MC / TD(0) / TD(λ) on sampled episodes, with the trajectory drawn on the maze and every update written out (`data-info="#id"`). `data-algo="mc,td"` runs two methods side by side on the same episodes. Scripted episodes for by-hand exercises: `data-episodes="1 2 3 6 7 10 G | 1 2 g"`.
- `<div class="learn-chart" …>`: RMS error vs. episodes (or vs. λ with `data-mode="lambda"`), averaged over runs, computed live in the browser.
- `<div class="target-spread" …>`: spread of MC vs. TD targets at one state (the bias/variance picture).
- The fixed policy is ε-greedy w.r.t. π* (`data-eps`, default 0.2). Its true value V^π is computed from the model, for comparison only.

## License

Lecture content is available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original software in `assets/` is available under the [MIT License](https://opensource.org/license/mit/). See [LICENSE.md](LICENSE.md) for details.
