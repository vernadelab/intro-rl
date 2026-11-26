/* =====================================================================
   Policy-gradient demos in the two-exit maze (Lecture 6). Needs maze.js.

   1) Occupancy measure of a fixed policy
      <div class="occupancy" data-policy="uniform" data-gamma="0.9" data-size="360"
           data-controls="policy,gamma,sample"></div>
      policies: uniform | route (→ big exit, ε-noise) | toSmall (→ small exit, ε-noise)
      Shows d^π(s) = Σ_a d^π(s,a) as a heat map, π as arrows, the mass absorbed at the exits,
      and the check  ⟨d^π, r⟩ / (1−γ) = V^π(s_1).  "+N trajectories" replaces the exact d^π by a
      Monte Carlo estimate from N sampled trajectories (discounted visit counts).

   2) Exact policy-gradient ascent, tabular softmax  π_θ(a|s) ∝ exp θ_{s,a},  θ_0 = 0 (uniform)
      <div class="pg-ascent" data-alpha="1" data-gamma="0.9" data-size="330"></div>
      Uses the PG theorem in closed form:  ∂J/∂θ_{s,a} = η^π(s) π(a|s) (Q^π(s,a) − V^π(s)).

   3) The θ-plane lab: linear softmax with two features,
         φ_big(s,a)   = 1{a = shortest-route action to the big exit}
         φ_small(s,a) = 1{a = shortest-route action to the small exit}
         π_θ(a|s) ∝ exp(θ_big φ_big(s,a) + θ_small φ_small(s,a))
      <div class="pg-lab" data-mode="cloud|runs" data-theta="0.5,-0.5" data-alpha="0.2"></div>
      cloud: click to move θ; sample single-trajectory REINFORCE estimates, shown as θ + α ĝ;
             compare estimators (no baseline / constant baseline / b(s) = V^π(s)).
      runs:  run several REINFORCE runs from θ_0 on top of the J landscape.
   ===================================================================== */
(function () {
  const M = window.RLMaze;
  if (!M) return;
  const NA = 4;
  const fmt = (x, d = 2) => (Math.abs(x) < 5e-13 ? 0 : x).toFixed(d);
  const tex = (el) => window.RLDeck && RLDeck.typeset(el);

  // ------------------------------------------------------------ RNG
  function rng(seed) { // mulberry32 (same as learn.js)
    let a = seed >>> 0;
    return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }
  const pick = (p, u) => { let a = 0; while (a < p.length - 1 && u >= p[a]) { u -= p[a]; a++; } return a; };

  // ------------------------------------------------------------ linear algebra (n ≤ 12)
  function solve(A, b) { // Gaussian elimination with partial pivoting
    const n = b.length, X = A.map((r, i) => r.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(X[r][c]) > Math.abs(X[p][c])) p = r;
      [X[c], X[p]] = [X[p], X[c]];
      for (let r = c + 1; r < n; r++) { const f = X[r][c] / X[c][c]; if (f) for (let k = c; k <= n; k++) X[r][k] -= f * X[c][k]; }
    }
    const x = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) { let s = X[r][n]; for (let k = r + 1; k < n; k++) s -= X[r][k] * x[k]; x[r] = s / X[r][r]; }
    return x;
  }

  // Everything about a policy π (array n×4) in model m, discount g, start m.start:
  //   V, Q, η(s) = Σ_t γ^t P(S_t = s)  (unnormalised discounted visitation),
  //   d(s) = (1−γ) η(s), and the occupancy of the two absorbing exits.
  function analyse(m, pi, g) {
    const n = m.n, I = (i, j) => (i === j ? 1 : 0);
    const P = Array.from({ length: n }, () => new Array(n).fill(0)), rpi = new Array(n).fill(0);
    const flowInto = { G: new Array(n).fill(0), g: new Array(n).fill(0) };
    for (let s = 0; s < n; s++) for (let a = 0; a < NA; a++) m.P[s][a].forEach((o) => {
      const w = pi[s][a] * o.p; rpi[s] += w * o.reward;
      if (o.done) flowInto[o.exit][s] += w; else P[s][o.next] += w;
    });
    const A = P.map((row, i) => row.map((x, j) => I(i, j) - g * x));
    const V = solve(A, rpi);
    const At = A.map((_, i) => A.map((row) => row[i]));
    const e0 = new Array(n).fill(0); e0[m.start] = 1;
    const eta = solve(At, e0);
    const Q = m.states.map((_, s) => [0, 1, 2, 3].map((a) => m.P[s][a].reduce((t, o) => t + o.p * (o.reward + (o.done ? 0 : g * V[o.next])), 0)));
    const d = eta.map((x) => (1 - g) * x);
    const fG = eta.reduce((t, x, s) => t + x * flowInto.G[s], 0), fg = eta.reduce((t, x, s) => t + x * flowInto.g[s], 0);
    const rsa = m.states.map((_, s) => [0, 1, 2, 3].map((a) => m.P[s][a].reduce((t, o) => t + o.p * o.reward, 0)));
    const dr = d.reduce((t, ds, s) => t + ds * pi[s].reduce((u, p, a) => u + p * rsa[s][a], 0), 0);
    return { V, Q, eta, d, dG: g * fG, dg: g * fg, J: V[m.start], dr };
  }

  function sampleEpisode(m, pi, rand, maxLen = 300) {
    const ep = []; let s = m.start;
    for (let t = 0; t < maxLen; t++) {
      const a = pick(pi[s], rand()), outs = m.P[s][a], o = outs[pick(outs.map((x) => x.p), rand())];
      ep.push({ s, a, r: o.reward, done: o.done, exit: o.exit, next: o.next });
      if (o.done) break; s = o.next;
    }
    return ep;
  }
  const softmaxRow = (z) => { const mx = Math.max(...z), e = z.map((x) => Math.exp(x - mx)), S = e.reduce((a, b) => a + b); return e.map((x) => x / S); };

  // Named stochastic policies for the occupancy demo
  function policyByName(m, name, eps = 0.2) {
    if (name === "route") return M.epsMix(M.towards(m, "G"), eps);
    if (name === "toSmall") return M.epsMix(M.towards(m, "g"), eps);
    return M.namedPolicy(m, "uniform", 0.9);
  }
  const POLICY_LABEL = { uniform: "uniform", route: "→ big exit (ε = 0.2)", toSmall: "→ small exit (ε = 0.2)" };

  // A maze.js grid that displays arbitrary numbers + arrows
  function makeGrid(host, size, vmax, digits = 2, values = "fixed") {
    const div = document.createElement("div");
    div.className = "maze vertical";
    Object.assign(div.dataset, { values, v: "0,0,0,0,0,0,0,0,0,0,0,0", policy: "uniform", arrows: "policy", size: String(size), vmax: String(vmax), digits: String(digits), labels: "ids", controls: "" });
    host.appendChild(div);
    return new M.Maze(div);
  }
  function button(bar, label, fn) {
    const b = document.createElement("button"); b.textContent = label;
    b.onclick = (e) => { e.stopPropagation(); fn(b); }; bar.appendChild(b); return b;
  }
  function slider(bar, label, min, max, step, val, fn) {
    const l = document.createElement("label"); l.className = "maze-gamma";
    l.innerHTML = `<span>${label} = <b>${val}</b></span><input type="range" min="${min}" max="${max}" step="${step}" value="${val}">`;
    const i = l.querySelector("input");
    i.oninput = () => { l.querySelector("b").textContent = (+i.value).toFixed(2); fn(+i.value); };
    i.addEventListener("keydown", (e) => e.stopPropagation());
    bar.appendChild(l); return l;
  }

  // ============================================================ 1) occupancy
  class Occupancy {
    constructor(root) {
      const ds = root.dataset;
      this.root = root; this.g = parseFloat(ds.gamma || "0.9");
      this.policy = ds.policy || "uniform";
      this.controls = (ds.controls || "policy,gamma,sample").split(",").map((s) => s.trim());
      this.m = M.buildModel(0, "two-exit");
      this.seed = 1; this.rand = rng(this.seed);
      root.classList.add("maze", "pg-row"); root.innerHTML = "";
      const left = document.createElement("div"); root.appendChild(left);
      this.grid = makeGrid(left, parseFloat(ds.size || "360"), parseFloat(ds.vmax || "0.25"));
      const side = document.createElement("div"); side.className = "maze-side"; root.appendChild(side);
      const bar = document.createElement("div"); bar.className = "maze-controls"; side.appendChild(bar);
      if (this.controls.includes("policy")) {
        this.pbtn = {};
        ["uniform", "route", "toSmall"].forEach((p) => { this.pbtn[p] = button(bar, POLICY_LABEL[p], () => { this.policy = p; this.resetSamples(); this.update(); }); });
      }
      const bar2 = document.createElement("div"); bar2.className = "maze-controls"; side.appendChild(bar2);
      if (this.controls.includes("gamma")) slider(bar2, "γ", 0.5, 0.99, 0.01, this.g.toFixed(2), (v) => { this.g = v; this.resetSamples(); this.update(); });
      if (this.controls.includes("sample")) {
        const bar3 = document.createElement("div"); bar3.className = "maze-controls"; side.appendChild(bar3);
        button(bar3, "+10 trajectories", () => this.addSamples(10));
        button(bar3, "+100 trajectories", () => this.addSamples(100));
        button(bar3, "exact", () => { this.resetSamples(); this.update(); });
      }
      this.info = document.createElement("div"); this.info.className = "maze-info"; side.appendChild(this.info);
      this.resetSamples(); this.update();
    }
    resetSamples() { this.N = 0; this.cnt = new Array(this.m.n).fill(0); this.cntG = 0; this.cntg = 0; }
    addSamples(k) {
      const pi = policyByName(this.m, this.policy), g = this.g;
      for (let i = 0; i < k; i++) {
        const ep = sampleEpisode(this.m, pi, this.rand, 400);
        ep.forEach((x, t) => { this.cnt[x.s] += (1 - g) * Math.pow(g, t); });
        const last = ep[ep.length - 1];
        if (last.done) { const T = ep.length; const mass = Math.pow(g, T); if (last.exit === "G") this.cntG += mass; else this.cntg += mass; }
      }
      this.N += k; this.update();
    }
    update() {
      const pi = policyByName(this.m, this.policy), A = analyse(this.m, pi, this.g);
      const est = this.N > 0;
      const d = est ? this.cnt.map((c) => c / this.N) : A.d;
      const dG = est ? this.cntG / this.N : A.dG, dg = est ? this.cntg / this.N : A.dg;
      this.grid.pi = pi; this.grid.V = d; this.grid.draw();
      if (this.pbtn) Object.entries(this.pbtn).forEach(([k, b]) => b.classList.toggle("on", k === this.policy));
      const sum = d.reduce((a, b) => a + b, 0);
      const l1 = est ? A.d.reduce((t, x, s) => t + Math.abs(x - d[s]), 0) + Math.abs(A.dG - dG) + Math.abs(A.dg - dg) : 0;
      this.info.innerHTML =
        `<div>${est ? `<b>estimated</b> from $N = ${this.N}$ trajectories &nbsp;·&nbsp; $\\|\\hat d - d^\\pi\\|_1 = ${fmt(l1, 3)}$` : "<b>exact</b> $d^\\pi$ (needs $P$)"}</div>` +
        `<div>maze cells: $\\sum_s d(s) = ${fmt(sum, 3)}$</div>` +
        `<div>absorbed at exits: $d(\\text{big}) = ${fmt(dG, 3)}$, $d(\\text{small}) = ${fmt(dg, 3)}$</div>` +
        `<div style="margin-top:8px">$\\dfrac{\\langle d^\\pi, r\\rangle}{1-\\gamma} = ${fmt(A.dr / (1 - this.g), 3)} = V^\\pi(s_1)$</div>`;
      tex(this.info);
    }
  }

  // ============================================================ 2) exact PG ascent, tabular softmax
  class Ascent {
    constructor(root) {
      const ds = root.dataset;
      this.root = root; this.g = parseFloat(ds.gamma || "0.9"); this.alpha = parseFloat(ds.alpha || "1");
      this.show = ds.show || "d"; // d | V
      this.m = M.buildModel(0, "two-exit");
      this.Vstar = M.optimal(this.m, this.g)[this.m.start];
      root.classList.add("maze", "pg-row"); root.innerHTML = "";
      const left = document.createElement("div"); root.appendChild(left);
      this.grid = makeGrid(left, parseFloat(ds.size || "330"), this.show === "d" ? 0.25 : 5);
      const side = document.createElement("div"); side.className = "maze-side"; root.appendChild(side);
      const bar = document.createElement("div"); bar.className = "maze-controls"; side.appendChild(bar);
      button(bar, "gradient step", () => this.step(1));
      button(bar, "×10", () => this.step(10));
      button(bar, "×100", () => this.step(100));
      button(bar, "reset", () => this.reset());
      this.showBtn = button(bar, "", () => { this.show = this.show === "d" ? "V" : "d"; this.grid.vmax = this.show === "d" ? 0.25 : 5; this.draw(); });
      const bar2 = document.createElement("div"); bar2.className = "maze-controls"; side.appendChild(bar2);
      slider(bar2, "α", 0.1, 10, 0.1, this.alpha.toFixed(2), (v) => { this.alpha = v; });
      this.chartHost = document.createElement("div"); side.appendChild(this.chartHost);
      this.status = document.createElement("div"); this.status.className = "maze-status"; side.appendChild(this.status);
      this.cw = parseFloat(ds.chartWidth || "520"); this.ch = parseFloat(ds.chartHeight || "200");
      this.reset();
    }
    reset() { this.th = this.m.states.map(() => [0, 0, 0, 0]); this.k = 0; this.hist = []; this.compute(); this.hist.push(this.A.J); this.draw(); }
    compute() { this.pi = this.th.map(softmaxRow); this.A = analyse(this.m, this.pi, this.g); }
    step(K) {
      for (let k = 0; k < K; k++) {
        const { eta, Q, V } = this.A;
        // PG theorem, tabular softmax: ∂J/∂θ_{s,a} = η(s) π(a|s) (Q(s,a) − V(s))
        this.th = this.th.map((row, s) => row.map((x, a) => x + this.alpha * eta[s] * this.pi[s][a] * (Q[s][a] - V[s])));
        this.k++; this.compute(); this.hist.push(this.A.J);
      }
      this.draw();
    }
    draw() {
      this.grid.pi = this.pi; this.grid.V = this.show === "d" ? this.A.d : this.A.V; this.grid.draw();
      this.showBtn.textContent = this.show === "d" ? "show V^π" : "show d^π";
      // learning curve J(θ_k)
      const W = this.cw, H = this.ch, pl = 44, pr = 10, pt = 10, pb = 30, K = Math.max(50, this.hist.length - 1);
      const X = (k) => pl + (W - pl - pr) * k / K, Y = (v) => pt + (H - pt - pb) * (1 - v / 3);
      let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" class="learn-svg">`;
      [0, 1, 2, 3].forEach((v) => { s += `<line x1="${pl}" x2="${W - pr}" y1="${Y(v)}" y2="${Y(v)}" stroke="#e4e4dc"/><text x="${pl - 6}" y="${Y(v) + 5}" font-size="16" text-anchor="end" fill="#8a9086">${v}</text>`; });
      s += `<line x1="${pl}" x2="${W - pr}" y1="${Y(this.Vstar)}" y2="${Y(this.Vstar)}" stroke="#b5541c" stroke-dasharray="6 4"/><text x="${W - pr}" y="${Y(this.Vstar) - 6}" font-size="16" text-anchor="end" fill="#b5541c">V*(s₁) = ${fmt(this.Vstar)}</text>`;
      s += `<line x1="${pl}" x2="${W - pr}" y1="${Y(0.9)}" y2="${Y(0.9)}" stroke="#8a9086" stroke-dasharray="3 4"/><text x="${W - pr}" y="${Y(0.9) - 6}" font-size="15" text-anchor="end" fill="#8a9086">small exit: γ·1 = 0.9</text>`;
      s += `<polyline fill="none" stroke="#2f5a24" stroke-width="3" points="${this.hist.map((v, k) => X(k) + "," + Y(v)).join(" ")}"/>`;
      s += `<text x="${(W + pl) / 2}" y="${H - 6}" font-size="16" text-anchor="middle" fill="#4a4f47">gradient steps k (0…${K})</text>`;
      s += `<text x="12" y="${pt + 12}" font-size="17" fill="#4a4f47">J</text></svg>`;
      this.chartHost.innerHTML = s;
      const gn = Math.sqrt(this.A.eta.reduce((t, e, s) => t + this.pi[s].reduce((u, p, a) => u + Math.pow(e * p * (this.A.Q[s][a] - this.A.V[s]), 2), 0), 0));
      this.status.innerHTML = `$k = ${this.k}$ &nbsp; $J(\\theta_k) = V^{\\pi_{\\theta_k}}(s_1) = ${fmt(this.A.J, 3)}$ &nbsp; $\\|\\nabla J\\| = ${fmt(gn, 3)}$`;
      tex(this.status);
    }
  }

  // ============================================================ 3) the θ-plane lab
  // The θ-plane lab environment. cost > 0: every step costs `cost` (rewards spread along the trajectory).
  const LABS = {};
  const makeLab = (cost = 0) => LABS[cost] || (LABS[cost] = (() => {
    const m0 = M.buildModel(0, "two-exit");
    const m = cost ? Object.assign({}, m0, { P: m0.P.map((row) => row.map((outs) => outs.map((o) => Object.assign({}, o, { reward: o.reward - cost })))) }) : m0;
    const aB = M.towards(m, "G").map((r) => r.indexOf(1)), ag = M.towards(m, "g").map((r) => r.indexOf(1));
    const phi = (s, a) => [a === aB[s] ? 1 : 0, a === ag[s] ? 1 : 0];
    const policy = (th) => m.states.map((_, s) => softmaxRow([0, 1, 2, 3].map((a) => { const f = phi(s, a); return th[0] * f[0] + th[1] * f[1]; })));
    const score = (pi, s, a) => { const f = phi(s, a); let e0 = 0, e1 = 0; for (let b = 0; b < NA; b++) { const fb = phi(s, b); e0 += pi[s][b] * fb[0]; e1 += pi[s][b] * fb[1]; } return [f[0] - e0, f[1] - e1]; };
    const g = 0.9;
    function exact(th) {
      const pi = policy(th), A = analyse(m, pi, g); const G = [0, 0];
      for (let s = 0; s < m.n; s++) for (let a = 0; a < NA; a++) { const sc = score(pi, s, a), w = A.eta[s] * pi[s][a] * A.Q[s][a]; G[0] += w * sc[0]; G[1] += w * sc[1]; }
      return { pi, A, grad: G };
    }
    // single-trajectory REINFORCE estimate.  kind: none | const | value
    function estimate(ex, rand, kind, bval = 0) {
      const { pi, A } = ex, ep = sampleEpisode(m, pi, rand), T = ep.length;
      const Gt = new Array(T); let acc = 0; for (let t = T - 1; t >= 0; t--) { acc = ep[t].r + g * acc; Gt[t] = acc; }
      const out = [0, 0];
      ep.forEach((x, t) => {
        // none:  γ^t G_t                (reward-to-go; = R(τ) here since all reward comes at the exit)
        // const: R(τ) − J(θ)            (classic "average return" baseline, same weight for every t)
        // value: γ^t (G_t − V^π(S_t))   (state-dependent baseline → advantage estimate)
        const w = kind === "full" ? Gt[0] : kind === "rtg" ? Math.pow(g, t) * Gt[t] : kind === "b" ? Gt[0] - bval : kind === "const" ? Gt[0] - A.J : kind === "arb" ? Gt[0] - 3 : kind === "value" ? Math.pow(g, t) * (Gt[t] - A.V[x.s]) : Math.pow(g, t) * Gt[t];
        const sc = score(pi, x.s, x.a); out[0] += w * sc[0]; out[1] += w * sc[1];
      });
      return { g: out, T, exit: ep[T - 1].done ? ep[T - 1].exit : null, ep, R: Gt[0] };
    }
    // Monte Carlo return of one trajectory (used by CEM to evaluate a candidate θ)
    function ret(th, rand) { const ep = sampleEpisode(m, policy(th), rand); return ep.reduce((t, x, k) => t + Math.pow(g, k) * x.r, 0); }
    const disc = (n) => (1 - Math.pow(g, n)) / (1 - g); // 1 + γ + … + γ^{n−1}
    const Jbig = 5 * Math.pow(g, 5) - cost * disc(6), Jsmall = g * 1 - cost * disc(2), Jlow = Math.min(0, -cost / (1 - g) * 0.6);
    return { m, policy, exact, estimate, ret, cost, Jbig, Jsmall, Jlow, J: (th) => analyse(m, policy(th), g).J };
  })());
  const LAB = makeLab(0);

  const KIND_LABEL = { full: "full return R(τ)", rtg: "reward-to-go γᵗGₜ", none: "no baseline", arb: "b = 3 (arbitrary)", avg: "b = running average", const: "b = J(θ) (oracle)", value: "b(s) = V^π(s) (oracle)" };
  const KIND_COLOR = { full: "#b5541c", rtg: "#2b5b84", none: "#b5541c", arb: "#7a6f8f", avg: "#2f5a24", const: "#2b5b84", value: "#8a6a00" };
  const AVG_RATE = 0.3; // running-average baseline: b = first return, then b ← b + 0.3 (R − b); b is fixed before each new trajectory
  const avgUpdate = (b, R) => (b === null ? R : b + AVG_RATE * (R - b));
  const KINDS = ["none", "arb", "const", "value"];

  class Lab {
    constructor(root) {
      const ds = root.dataset;
      this.root = root; this.mode = ds.mode || "cloud"; this.uid = "l" + (Lab.count = (Lab.count || 0) + 1);
      this.lab = makeLab(parseFloat(ds.cost || "0"));
      this.th0 = (ds.theta || "0.5,-0.5").split(",").map(Number); this.th = this.th0.slice();
      this.alpha = parseFloat(ds.alpha || "0.2"); this.kind = ds.kind || "none";
      this.lo = -3; this.hi = 7; this.S = parseFloat(ds.size || "470");
      this.seed = parseInt(ds.seed || "7", 10); this.rand = rng(this.seed); this.samples = []; this.runs = []; this.bars = null;
      root.classList.add("pg-row"); root.innerHTML = "";
      const plot = document.createElement("div"); plot.className = "pg-plot"; plot.style.cssText = `position:relative;width:${this.S}px;height:${this.S}px;flex:none`;
      root.appendChild(plot);
      this.canvas = document.createElement("canvas"); this.canvas.width = this.S * 2; this.canvas.height = this.S * 2;
      this.canvas.style.cssText = `position:absolute;left:0;top:0;width:${this.S}px;height:${this.S}px`;
      plot.appendChild(this.canvas);
      this.svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      this.svg.setAttribute("viewBox", `0 0 ${this.S} ${this.S}`); this.svg.setAttribute("width", this.S); this.svg.setAttribute("height", this.S);
      this.svg.style.cssText = "position:absolute;left:0;top:0;overflow:visible"; plot.appendChild(this.svg);
      if (this.mode === "cloud") plot.addEventListener("click", (e) => { e.stopPropagation(); const r = this.svg.getBoundingClientRect(); const k = this.S / r.width; this.th = [this.toTh((e.clientX - r.left) * k), this.toTh(this.S - (e.clientY - r.top) * k)]; this.samples = []; this.bars = null; this.drawOverlay(); this.updateInfo(); });
      this.withMaze = ds.maze === "1";
      this.static = ds.static === "1" || (this.mode === "landscape" && !this.withMaze);
      this.legend = ds.legend === "1"; if (this.legend) plot.style.marginRight = "70px";
      if (this.mode === "landscape" && this.withMaze) plot.addEventListener("click", (e) => { e.stopPropagation(); const r = this.svg.getBoundingClientRect(); const k = this.S / r.width; this.th = [this.toTh((e.clientX - r.left) * k), this.toTh(this.S - (e.clientY - r.top) * k)]; this.drawOverlay(); this.updateInfo(); });
      // landscape mode: labelled reference points "x,y,label|x,y,label"
      this.marks = (ds.marks || "").split("|").filter(Boolean).map((m) => { const [x, y, ...l] = m.split(","); return { th: [+x, +y], label: l.join(",") }; });
      const side = document.createElement("div"); side.className = "maze-side"; side.style.minWidth = "0"; if (this.static) side.style.display = "none"; root.appendChild(side);
      if (this.mode === "cem") { this.pop = parseInt(ds.pop || "20", 10); this.nElite = parseInt(ds.elite || "5", 10); this.evals = parseInt(ds.evals || "5", 10); this.sigma0 = parseFloat(ds.sigma || "2"); this.floor = parseFloat(ds.floor || "0.15"); }
      this.kinds = (ds.kinds || (this.mode === "cloud" ? "none,arb,const,value" : this.mode === "live" ? "none" : "none,const,value")).split(",").map((k) => k.trim());
      this.cmpKinds = this.kinds.filter((k) => k !== "avg");
      if (!this.kinds.includes(this.kind)) this.kind = this.kinds[0];
      this.showKinds = ds.estimators !== "0" && this.mode !== "cem" && this.mode !== "landscape" && this.kinds.length > 1; this.showCompare = ds.compare !== "0"; this.showConv = ds.convergence === "1";
      if (ds.caption) { const c = document.createElement("div"); c.className = "pg-caption"; c.innerHTML = ds.caption; side.appendChild(c); tex(c); }
      const bar = document.createElement("div"); bar.className = "maze-controls"; side.appendChild(bar);
      this.kbtn = {};
      if (this.showKinds) this.kinds.forEach((k) => { this.kbtn[k] = button(bar, KIND_LABEL[k], () => { this.kind = k; if (this.mode !== "cloud") this.samples = []; this.runs = []; if (this.mode === "live") this.liveReset(); this.drawOverlay(); this.updateInfo(); }); });
      else bar.remove();
      const bar2 = document.createElement("div"); bar2.className = "maze-controls"; side.appendChild(bar2);
      if (this.mode === "cloud") {
        button(bar2, "sample 1", () => this.sample(1));
        button(bar2, "sample 20", () => this.sample(20));
        if (this.showConv) button(bar2, "sample 200", () => this.sample(200));
        button(bar2, "clear", () => { this.samples = []; this.drawOverlay(); this.updateInfo(); });
        if (this.showCompare) button(bar2, "compare estimators", () => this.compare());
      } else if (this.mode === "landscape") {
        bar2.remove();
      } else if (this.mode === "live") {
        this.N = parseInt(ds.batch || "1", 10); this.maxK = parseInt(ds.steps || "150", 10);
        button(bar2, "1 update", () => { this.stop(); this.liveStep(1); });
        this.playBtn = button(bar2, "▶ play", () => (this.timer ? this.stop() : this.play()));
        button(bar2, "reset", () => { this.stop(); this.liveReset(); this.drawOverlay(); this.updateInfo(); });
        button(bar2, "new seed", () => { this.stop(); this.seed += 1; this.liveReset(); this.drawOverlay(); this.updateInfo(); });
      } else if (this.mode === "cem") {
        this.stepBtn = button(bar2, "sample population", () => this.cemStep());
        button(bar2, "iterate ×5", () => { for (let i = 0; i < 10; i++) this.cemStep(); });
        button(bar2, "reset", () => { this.cemReset(); this.drawOverlay(); this.updateInfo(); });
      } else {
        this.N = parseInt(ds.batch || "1", 10);
        this.nbtn = {};
        [1, 10].forEach((N) => { this.nbtn[N] = button(bar2, `${N} ${N > 1 ? "trajectories" : "trajectory"} / update`, () => { this.N = N; this.updateInfo(); }); });
        button(bar2, "run ×10", () => this.runMany(10));
        button(bar2, "clear", () => { this.runs = []; this.drawOverlay(); this.updateInfo(); });
      }
      const bar3 = document.createElement("div"); bar3.className = "maze-controls"; side.appendChild(bar3);
      if (this.mode !== "cem" && this.mode !== "landscape") slider(bar3, "α", 0.05, this.mode === "cloud" ? 3 : 1, 0.05, this.alpha.toFixed(2), (v) => { this.alpha = v; if (this.mode === "live") { this.stop(); this.liveReset(); } this.drawOverlay(); this.updateInfo(); });
      else bar3.remove();
      if (this.withMaze || this.mode === "live") {
        const row = document.createElement("div"); row.className = "pg-maze-row"; side.appendChild(row);
        const mh = document.createElement("div"); row.appendChild(mh);
        this.grid = makeGrid(mh, parseFloat(ds.mazeSize || "220"), 1, 2, "none"); this.grid.pathColors = ["#2b5b84"];
        this.info = document.createElement("div"); this.info.className = "maze-info"; row.appendChild(this.info);
      } else { this.info = document.createElement("div"); this.info.className = "maze-info"; side.appendChild(this.info); }
      this.chartHost = document.createElement("div"); side.appendChild(this.chartHost);
      if (ds.footer) { const f = document.createElement("div"); f.innerHTML = ds.footer; side.appendChild(f); tex(f); }
      this.barHost = document.createElement("div"); side.appendChild(this.barHost);
      this.convHost = document.createElement("div"); side.appendChild(this.convHost);
      if (this.mode === "cem") this.cemReset();
      if (this.mode === "live") this.liveReset();
      this.drawLandscape(); this.exactPath(); this.drawOverlay(); this.updateInfo();
    }
    // ---------------- REINFORCE, one update at a time
    liveReset() {
      this.rand = rng(this.seed); this.th = this.th0.slice(); this.k = 0; this.used = 0; this.bbar = null;
      this.lpath = [this.th.slice()]; this.Jh = [this.lab.J(this.th)]; this.lastEp = null; this.lastR = null; this.lastStep = null;
      // exact gradient ascent with the same step size and number of updates: the noise-free reference
      let t = this.th0.slice(); this.exTh = [t.slice()]; this.exJ = [this.lab.J(t)];
      for (let k = 0; k < this.maxK; k++) { const G = this.lab.exact(t).grad; t = [t[0] + this.alpha * G[0], t[1] + this.alpha * G[1]]; this.exTh.push(t.slice()); this.exJ.push(this.lab.J(t)); }
    }
    liveStep(n) {
      for (let j = 0; j < n && this.k < this.maxK; j++) {
        const ex = this.lab.exact(this.th); let G = [0, 0]; const b0 = this.bbar === null ? 0 : this.bbar; let e;
        for (let i = 0; i < this.N; i++) {
          e = this.lab.estimate(ex, this.rand, this.kind === "avg" ? "b" : this.kind, b0);
          G[0] += e.g[0] / this.N; G[1] += e.g[1] / this.N;
          if (this.kind === "avg") this.bbar = avgUpdate(this.bbar, e.R);
        }
        const from = this.th.slice();
        this.th = [Math.max(this.lo, Math.min(this.hi + 2, from[0] + this.alpha * G[0])), Math.max(this.lo, Math.min(this.hi + 2, from[1] + this.alpha * G[1]))];
        this.lastStep = { from, to: this.th.slice(), exact: [from[0] + this.alpha * ex.grad[0], from[1] + this.alpha * ex.grad[1]], b: this.kind === "avg" ? b0 : this.kind === "const" ? ex.A.J : 0 };
        this.lastEp = e.ep; this.lastR = e.R; this.k++; this.used += this.N;
        this.lpath.push(this.th.slice()); this.Jh.push(this.lab.J(this.th));
      }
      this.drawOverlay(); this.updateInfo();
    }
    play() { if (this.k >= this.maxK) this.liveReset(); this.playBtn.textContent = "❚❚ pause"; this.timer = setInterval(() => { this.liveStep(1); if (this.k >= this.maxK) this.stop(); }, 110); }
    stop() { if (this.timer) clearInterval(this.timer); this.timer = null; if (this.playBtn) this.playBtn.textContent = "▶ play"; }
    // ---------------- cross-entropy method on the θ-plane
    cemReset() { this.mu = this.th0.slice(); this.sig = [this.sigma0, this.sigma0]; this.popl = []; this.phase = "sample"; this.k = 0; this.used = 0; this.muHist = [this.mu.slice()]; this.rand = rng(this.seed); }
    cemStep() { // alternates: sample + evaluate a population  /  refit the Gaussian to the elites
      if (this.phase === "sample") {
        const gauss = () => { const u = Math.max(1e-12, this.rand()), v = this.rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
        this.popl = [];
        for (let n = 0; n < this.pop; n++) {
          const th = [this.mu[0] + this.sig[0] * gauss(), this.mu[1] + this.sig[1] * gauss()];
          let J = 0; for (let i = 0; i < this.evals; i++) J += this.lab.ret(th, this.rand) / this.evals;
          this.popl.push({ th, J, elite: false });
        }
        this.used += this.pop * this.evals;
        this.popl.slice().sort((a, b) => b.J - a.J).slice(0, this.nElite).forEach((x) => { x.elite = true; });
        this.phase = "refit";
      } else {
        const E = this.popl.filter((x) => x.elite);
        this.mu = [0, 1].map((i) => E.reduce((t, x) => t + x.th[i], 0) / E.length);
        this.sig = [0, 1].map((i) => Math.sqrt(E.reduce((t, x) => t + (x.th[i] - this.mu[i]) ** 2, 0) / E.length + this.floor ** 2));
        this.muHist.push(this.mu.slice()); this.k++; this.phase = "sample";
        this.popl.forEach((x) => { x.old = true; });
      }
      this.drawOverlay(); this.updateInfo();
    }
    toPx(t) { return (t - this.lo) / (this.hi - this.lo) * this.S; }
    toTh(px) { return this.lo + px / this.S * (this.hi - this.lo); }
    drawLandscape() {
      const ctx = this.canvas.getContext("2d"), R = 120, W = this.canvas.width, img = ctx.createImageData(R, R);
      for (let i = 0; i < R; i++) for (let j = 0; j < R; j++) {
        const tb = this.lo + (j + 0.5) / R * (this.hi - this.lo), ts = this.hi - (i + 0.5) / R * (this.hi - this.lo);
        const v = this.lab.J([tb, ts]), lo = this.lab.Jlow, hi = this.lab.Jbig, lev = Math.max(0, Math.min(14, Math.floor((v - lo) / (hi - lo) * 14 + 1e-9))) / 14, c = M.heat(0.04 + 0.96 * lev).match(/\d+/g).map(Number), k = 4 * (i * R + j);
        img.data[k] = c[0]; img.data[k + 1] = c[1]; img.data[k + 2] = c[2]; img.data[k + 3] = 255;
      }
      const off = document.createElement("canvas"); off.width = R; off.height = R; off.getContext("2d").putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true; ctx.drawImage(off, 0, 0, W, W);
    }
    exactPath() { // exact gradient ascent from θ0 (small steps) = the "noise-free" path
      if (this.mode === "cem" || this.mode === "landscape" || this.mode === "live") { this.path = []; return; }
      let th = this.th0.slice(); this.path = [th.slice()];
      for (let k = 0; k < 400; k++) { const G = this.lab.exact(th).grad; th = [th[0] + 0.1 * G[0], th[1] + 0.1 * G[1]]; this.path.push(th.slice()); if (th[0] > this.hi) break; }
    }
    sample(k) {
      const ex = this.lab.exact(this.th);
      for (let i = 0; i < k; i++) this.samples.push({ g: this.lab.estimate(ex, this.rand, this.kind).g, kind: this.kind });
      this.drawOverlay(); this.updateInfo();
    }
    compare() {
      const ex = this.lab.exact(this.th), r = rng(11), K = 2000, res = {};
      this.cmpKinds.forEach((kind) => {
        let mse = 0; for (let i = 0; i < K; i++) { const e = this.lab.estimate(ex, r, kind).g; mse += Math.pow(e[0] - ex.grad[0], 2) + Math.pow(e[1] - ex.grad[1], 2); }
        res[kind] = mse / K;
      });
      this.bars = { res, norm: ex.grad[0] ** 2 + ex.grad[1] ** 2 }; this.updateInfo();
    }
    runMany(R) {
      const iters = Math.round(150 / Math.sqrt(this.N));
      for (let r = 0; r < R; r++) {
        const rand = rng(1000 + this.runs.length + 97 * r); let th = this.th0.slice(); const path = [th.slice()]; let bbar = null;
        for (let k = 0; k < iters; k++) {
          const ex = this.lab.exact(th); let G = [0, 0]; const b0 = bbar === null ? 0 : bbar;
          for (let i = 0; i < this.N; i++) { const e = this.lab.estimate(ex, rand, this.kind === "avg" ? "b" : this.kind, b0); G[0] += e.g[0] / this.N; G[1] += e.g[1] / this.N; if (this.kind === "avg") bbar = avgUpdate(bbar, e.R); }
          th = [Math.max(this.lo, Math.min(this.hi + 2, th[0] + this.alpha * G[0])), Math.max(this.lo, Math.min(this.hi + 2, th[1] + this.alpha * G[1]))];
          path.push(th.slice());
        }
        this.runs.push({ path, kind: this.kind, N: this.N, J: this.lab.J(th) });
      }
      this.drawOverlay(); this.updateInfo();
    }
    drawOverlay() {
      const S = this.S, P = (t) => [this.toPx(t[0]), S - this.toPx(t[1])];
      let s = `<defs><marker id="pgl-ah" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4" markerHeight="4" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#1b1d1a"/></marker>
        <marker id="pgl-ah2" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4" markerHeight="4" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#b5541c"/></marker>
        <clipPath id="pgl-clip-${this.uid}"><rect x="0" y="0" width="${S}" height="${S}"/></clipPath></defs>`;
      // axes
      for (let v = this.lo; v <= this.hi; v++) {
        const x = this.toPx(v);
        s += `<line x1="${x}" x2="${x}" y1="${S}" y2="${S + (v % 2 ? 4 : 8)}" stroke="#4a4f47"/><line y1="${S - x}" y2="${S - x}" x1="0" x2="${v % 2 ? -4 : -8}" stroke="#4a4f47"/>`;
        if (Math.abs(v % 2) === 1 || v === 0) s += `<text x="${x}" y="${S + 24}" font-size="14" text-anchor="middle" fill="#4a4f47">${v}</text><text x="-12" y="${S - x + 5}" font-size="14" text-anchor="end" fill="#4a4f47">${v}</text>`;
      }
      s += `<line x1="${this.toPx(0)}" x2="${this.toPx(0)}" y1="0" y2="${S}" stroke="#fff" stroke-opacity=".35"/><line y1="${S - this.toPx(0)}" y2="${S - this.toPx(0)}" x1="0" x2="${S}" stroke="#fff" stroke-opacity=".35"/>`;
      s += `<text x="${S / 2}" y="${S + 44}" font-size="17" text-anchor="middle" fill="#1b1d1a">θ_big  (weight on "go to big exit")</text>`;
      s += `<text x="${-44}" y="${S / 2}" font-size="17" text-anchor="middle" fill="#1b1d1a" transform="rotate(-90 ${-44} ${S / 2})">θ_small  (weight on "go to small exit")</text>`;
      s += `<text x="${this.toPx(-2.6)}" y="${S - this.toPx(5.6)}" font-size="16" fill="#1b1d1a">small exit: J ≈ ${fmt(this.lab.Jsmall, 2)}</text>`;
      s += `<text x="${this.toPx(6.8)}" y="${S - this.toPx(-2.4)}" font-size="16" text-anchor="end" fill="#fff" font-weight="700">big exit: J → ${fmt(this.lab.Jbig, 2)}</text>`;
      s += `<rect x="0" y="0" width="${S}" height="${S}" fill="none" stroke="#1b1d1a" stroke-width="1.5"/>`;
      // exact gradient-ascent path from θ0
      s += `<polyline points="${this.path.map((t) => P(t).join(",")).join(" ")}" fill="none" stroke="#1b1d1a" stroke-width="2.5" stroke-dasharray="7 5" opacity=".8"/>`;
      if (this.legend) { // colour scale of J(θ), same bands as the landscape
        const lo = this.lab.Jlow, hi = this.lab.Jbig, x0 = S + 22, w = 16, Y = (v) => S * (1 - (v - lo) / (hi - lo));
        for (let l = 0; l < 15; l++) { const v0 = lo + l / 14 * (hi - lo), v1 = Math.min(hi, lo + (l + 1) / 14 * (hi - lo)); if (v0 >= hi) break;
          s += `<rect x="${x0}" y="${Y(v1)}" width="${w}" height="${Y(v0) - Y(v1) + 0.5}" fill="${M.heat(0.04 + 0.96 * l / 14)}"/>`; }
        s += `<rect x="${x0}" y="0" width="${w}" height="${S}" fill="none" stroke="#1b1d1a" stroke-width="1"/>`;
        [[lo, fmt(lo, 2)], [0, "0"], [this.lab.Jsmall, fmt(this.lab.Jsmall, 2)], [hi, fmt(hi, 2)]].filter((x, i, A) => i === 0 || Math.abs(x[0] - A[i - 1][0]) > 0.15).forEach(([v, l]) => { s += `<line x1="${x0 + w}" x2="${x0 + w + 5}" y1="${Y(v)}" y2="${Y(v)}" stroke="#1b1d1a"/><text x="${x0 + w + 8}" y="${Y(v) + 5}" font-size="14" fill="#1b1d1a">${l}</text>`; });
        s += `<text x="${x0 + w / 2}" y="-10" font-size="15" text-anchor="middle" fill="#1b1d1a" font-weight="700">J(θ)</text>`;
      }
      if (this.mode === "live") {
        s += `<polyline points="${this.exTh.map((t) => P(t).join(",")).join(" ")}" fill="none" stroke="#1b1d1a" stroke-width="2.5" stroke-dasharray="7 5" opacity=".75" clip-path="url(#pgl-clip-${this.uid})"/>`;
        s += `<polyline points="${this.lpath.map((t) => P(t).join(",")).join(" ")}" fill="none" stroke="${KIND_COLOR[this.kind]}" stroke-width="2" opacity=".9" clip-path="url(#pgl-clip-${this.uid})"/>`;
        if (this.lastStep) {
          const a = P(this.lastStep.from), q = P(this.lastStep.exact);
          s += `<line x1="${a[0]}" y1="${a[1]}" x2="${q[0]}" y2="${q[1]}" stroke="#1b1d1a" stroke-width="3.5" marker-end="url(#pgl-ah)"/>`;
        }
        const p0 = P(this.th0), c = P(this.th);
        s += `<circle cx="${p0[0]}" cy="${p0[1]}" r="5" fill="#fff" stroke="#1b1d1a" stroke-width="2"/><text x="${p0[0] - 8}" y="${p0[1] + 22}" font-size="16" font-weight="700" fill="#1b1d1a" text-anchor="end">θ₀</text>`;
        s += `<circle cx="${Math.max(0, Math.min(S, c[0]))}" cy="${Math.max(0, Math.min(S, c[1]))}" r="7.5" fill="${KIND_COLOR[this.kind]}" stroke="#fff" stroke-width="2.5"/>`;
      } else if (this.mode === "landscape") {
        if (this.withMaze) { const q = P(this.th), first = this.th[0] === this.th0[0] && this.th[1] === this.th0[1];
          s += `<circle cx="${q[0]}" cy="${q[1]}" r="8" fill="#fff" stroke="#1b1d1a" stroke-width="2.5"/><text x="${q[0] + 12}" y="${q[1] + 6}" font-size="18" font-weight="700" fill="#1b1d1a" paint-order="stroke" stroke="#fff" stroke-width="4">${first ? "θ₀" : "θ"}</text>`; }
        this.marks.forEach((mk) => { const q = P(mk.th), right = q[0] < S - 150;
          s += `<circle cx="${q[0]}" cy="${q[1]}" r="6.5" fill="#fff" stroke="#1b1d1a" stroke-width="2"/>`;
          s += `<text x="${q[0] + (right ? 11 : -11)}" y="${q[1] + 5}" font-size="15" font-weight="700" text-anchor="${right ? "start" : "end"}" fill="#1b1d1a" paint-order="stroke" stroke="#fff" stroke-width="4">${mk.label}</text>`; });
      } else if (this.mode === "cem") {
        const sx = S / (this.hi - this.lo), m = P(this.mu);
        [2, 1].forEach((c) => { s += `<ellipse clip-path="url(#pgl-clip-${this.uid})" cx="${m[0]}" cy="${m[1]}" rx="${c * this.sig[0] * sx}" ry="${c * this.sig[1] * sx}" fill="${c === 1 ? "rgba(255,255,255,.16)" : "none"}" stroke="#ffffff" stroke-width="${c === 1 ? 2.5 : 1.8}" stroke-dasharray="${c === 1 ? "none" : "6 4"}"/>`; });
        if (this.muHist.length > 1) s += `<polyline points="${this.muHist.map((t) => P(t).join(",")).join(" ")}" fill="none" stroke="#ffffff" stroke-width="2.5" opacity=".85"/>`;
        this.popl.forEach((x) => { const q = P(x.th); if (q[0] < 0 || q[0] > S || q[1] < 0 || q[1] > S) return;
          s += `<circle cx="${q[0]}" cy="${q[1]}" r="${x.elite ? 6.5 : 4.5}" fill="${x.elite ? "#f3d97a" : "#fff"}" stroke="#1b1d1a" stroke-width="1.3" opacity="${x.old ? 0.35 : 1}"/>`; });
        s += `<circle cx="${m[0]}" cy="${m[1]}" r="7" fill="#fff" stroke="#1b1d1a" stroke-width="2.5"/><text x="${m[0] + 10}" y="${m[1] - 10}" font-size="16" font-weight="700" fill="#1b1d1a">μ</text>`;
      } else if (this.mode === "runs") {
        this.runs.forEach((r) => {
          const pts = r.path.map(P); const c = r.J < (this.lab.Jsmall + this.lab.Jbig) / 2 ? "#b5541c" : "#ffffff";
          s += `<polyline clip-path="url(#pgl-clip-${this.uid})" points="${pts.map((p) => p.join(",")).join(" ")}" fill="none" stroke="${KIND_COLOR[r.kind]}" stroke-width="1.6" opacity=".75"/>`;
          const e = pts[pts.length - 1]; s += `<circle cx="${Math.max(0, Math.min(S, e[0]))}" cy="${Math.max(0, Math.min(S, e[1]))}" r="6" fill="${c}" stroke="#1b1d1a" stroke-width="1.5"/>`;
        });
        const p0 = P(this.th0); s += `<circle cx="${p0[0]}" cy="${p0[1]}" r="7" fill="#fff" stroke="#1b1d1a" stroke-width="2.5"/><text x="${p0[0] + 10}" y="${p0[1] + 22}" font-size="16" font-weight="700" fill="#1b1d1a">θ₀</text>`;
      } else {
        const ex = this.lab.exact(this.th), p = P(this.th);
        this.samples.forEach(({ g, kind }) => { const c = KIND_COLOR[kind], q = P([this.th[0] + this.alpha * g[0], this.th[1] + this.alpha * g[1]]); s += `<g clip-path="url(#pgl-clip-${this.uid})"><line x1="${p[0]}" y1="${p[1]}" x2="${q[0]}" y2="${q[1]}" stroke="${c}" stroke-width="1.3" opacity=".55"/><circle cx="${q[0]}" cy="${q[1]}" r="3.2" fill="${c}"/></g>`; });
        this.kinds.forEach((kind) => { const G = this.samples.filter((x) => x.kind === kind).map((x) => x.g); if (G.length < 2) return;
          const mg = [0, 1].map((i) => G.reduce((t, g) => t + g[i], 0) / G.length), q = P([this.th[0] + this.alpha * mg[0], this.th[1] + this.alpha * mg[1]]);
          s += `<circle cx="${q[0]}" cy="${q[1]}" r="7.5" fill="#fff" fill-opacity=".35" stroke="${KIND_COLOR[kind]}" stroke-width="3.5"/>`; });
        const q = P([this.th[0] + this.alpha * ex.grad[0], this.th[1] + this.alpha * ex.grad[1]]);
        s += `<line x1="${p[0]}" y1="${p[1]}" x2="${q[0]}" y2="${q[1]}" stroke="#1b1d1a" stroke-width="4" marker-end="url(#pgl-ah)"/>`;
        s += `<circle cx="${p[0]}" cy="${p[1]}" r="6" fill="#fff" stroke="#1b1d1a" stroke-width="2.5"/>`;
      }
      this.svg.innerHTML = s;
    }
    updateInfo() {
      Object.entries(this.kbtn).forEach(([k, b]) => b.classList.toggle("on", k === this.kind));
      if (this.nbtn) Object.entries(this.nbtn).forEach(([k, b]) => b.classList.toggle("on", +k === this.N));
      const ex = this.lab.exact(this.th);
      let h;
      if (this.mode === "cloud") {
        h = `<div>$\\theta = (${fmt(this.th[0])}, ${fmt(this.th[1])})$ &nbsp; $J(\\theta) = ${fmt(ex.A.J, 3)}$</div>` +
          `<div>$\\nabla J(\\theta) = (${fmt(ex.grad[0], 3)}, ${fmt(ex.grad[1], 3)})$ <span class="muted">(black arrow, ×α)</span></div>`;
        const multi = this.kinds.length > 1, used = this.kinds.filter((k) => this.samples.some((x) => x.kind === k));
        used.forEach((kind) => {
          const G = this.samples.filter((x) => x.kind === kind).map((x) => x.g), K = G.length, mg = [0, 1].map((i) => G.reduce((t, g) => t + g[i], 0) / K);
          const cos = G.reduce((t, g) => { const ng = Math.hypot(g[0], g[1]), nG = Math.hypot(ex.grad[0], ex.grad[1]); return t + (ng > 1e-12 ? (g[0] * ex.grad[0] + g[1] * ex.grad[1]) / (ng * nG) : 0); }, 0) / K;
          const spread = Math.sqrt(G.reduce((t, g) => t + (g[0] - mg[0]) ** 2 + (g[1] - mg[1]) ** 2, 0) / K);
          h += multi
            ? `<div style="margin-top:4px"><b style="color:${KIND_COLOR[kind]}">● ${KIND_LABEL[kind]}</b>: ${K} estimates, spread $${fmt(spread, 2)}$</div>`
            : `<div>${K} estimates · mean $= (${fmt(mg[0], 3)}, ${fmt(mg[1], 3)})$ <span class="muted">(ring)</span></div><div>average $\\cos(\\hat g, \\nabla J) = ${fmt(cos, 2)}$</div>`;
        });
        if (!used.length) h += `<div class="muted">Click the plane to move θ.</div>`;
        else if (multi) h += `<div class="muted small">rings: mean of each cloud · spread: root-mean-square distance to that mean</div>`;
      } else if (this.mode === "landscape") {
        h = "";
        if (this.withMaze) {
          const pi = this.lab.policy(this.th); this.grid.pi = pi; this.grid.paths = null; this.grid.draw();
          const p1 = pi[this.lab.m.start];
          h = `<div>$\\theta = (${fmt(this.th[0], 1)}, ${fmt(this.th[1], 1)})$</div><div>$J(\\theta) = ${fmt(ex.A.J, 2)}$</div>` +
            `<div class="small" style="margin-top:8px">in $s_1$:</div><div class="small">${["↑", "→", "↓", "←"].map((a, i) => `${a}&nbsp;${fmt(p1[i], 2)}`).join("<br>")}</div>`;
        }
      } else if (this.mode === "live") {
        const pi = this.lab.policy(this.th); this.grid.pi = pi;
        this.grid.paths = this.lastEp ? [this.lastEp.map((x) => String(x.s + 1)).concat(this.lastEp[this.lastEp.length - 1].done ? [this.lastEp[this.lastEp.length - 1].exit] : [])] : null;
        this.grid.draw();
        const last = this.lastEp ? this.lastEp[this.lastEp.length - 1] : null;
        h = `<div>update $k = ${this.k}$ <span class="muted">/ ${this.maxK}</span></div>` +
          `<div>trajectories used: <b>${this.used}</b></div>` +
          (this.lastEp ? `<div style="margin-top:6px">last trajectory: ${this.lastEp.length} steps, ${last.done ? (last.exit === "G" ? "big exit" : "small exit") : "no exit"}</div><div>$R(\\tau) = ${fmt(this.lastR, 2)}$` + (this.kind === "avg" || this.kind === "const" ? `, $\\;b = ${fmt(this.lastStep.b, 2)}$` : "") + `</div>` : `<div class="muted" style="margin-top:6px">Press "1 update" or "play".</div>`) +
          `<div style="margin-top:6px">$J(\\theta_k) = ${fmt(this.Jh[this.Jh.length - 1], 2)}$</div>`;
      } else if (this.mode === "cem") {
        h = `<div>iteration $${this.k}$ &nbsp; $\\mu = (${fmt(this.mu[0])}, ${fmt(this.mu[1])})$ &nbsp; $\\sigma = (${fmt(this.sig[0])}, ${fmt(this.sig[1])})$</div>` +
          `<div>$J(\\mu) = ${fmt(this.lab.J(this.mu), 3)}$ <span class="muted">(exact, for reference only)</span></div>` +
          `<div>population $N = ${this.pop}$, elites $k = ${this.nElite}$, each $\\theta_n$ scored with $M = ${this.evals}$ trajectories</div>` +
          `<div>trajectories used so far: <b>${this.used}</b></div>` +
          `<div class="muted">next: ${this.phase === "sample" ? "sample and score a new population" : "refit N(μ, σ²) to the elites (yellow)"}</div>`;
        if (this.stepBtn) this.stepBtn.textContent = this.phase === "sample" ? "sample population" : "refit to elites";
      } else {
        const mid = (this.lab.Jsmall + this.lab.Jbig) / 2, done = this.runs.length, trapped = this.runs.filter((r) => r.J < mid).length;
        h = `<div>REINFORCE from $\\theta_0 = (${fmt(this.th0[0], 1)}, ${fmt(this.th0[1], 1)})$</div><div>${Math.round(150 / Math.sqrt(this.N))} updates × ${this.N} ${this.N > 1 ? "trajectories" : "trajectory"} = ${Math.round(150 / Math.sqrt(this.N)) * this.N} trajectories per run</div>` +
          `<div class="muted">dashed: exact gradient ascent (the "noise-free" path)</div>` +
          (done ? `<div>${done} runs · <b class="orange">${trapped}</b> ended at the small exit <span class="muted">(orange dots)</span></div>` : "");
      }
      this.info.innerHTML = h; tex(this.info);
      if (this.mode === "live") this.drawLiveChart();
      if (this.bars) {
        const W = 520, rowH = 34, mx = Math.max(...Object.values(this.bars.res));
        let s = `<div class="small" style="margin-top:8px">$\\E\\|\\hat g - \\nabla J\\|^2$ over 2000 trajectories at this θ <span class="muted">($\\|\\nabla J\\|^2 = ${fmt(this.bars.norm, 3)}$)</span></div><svg viewBox="0 0 ${W} ${this.cmpKinds.length * rowH + 6}" width="${W}" height="${this.cmpKinds.length * rowH + 6}" class="learn-svg">`;
        this.cmpKinds.forEach((k, i) => {
          const v = this.bars.res[k], w = (W - 280) * v / mx;
          s += `<text x="0" y="${i * rowH + 22}" font-size="16" fill="#1b1d1a">${KIND_LABEL[k]}</text><rect x="210" y="${i * rowH + 6}" width="${w}" height="22" rx="4" fill="${KIND_COLOR[k]}"/><text x="${216 + w}" y="${i * rowH + 23}" font-size="15" fill="#4a4f47">${fmt(v, 2)}</text>`;
        });
        this.barHost.innerHTML = s + "</svg>"; tex(this.barHost);
      } else this.barHost.innerHTML = "";
      if (this.showConv && this.mode === "cloud") this.drawConvergence(ex.grad);
    }
    drawLiveChart() {
      const W = 560, H = 150, pl = 40, pr = 12, pt = 8, pb = 28, K = this.maxK;
      const y0 = Math.min(0, Math.floor(this.lab.Jlow)), X = (k) => pl + (W - pl - pr) * k / K, Y = (v) => pt + (H - pt - pb) * (1 - (v - y0) / (3 - y0));
      let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" class="learn-svg">`;
      for (let v = y0; v <= 3; v++) s += `<line x1="${pl}" x2="${W - pr}" y1="${Y(v)}" y2="${Y(v)}" stroke="#e4e4dc"/><text x="${pl - 6}" y="${Y(v) + 5}" font-size="13" text-anchor="end" fill="#8a9086">${v}</text>`;
      const js = this.lab.Jsmall; s += `<line x1="${pl}" x2="${W - pr}" y1="${Y(js)}" y2="${Y(js)}" stroke="#8a9086" stroke-dasharray="3 4"/><text x="${W - pr}" y="${Y(js) - 5}" font-size="13" text-anchor="end" fill="#8a9086">small exit ${fmt(js, 2)}</text>`;
      s += `<polyline fill="none" stroke="#1b1d1a" stroke-width="2" stroke-dasharray="6 4" points="${this.exJ.map((v, k) => X(k) + "," + Y(v)).join(" ")}"/>`;
      s += `<polyline fill="none" stroke="${KIND_COLOR[this.kind]}" stroke-width="2.5" points="${this.Jh.map((v, k) => X(k) + "," + Y(v)).join(" ")}"/>`;
      s += `<text x="${(W + pl) / 2}" y="${H - 6}" font-size="14" text-anchor="middle" fill="#4a4f47">J(θ_k) after k updates · dashed: exact gradient ascent, same α</text>`;
      s += `<text x="8" y="${pt + 12}" font-size="15" fill="#4a4f47">J</text></svg>`;
      this.chartHost.innerHTML = s;
    }
    // cos( mean of the first k estimates , ∇J ) as a function of k
    drawConvergence(G) {
      const SM = this.samples.filter((x) => x.kind === this.kind).map((x) => x.g), K = SM.length, W = 520, H = 118, pl = 34, pr = 10, pt = 8, pb = 26, nG = Math.hypot(G[0], G[1]);
      const Kmax = Math.max(20, K), X = (k) => pl + (W - pl - pr) * (k - 1) / Math.max(1, Kmax - 1), Y = (c) => pt + (H - pt - pb) * (1 - c) / 2;
      // cosine of the running average with the true gradient
      const cs = []; let a0 = 0, a1 = 0;
      SM.forEach((g) => { a0 += g[0]; a1 += g[1]; const n = Math.hypot(a0, a1); cs.push(n > 1e-12 ? (a0 * G[0] + a1 * G[1]) / (n * nG) : 0); });
      let settle = null; if (K) { let last = 0; cs.forEach((c, i) => { if (c < 0.9) last = i + 1; }); settle = last < K ? last + 1 : null; }
      let s = `<div class="small" style="margin-top:4px">Average of the first $k$ estimates: $\\cos(\\bar g_k, \\nabla J)$` +
        (K ? (settle ? ` · stays above $0.9$ from $k = ${settle}$` : ` · not yet stably above $0.9$`) : "") + `</div>`;
      s += `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" class="learn-svg">`;
      [-1, 0, 0.9, 1].forEach((c) => { s += `<line x1="${pl}" x2="${W - pr}" y1="${Y(c)}" y2="${Y(c)}" stroke="${c === 0 ? "#8a9086" : c === 0.9 ? "#2f5a24" : "#e4e4dc"}" ${c === 0 || c === 0.9 ? 'stroke-dasharray="4 4"' : ""}/>`; if (c !== 0.9) s += `<text x="${pl - 6}" y="${Y(c) + 5}" font-size="13" text-anchor="end" fill="#8a9086">${c}</text>`; });
      if (K) {
        SM.forEach((g, i) => { const n = Math.hypot(g[0], g[1]); if (n > 1e-12) s += `<circle cx="${X(i + 1)}" cy="${Y((g[0] * G[0] + g[1] * G[1]) / (n * nG))}" r="1.8" fill="${KIND_COLOR[this.kind]}" opacity=".3"/>`; });
        s += `<polyline fill="none" stroke="${KIND_COLOR[this.kind]}" stroke-width="2.5" points="${cs.map((c, i) => X(i + 1) + "," + Y(c)).join(" ")}"/>`;
      }
      s += `<text x="${(W + pl) / 2}" y="${H - 6}" font-size="14" text-anchor="middle" fill="#4a4f47">trajectories k = 1…${Kmax} (line: running average · dots: single trajectories)</text></svg>`;
      this.convHost.innerHTML = s; tex(this.convHost);
    }
  }

  const css = `.pg-row { display:flex; gap:56px; align-items:flex-start; font-family: var(--font-text); }
    .pg-row .maze-side { flex: 1; }
    .pg-row .maze-info > div { margin: 2px 0; }
    .pg-plot { margin: 4px 0 0 56px; }
    .pg-caption { font-size: 19px; line-height: 1.4; color: var(--ink); }
    .pg-maze-row { display: flex; gap: 22px; align-items: flex-start; }
    .pg-maze-row .maze-info { margin-top: 0; }`;
  const st = document.createElement("style"); st.textContent = css; document.head.appendChild(st);

  function init() {
    document.querySelectorAll(".occupancy").forEach((d) => new Occupancy(d));
    document.querySelectorAll(".pg-ascent").forEach((d) => new Ascent(d));
    document.querySelectorAll(".pg-lab").forEach((d) => new Lab(d));
  }
  window.RLPG = { analyse, sampleEpisode, LAB, makeLab, rng };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
