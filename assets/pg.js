/* =====================================================================
   Policy-gradient demos in the two-exit maze (Lecture 6). Needs maze.js.

   1) Occupancy measure of a fixed policy
      <div class="occupancy" data-policy="uniform" data-gamma="0.9" data-size="360"
           data-controls="policy,gamma,sample"></div>
      policies: uniform | route (→ big exit, ε-noise) | toSmall (→ small exit, ε-noise)
      Shows d^π(s) = Σ_a d^π(s,a) as a heat map, π as arrows, the mass absorbed at the exits,
      and the check  ⟨d^π, r⟩ / (1−γ) = V^π(s_1).  "+N episodes" replaces the exact d^π by a
      Monte Carlo estimate from N sampled episodes (discounted visit counts).

   2) Exact policy-gradient ascent, tabular softmax  π_θ(a|s) ∝ exp θ_{s,a},  θ_0 = 0 (uniform)
      <div class="pg-ascent" data-alpha="1" data-gamma="0.9" data-size="330"></div>
      Uses the PG theorem in closed form:  ∂J/∂θ_{s,a} = η^π(s) π(a|s) (Q^π(s,a) − V^π(s)).

   3) The θ-plane lab: linear softmax with two features,
         φ_big(s,a)   = 1{a = shortest-route action to the big exit}
         φ_small(s,a) = 1{a = shortest-route action to the small exit}
         π_θ(a|s) ∝ exp(θ_big φ_big(s,a) + θ_small φ_small(s,a))
      <div class="pg-lab" data-mode="cloud|runs" data-theta="0.5,-0.5" data-alpha="0.2"></div>
      cloud: click to move θ; sample single-episode REINFORCE estimates, shown as θ + α ĝ;
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
  function makeGrid(host, size, vmax, digits = 2) {
    const div = document.createElement("div");
    div.className = "maze vertical";
    Object.assign(div.dataset, { values: "fixed", v: "0,0,0,0,0,0,0,0,0,0,0,0", policy: "uniform", arrows: "policy", size: String(size), vmax: String(vmax), digits: String(digits), labels: "ids", controls: "" });
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
        button(bar3, "+10 episodes", () => this.addSamples(10));
        button(bar3, "+100 episodes", () => this.addSamples(100));
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
        `<div>${est ? `<b>estimated</b> from $N = ${this.N}$ episodes &nbsp;·&nbsp; $\\|\\hat d - d^\\pi\\|_1 = ${fmt(l1, 3)}$` : "<b>exact</b> $d^\\pi$ (needs $P$)"}</div>` +
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
  const LAB = (() => {
    const m = M.buildModel(0, "two-exit");
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
    // single-episode REINFORCE estimate.  kind: none | const | value
    function estimate(ex, rand, kind) {
      const { pi, A } = ex, ep = sampleEpisode(m, pi, rand), T = ep.length;
      const Gt = new Array(T); let acc = 0; for (let t = T - 1; t >= 0; t--) { acc = ep[t].r + g * acc; Gt[t] = acc; }
      const out = [0, 0];
      ep.forEach((x, t) => {
        // none:  γ^t G_t                (reward-to-go; = R(τ) here since all reward comes at the exit)
        // const: R(τ) − J(θ)            (classic "average return" baseline, same weight for every t)
        // value: γ^t (G_t − V^π(S_t))   (state-dependent baseline → advantage estimate)
        const w = kind === "const" ? Gt[0] - A.J : kind === "value" ? Math.pow(g, t) * (Gt[t] - A.V[x.s]) : Math.pow(g, t) * Gt[t];
        const sc = score(pi, x.s, x.a); out[0] += w * sc[0]; out[1] += w * sc[1];
      });
      return { g: out, T, exit: ep[T - 1].done ? ep[T - 1].exit : null };
    }
    return { m, policy, exact, estimate, J: (th) => analyse(m, policy(th), g).J };
  })();

  const KIND_LABEL = { none: "no baseline", const: "b = J(θ) (constant)", value: "b(s) = V^π(s)" };
  const KIND_COLOR = { none: "#b5541c", const: "#8a6a00", value: "#2b5b84" };

  class Lab {
    constructor(root) {
      const ds = root.dataset;
      this.root = root; this.mode = ds.mode || "cloud"; this.uid = "l" + (Lab.count = (Lab.count || 0) + 1);
      this.th0 = (ds.theta || "0.5,-0.5").split(",").map(Number); this.th = this.th0.slice();
      this.alpha = parseFloat(ds.alpha || "0.2"); this.kind = ds.kind || "none";
      this.lo = -3; this.hi = 7; this.S = parseFloat(ds.size || "470");
      this.seed = 7; this.rand = rng(this.seed); this.samples = []; this.runs = []; this.bars = null;
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
      this.static = ds.static === "1";
      const side = document.createElement("div"); side.className = "maze-side"; side.style.minWidth = "0"; if (this.static) side.style.display = "none"; root.appendChild(side);
      const bar = document.createElement("div"); bar.className = "maze-controls"; side.appendChild(bar);
      this.kbtn = {};
      ["none", "const", "value"].forEach((k) => { this.kbtn[k] = button(bar, KIND_LABEL[k], () => { this.kind = k; this.samples = []; this.runs = []; this.drawOverlay(); this.updateInfo(); }); });
      const bar2 = document.createElement("div"); bar2.className = "maze-controls"; side.appendChild(bar2);
      if (this.mode === "cloud") {
        button(bar2, "sample 1", () => this.sample(1));
        button(bar2, "sample 20", () => this.sample(20));
        button(bar2, "clear", () => { this.samples = []; this.drawOverlay(); this.updateInfo(); });
        button(bar2, "compare estimators", () => this.compare());
      } else {
        this.N = parseInt(ds.batch || "1", 10);
        this.nbtn = {};
        [1, 10].forEach((N) => { this.nbtn[N] = button(bar2, `${N} episode${N > 1 ? "s" : ""} / update`, () => { this.N = N; this.updateInfo(); }); });
        button(bar2, "run ×10", () => this.runMany(10));
        button(bar2, "clear", () => { this.runs = []; this.drawOverlay(); this.updateInfo(); });
      }
      const bar3 = document.createElement("div"); bar3.className = "maze-controls"; side.appendChild(bar3);
      slider(bar3, "α", 0.05, 1, 0.05, this.alpha.toFixed(2), (v) => { this.alpha = v; this.drawOverlay(); });
      this.info = document.createElement("div"); this.info.className = "maze-info"; side.appendChild(this.info);
      this.barHost = document.createElement("div"); side.appendChild(this.barHost);
      this.drawLandscape(); this.exactPath(); this.drawOverlay(); this.updateInfo();
    }
    toPx(t) { return (t - this.lo) / (this.hi - this.lo) * this.S; }
    toTh(px) { return this.lo + px / this.S * (this.hi - this.lo); }
    drawLandscape() {
      const ctx = this.canvas.getContext("2d"), R = 120, W = this.canvas.width, img = ctx.createImageData(R, R);
      for (let i = 0; i < R; i++) for (let j = 0; j < R; j++) {
        const tb = this.lo + (j + 0.5) / R * (this.hi - this.lo), ts = this.hi - (i + 0.5) / R * (this.hi - this.lo);
        const v = LAB.J([tb, ts]), lev = Math.min(14, Math.floor(v / 2.95 * 14 + 1e-9)) / 14, c = M.heat(0.04 + 0.96 * lev).match(/\d+/g).map(Number), k = 4 * (i * R + j);
        img.data[k] = c[0]; img.data[k + 1] = c[1]; img.data[k + 2] = c[2]; img.data[k + 3] = 255;
      }
      const off = document.createElement("canvas"); off.width = R; off.height = R; off.getContext("2d").putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true; ctx.drawImage(off, 0, 0, W, W);
    }
    exactPath() { // exact gradient ascent from θ0 (small steps) = the "noise-free" path
      let th = this.th0.slice(); this.path = [th.slice()];
      for (let k = 0; k < 400; k++) { const G = LAB.exact(th).grad; th = [th[0] + 0.1 * G[0], th[1] + 0.1 * G[1]]; this.path.push(th.slice()); if (th[0] > this.hi) break; }
    }
    sample(k) {
      const ex = LAB.exact(this.th);
      for (let i = 0; i < k; i++) this.samples.push(LAB.estimate(ex, this.rand, this.kind).g);
      this.drawOverlay(); this.updateInfo();
    }
    compare() {
      const ex = LAB.exact(this.th), r = rng(11), K = 2000, res = {};
      ["none", "const", "value"].forEach((kind) => {
        let mse = 0; for (let i = 0; i < K; i++) { const e = LAB.estimate(ex, r, kind).g; mse += Math.pow(e[0] - ex.grad[0], 2) + Math.pow(e[1] - ex.grad[1], 2); }
        res[kind] = mse / K;
      });
      this.bars = { res, norm: ex.grad[0] ** 2 + ex.grad[1] ** 2 }; this.updateInfo();
    }
    runMany(R) {
      const iters = Math.round(150 / Math.sqrt(this.N));
      for (let r = 0; r < R; r++) {
        const rand = rng(1000 + this.runs.length + 97 * r); let th = this.th0.slice(); const path = [th.slice()];
        for (let k = 0; k < iters; k++) {
          const ex = LAB.exact(th); let G = [0, 0];
          for (let i = 0; i < this.N; i++) { const e = LAB.estimate(ex, rand, this.kind).g; G[0] += e[0] / this.N; G[1] += e[1] / this.N; }
          th = [Math.max(this.lo, Math.min(this.hi + 2, th[0] + this.alpha * G[0])), Math.max(this.lo, Math.min(this.hi + 2, th[1] + this.alpha * G[1]))];
          path.push(th.slice());
        }
        this.runs.push({ path, kind: this.kind, N: this.N, J: LAB.J(th) });
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
      s += `<text x="${this.toPx(-2.6)}" y="${S - this.toPx(5.6)}" font-size="16" fill="#1b1d1a">small exit: J ≈ 0.9</text>`;
      s += `<text x="${this.toPx(6.8)}" y="${S - this.toPx(-2.4)}" font-size="16" text-anchor="end" fill="#fff" font-weight="700">big exit: J → 2.95</text>`;
      s += `<rect x="0" y="0" width="${S}" height="${S}" fill="none" stroke="#1b1d1a" stroke-width="1.5"/>`;
      // exact gradient-ascent path from θ0
      s += `<polyline points="${this.path.map((t) => P(t).join(",")).join(" ")}" fill="none" stroke="#1b1d1a" stroke-width="2.5" stroke-dasharray="7 5" opacity=".8"/>`;
      if (this.mode === "runs") {
        this.runs.forEach((r) => {
          const pts = r.path.map(P); const c = r.J < 1.5 ? "#b5541c" : "#ffffff";
          s += `<polyline clip-path="url(#pgl-clip-${this.uid})" points="${pts.map((p) => p.join(",")).join(" ")}" fill="none" stroke="${KIND_COLOR[r.kind]}" stroke-width="1.6" opacity=".75"/>`;
          const e = pts[pts.length - 1]; s += `<circle cx="${Math.max(0, Math.min(S, e[0]))}" cy="${Math.max(0, Math.min(S, e[1]))}" r="6" fill="${c}" stroke="#1b1d1a" stroke-width="1.5"/>`;
        });
        const p0 = P(this.th0); s += `<circle cx="${p0[0]}" cy="${p0[1]}" r="7" fill="#fff" stroke="#1b1d1a" stroke-width="2.5"/><text x="${p0[0] + 10}" y="${p0[1] + 22}" font-size="16" font-weight="700" fill="#1b1d1a">θ₀</text>`;
      } else {
        const ex = LAB.exact(this.th), p = P(this.th);
        this.samples.forEach((g) => { const q = P([this.th[0] + this.alpha * g[0], this.th[1] + this.alpha * g[1]]); s += `<g clip-path="url(#pgl-clip-${this.uid})"><line x1="${p[0]}" y1="${p[1]}" x2="${q[0]}" y2="${q[1]}" stroke="${KIND_COLOR[this.kind]}" stroke-width="1.3" opacity=".55"/><circle cx="${q[0]}" cy="${q[1]}" r="3.2" fill="${KIND_COLOR[this.kind]}"/></g>`; });
        if (this.samples.length > 1) {
          const mg = [0, 1].map((i) => this.samples.reduce((t, g) => t + g[i], 0) / this.samples.length), q = P([this.th[0] + this.alpha * mg[0], this.th[1] + this.alpha * mg[1]]);
          s += `<circle cx="${q[0]}" cy="${q[1]}" r="7" fill="none" stroke="${KIND_COLOR[this.kind]}" stroke-width="3"/>`;
        }
        const q = P([this.th[0] + this.alpha * ex.grad[0], this.th[1] + this.alpha * ex.grad[1]]);
        s += `<line x1="${p[0]}" y1="${p[1]}" x2="${q[0]}" y2="${q[1]}" stroke="#1b1d1a" stroke-width="4" marker-end="url(#pgl-ah)"/>`;
        s += `<circle cx="${p[0]}" cy="${p[1]}" r="6" fill="#fff" stroke="#1b1d1a" stroke-width="2.5"/>`;
      }
      this.svg.innerHTML = s;
    }
    updateInfo() {
      Object.entries(this.kbtn).forEach(([k, b]) => b.classList.toggle("on", k === this.kind));
      if (this.nbtn) Object.entries(this.nbtn).forEach(([k, b]) => b.classList.toggle("on", +k === this.N));
      const ex = LAB.exact(this.th);
      let h;
      if (this.mode === "cloud") {
        h = `<div>$\\theta = (${fmt(this.th[0])}, ${fmt(this.th[1])})$ &nbsp; $J(\\theta) = ${fmt(ex.A.J, 3)}$</div>` +
          `<div>$\\nabla J(\\theta) = (${fmt(ex.grad[0], 3)}, ${fmt(ex.grad[1], 3)})$ <span class="muted">(black arrow, ×α)</span></div>`;
        if (this.samples.length) {
          const K = this.samples.length, mg = [0, 1].map((i) => this.samples.reduce((t, g) => t + g[i], 0) / K);
          const cos = this.samples.reduce((t, g) => { const ng = Math.hypot(g[0], g[1]), nG = Math.hypot(ex.grad[0], ex.grad[1]); return t + (ng > 1e-12 ? (g[0] * ex.grad[0] + g[1] * ex.grad[1]) / (ng * nG) : 0); }, 0) / K;
          h += `<div>${K} estimates · mean $= (${fmt(mg[0], 3)}, ${fmt(mg[1], 3)})$ <span class="muted">(ring)</span></div><div>average $\\cos(\\hat g, \\nabla J) = ${fmt(cos, 2)}$</div>`;
        } else h += `<div class="muted">Click the plane to move θ. Each dot: θ + α ĝ for one episode.</div>`;
      } else {
        const done = this.runs.length, trapped = this.runs.filter((r) => r.J < 1.5).length;
        h = `<div>REINFORCE from $\\theta_0 = (${fmt(this.th0[0], 1)}, ${fmt(this.th0[1], 1)})$, ${Math.round(150 / Math.sqrt(this.N))} updates per run</div>` +
          `<div class="muted">dashed: exact gradient ascent (the "noise-free" path)</div>` +
          (done ? `<div>${done} runs · <b class="orange">${trapped}</b> ended at the small exit <span class="muted">(orange dots)</span></div>` : "");
      }
      this.info.innerHTML = h; tex(this.info);
      if (this.bars) {
        const W = 520, rowH = 34, mx = Math.max(...Object.values(this.bars.res));
        let s = `<div class="small" style="margin-top:8px">$\\E\\|\\hat g - \\nabla J\\|^2$ over 2000 episodes at this θ <span class="muted">($\\|\\nabla J\\|^2 = ${fmt(this.bars.norm, 3)}$)</span></div><svg viewBox="0 0 ${W} ${3 * rowH + 6}" width="${W}" height="${3 * rowH + 6}" class="learn-svg">`;
        ["none", "const", "value"].forEach((k, i) => {
          const v = this.bars.res[k], w = (W - 260) * v / mx;
          s += `<text x="0" y="${i * rowH + 22}" font-size="16" fill="#1b1d1a">${KIND_LABEL[k]}</text><rect x="190" y="${i * rowH + 6}" width="${w}" height="22" rx="4" fill="${KIND_COLOR[k]}"/><text x="${196 + w}" y="${i * rowH + 23}" font-size="15" fill="#4a4f47">${fmt(v, 2)}</text>`;
        });
        this.barHost.innerHTML = s + "</svg>"; tex(this.barHost);
      } else this.barHost.innerHTML = "";
    }
  }

  const css = `.pg-row { display:flex; gap:56px; align-items:flex-start; font-family: var(--font-text); }
    .pg-row .maze-side { flex: 1; }
    .pg-row .maze-info > div { margin: 2px 0; }
    .pg-plot { margin: 4px 0 0 56px; }`;
  const st = document.createElement("style"); st.textContent = css; document.head.appendChild(st);

  function init() {
    document.querySelectorAll(".occupancy").forEach((d) => new Occupancy(d));
    document.querySelectorAll(".pg-ascent").forEach((d) => new Ascent(d));
    document.querySelectorAll(".pg-lab").forEach((d) => new Lab(d));
  }
  window.RLPG = { analyse, sampleEpisode, LAB, rng };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
