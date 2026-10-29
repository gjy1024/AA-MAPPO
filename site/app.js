"use strict";

// No remote services or dependencies: imported files stay in this browser tab.
(() => {
  const $ = (id) => document.getElementById(id);
  const COLORS = ["#247c9b", "#42836e", "#c7854d", "#8d6a9f", "#b45b60"];
  const PAGE_NAMES = {overview: "论文主页", playground: "交互演示", method: "方法框架", evidence: "实验结果", reproduce: "代码与复现"};
  const state = {run: null, frame: 0, selected: 0, playing: false, lastTime: 0, elapsed: 0, manifest: null, legacy: null, pilot: null, imported: false, requestVersion: 0, datasetPath: "builtin"};
  const number = (v, digits = 2) => Number.isFinite(v) ? (Math.abs(v) > 1e5 || (v !== 0 && Math.abs(v) < .001) ? v.toExponential(2) : v.toFixed(digits)) : "未记录";
  const finite = (x) => typeof x === "number" && Number.isFinite(x);
  let toastTimer;

  function toast(message) {
    $("toast").textContent = message; $("toast").hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { $("toast").hidden = true; }, 4200);
  }
  function route() {
    const page = location.hash.slice(1) in PAGE_NAMES ? location.hash.slice(1) : "overview";
    for (const name of Object.keys(PAGE_NAMES)) {
      $("page-" + name).hidden = name !== page;
      $("page-" + name).classList.toggle("active", name === page);
    }
    document.querySelectorAll("[data-page]").forEach((link) => {
      const active = link.dataset.page === page;
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
    });
    $("breadcrumb-name").textContent = PAGE_NAMES[page];
    document.title = PAGE_NAMES[page] + " · AA-MAPPO 论文项目";
    window.scrollTo(0, 0);
    if (page !== "playground") stop();
  }
  function safePoint(p) { return Array.isArray(p) && p.length === 2 && p.every(finite); }
  function validateRun(data) {
    if (!data || typeof data !== "object" || Array.isArray(data) || data.schema_version !== "1.0" || data.kind !== "evaluation") throw Error("需要 schema_version=1.0、kind=evaluation 的评估文件。");
    if (!data.provenance || typeof data.provenance.policy !== "string" || data.provenance.policy.length > 250) throw Error("缺少有效的策略来源 provenance.policy。");
    if (!data.config || !finite(data.config.map_size) || data.config.map_size <= 0 || data.config.map_size > 1e7) throw Error("config.map_size 必须为有限正数。");
    if (!Array.isArray(data.users) || !data.users.length || data.users.length > 100 || !data.users.every(safePoint)) throw Error("users 必须是1至100个二维坐标。");
    if (!Array.isArray(data.trajectory) || data.trajectory.length < 2 || data.trajectory.length > 20000) throw Error("需要2至20000个轨迹帧。");
    const agents = data.trajectory[0]?.positions?.length;
    if (!agents || agents > 64) throw Error("无人机数量须在1至64之间。");
    let previous = -Infinity;
    data.trajectory.forEach((frame, i) => {
      if (!frame || typeof frame !== "object" || !finite(frame.time) || frame.time < 0 || frame.time <= previous) throw Error("轨迹时间必须严格递增且非负。");
      previous = frame.time;
      if (!Number.isInteger(frame.step) || frame.step < 0) throw Error("step 必须为非负整数。");
      if (!Array.isArray(frame.positions) || frame.positions.length !== agents || !frame.positions.every(safePoint)) throw Error("每帧无人机坐标数量须一致，且坐标为有限数值。");
      if (frame.velocities !== undefined && (!Array.isArray(frame.velocities) || frame.velocities.length !== agents || !frame.velocities.every(safePoint))) throw Error("velocities 必须与无人机数量一致。");
      if (!frame.metrics || typeof frame.metrics !== "object" || Array.isArray(frame.metrics)) throw Error("每帧需要 metrics 对象。");
      for (const value of Object.values(frame.metrics)) if (!finite(value)) throw Error("metrics 中只能包含有限数值。");
      if (i > 0 && frame.step <= data.trajectory[i - 1].step) throw Error("step 必须严格递增。");
    });
    return data;
  }
  function makeTeachingDemo() {
    const positions = [[12, 25], [19, 81], [89, 73]], velocity = [[0, 0], [0, 0], [0, 0]], users = [[37, 65], [73, 38]];
    const trajectory = [], dt = .2;
    for (let step = 0; step <= 180; step++) {
      let communication = 0, sensing_proxy = 0, minimum = Infinity, speed_cost = 0;
      positions.forEach((p, i) => {
        users.forEach((u) => { const d2 = (p[0] - u[0]) ** 2 + (p[1] - u[1]) ** 2 + 100; communication += Math.log2(1 + 100 / d2); sensing_proxy += 100 / d2 ** 2; });
        speed_cost += .01 * Math.hypot(...velocity[i]);
        positions.slice(i + 1).forEach((q) => { minimum = Math.min(minimum, Math.hypot(p[0] - q[0], p[1] - q[1])); });
      });
      trajectory.push({step, time: step * dt, positions: positions.map(p => [...p]), velocities: velocity.map(v => [...v]), metrics: {communication, sensing_proxy, min_distance: minimum, speed_cost, reward: communication + 100 * sensing_proxy - speed_cost - (minimum < 2 ? 50 : 0)}});
      positions.forEach((p, i) => {
        const target = users[i % users.length];
        const a = p.map((x, k) => Math.tanh(.06 * (target[k] - x) - .6 * velocity[i][k]));
        const norm = Math.max(1, Math.hypot(...a));
        const next = velocity[i].map((v, k) => v + 2 * a[k] / norm * dt);
        const cap = Math.max(1, Math.hypot(...next) / 5);
        p.forEach((x, k) => { const v = next[k] / cap; p[k] = Math.max(0, Math.min(100, x + .5 * (velocity[i][k] + v) * dt)); velocity[i][k] = v; });
      });
    }
    return {schema_version: "1.0", kind: "evaluation", provenance: {status: "illustrative", policy: "rule-based teaching controller", seed: 0, checkpoint: null, notes: "浏览器规则控制教学示例；不是训练策略或原论文结果。"}, config: {map_size: 100, safe_distance: 2, dt}, users, trajectory, summary: {independent_training_runs: 0}};
  }
  const teachingDemo = makeTeachingDemo();
  function sourceLabel(run, imported = false) {
    if (imported) return "用户导入 · 来源待核实";
    if (run.provenance.status === "illustrative") return "规则控制 · 教学演示";
    if (run.provenance.status === "maintenance-smoke") return "维护版 · 短训练评估";
    if (run.provenance.status === "maintenance-pilot") return "维护版 · 探索性训练评估";
    return "归档评估 · 按来源记录";
  }
  function setRun(run, imported = false) {
    validateRun(run); stop(); state.run = run; state.imported = imported; state.frame = 0; state.selected = 0;
    $("timeline").max = run.trajectory.length - 1; $("timeline").value = 0;
    $("run-title").textContent = sourceLabel(run, imported);
    $("run-description").textContent = imported ? "文件仅在本地浏览器处理。来源字段由文件提供，结构校验不构成科学结果认证。" : run.provenance.status === "illustrative" ? "规则控制教学示例，用于解释场景与指标；不是训练后的AA-MAPPO策略，也不是原论文实验结果。" : run.provenance.notes || "按所示来源查看该次运行。";
    $("run-badge").textContent = imported ? "导入数据" : run.provenance.status === "illustrative" ? "教学示例" : run.provenance.status === "maintenance-pilot" ? "探索性评估" : "流程验证";
    $("run-badge").className = "badge " + (run.provenance.status === "illustrative" || imported ? "amber" : "blue");
    const selector = $("uav-selector"); selector.replaceChildren();
    run.trajectory[0].positions.forEach((_, i) => {
      const button = document.createElement("button"); button.textContent = `UAV ${i + 1}`; button.type = "button";
      button.addEventListener("click", () => { state.selected = i; renderFrame(); }); selector.append(button);
    });
    if (imported) { state.requestVersion++; state.datasetPath = ""; $("dataset-select").value = ""; $("dataset-feedback").textContent = "正在查看本地导入文件，来源仍需核实"; }
    const legend = $("scene-legend"); legend.replaceChildren();
    run.trajectory[0].positions.forEach((_, i) => { const label = document.createElement("span"), dot = document.createElement("i"); dot.style.setProperty("--legend", COLORS[i % COLORS.length]); label.append(dot, document.createTextNode(`UAV ${String(i + 1).padStart(2, "0")}`)); legend.append(label); });
    const userLabel = document.createElement("span"), diamond = document.createElement("i"); diamond.className = "legend-diamond"; userLabel.append(diamond, document.createTextNode("地面用户")); legend.append(userLabel);
    renderFrame(); renderEvaluationChart(); renderProvenance();
  }
  function stop() { state.playing = false; $("play-button").textContent = "▶"; $("play-button").setAttribute("aria-label", "播放轨迹"); }
  function renderFrame() {
    const run = state.run, frame = run.trajectory[state.frame], size = run.config.map_size;
    const left = 76, top = 30, width = 488, height = 488, x = v => left + v / size * width, y = v => top + height - v / size * height;
    let svg = '<rect x="76" y="30" width="488" height="488" fill="#fbfcf9" stroke="#d5e0d8"/>';
    for (let i = 0; i <= 5; i++) {
      const px = left + i * width / 5, py = top + i * height / 5;
      svg += `<path d="M${px} ${top}V${top + height}M${left} ${py}H${left + width}" stroke="#e2e9e3" stroke-width="1"/><text x="${px}" y="541" text-anchor="middle" font-size="12" fill="#7c8d81">${number(i * size / 5, 0)}</text><text x="61" y="${py + 4}" text-anchor="end" font-size="12" fill="#7c8d81">${number(size - i * size / 5, 0)}</text>`;
    }
    svg += '<text x="320" y="570" text-anchor="middle" font-size="13" fill="#667b70">水平位置 X / m</text><text x="21" y="274" text-anchor="middle" transform="rotate(-90 21 274)" font-size="13" fill="#667b70">水平位置 Y / m</text><text x="320" y="595" text-anchor="middle" font-size="9" fill="#93a18f">固定高度平面 · X、Y 等比例 · 终点标记来自完整回放记录</text>';
    run.users.forEach((p, i) => { svg += `<path d="M${x(p[0])} ${y(p[1]) - 7}l7 7-7 7-7-7Z" fill="#fff" stroke="#536c5d" stroke-width="1.7"/><text x="${x(p[0]) + 11}" y="${y(p[1]) + 4}" font-size="11" fill="#536c5d">UE ${i + 1}</text>`; });
    frame.positions.forEach((p, i) => {
      const color = COLORS[i % COLORS.length], cx = x(p[0]), cy = y(p[1]);
      const start = run.trajectory[0].positions[i], finish = run.trajectory.at(-1).positions[i];
      if ($("show-trails").checked) {
        const stride = Math.max(1, Math.ceil((state.frame + 1) / 500));
        const points = run.trajectory.slice(0, state.frame + 1).filter((_, index) => index % stride === 0 || index === state.frame).map(f => `${x(f.positions[i][0])},${y(f.positions[i][1])}`).join(" ");
        svg += `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="2" stroke-opacity=".6"/>`;
      }
      svg += `<circle cx="${x(start[0])}" cy="${y(start[1])}" r="5" fill="#fff" stroke="${color}" stroke-width="1.7"/><text x="${x(start[0]) + 9}" y="${y(start[1]) + 15}" fill="${color}" font-size="9">S${i + 1}</text><path d="M${x(finish[0])} ${y(finish[1]) - 7}l6 11h-12Z" fill="${color}" fill-opacity=".18" stroke="${color}" stroke-width="1.5"/>`;
      if ($("show-safe").checked && finite(run.config.safe_distance)) svg += `<ellipse cx="${cx}" cy="${cy}" rx="${run.config.safe_distance / 2 / size * width}" ry="${run.config.safe_distance / 2 / size * height}" fill="${color}" fill-opacity=".08" stroke="${color}" stroke-opacity=".35"/>`;
      svg += `<g transform="translate(${cx} ${cy})" data-uav="${i}" tabindex="0" role="button" aria-label="选取无人机${i + 1}" style="cursor:pointer" fill="${color}" stroke="${color}"><circle r="20" fill="transparent" stroke="${i === state.selected ? color : 'transparent'}" stroke-opacity=".3"/><path d="M-7-7 7 7M-7 7 7-7" stroke-width="2"/><circle cx="-8" cy="-8" r="4" fill="white"/><circle cx="8" cy="-8" r="4" fill="white"/><circle cx="-8" cy="8" r="4" fill="white"/><circle cx="8" cy="8" r="4" fill="white"/><circle r="3"/><text x="15" y="-13" stroke="none" font-size="11">${i + 1}</text></g>`;
    });
    $("scene-map").innerHTML = svg;
    $("timeline").value = state.frame; $("step-label").textContent = "STEP " + String(frame.step).padStart(3, "0");
    $("time-label").textContent = `${number(frame.time, 1)} / ${number(run.trajectory.at(-1).time, 1)} s`;
    $("metric-communication").textContent = number(frame.metrics.communication, 3);
    $("metric-sensing").textContent = number(frame.metrics.sensing_proxy, 5);
    $("metric-distance").textContent = number(frame.metrics.min_distance, 2) + " m";
    $("metric-reward").textContent = number(frame.metrics.reward, 3);
    const p = frame.positions[state.selected], v = frame.velocities?.[state.selected];
    $("uav-position").textContent = `${number(p[0])} / ${number(p[1])} m`;
    $("uav-speed").textContent = v ? number(Math.hypot(...v), 3) + " m/s" : "未记录";
    const prev = run.trajectory[state.frame - 1];
    $("uav-acceleration").textContent = prev?.velocities && v ? number(Math.hypot(...v.map((value, k) => (value - prev.velocities[state.selected][k]) / (frame.time - prev.time))), 3) + " m/s²" : "—";
    $("safe-threshold").textContent = finite(run.config.safe_distance) ? number(run.config.safe_distance) + " m" : "未记录";
    [...$("uav-selector").children].forEach((button, i) => { button.classList.toggle("selected", i === state.selected); button.setAttribute("aria-pressed", String(i === state.selected)); });
  }
  function animate(now) {
    if (state.playing) {
      state.elapsed += Math.min((now - state.lastTime) / 1000, .1) * Number($("play-speed").value);
      const frames = state.run.trajectory;
      const oldFrame = state.frame;
      while (state.frame < frames.length - 1 && state.elapsed >= frames[state.frame + 1].time - frames[state.frame].time) {
        state.elapsed -= frames[state.frame + 1].time - frames[state.frame].time; state.frame++;
      }
      if (oldFrame !== state.frame) renderFrame(); if (state.frame === frames.length - 1) stop();
    }
    state.lastTime = now; requestAnimationFrame(animate);
  }
  function drawChart(svg, lines, xLabel, transform = false, yLabel = "指标值") {
    const all = lines.flatMap(line => line.points).filter(p => finite(p.x) && finite(p.y));
    if (!all.length) { svg.innerHTML = '<text x="380" y="160" text-anchor="middle" fill="#718477" font-size="16">没有该指标的记录</text>'; return; }
    const symlog = v => Math.sign(v) * Math.log10(1 + Math.abs(v));
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    all.forEach(point => { const y = transform ? symlog(point.y) : point.y; xmin = Math.min(xmin, point.x); xmax = Math.max(xmax, point.x); ymin = Math.min(ymin, y); ymax = Math.max(ymax, y); });
    if (xmax === xmin) xmax = xmin + 1;
    if (ymin === ymax) { ymin -= .5; ymax += .5; }
    const pad = (ymax - ymin) * .08; ymin -= pad; ymax += pad;
    const xx = value => 92 + (value - xmin) / (xmax - xmin) * 642, yy = value => 278 - ((transform ? symlog(value) : value) - ymin) / (ymax - ymin) * 246;
    let content = '';
    for (let i = 0; i <= 4; i++) {
      const yv = ymin + (ymax - ymin) * i / 4, py = 278 - 246 * i / 4, xv = xmin + (xmax - xmin) * i / 4;
      const raw = transform ? Math.sign(yv) * (10 ** Math.abs(yv) - 1) : yv;
      content += `<path d="M92 ${py}H734" stroke="#e2e8e2"/><text x="81" y="${py + 4}" text-anchor="end" fill="#718477" font-size="11">${number(raw, 2)}</text><text x="${92 + 642 * i / 4}" y="303" text-anchor="middle" fill="#718477" font-size="11">${number(xv, Number.isInteger(xmax) && xmax > 100 ? 0 : 1)}</text>`;
    }
    lines.forEach((line, i) => { const valid = line.points.filter(p => finite(p.x) && finite(p.y)), stride = Math.max(1, Math.ceil(valid.length / 3000)); const points = valid.filter((_, index) => index % stride === 0 || index === valid.length - 1).map(p => `${xx(p.x)},${yy(p.y)}`).join(' '); content += `<polyline points="${points}" fill="none" stroke="${line.color || COLORS[i % COLORS.length]}" stroke-width="1.8" stroke-linejoin="round"/>`; });
    content += `<text x="413" y="333" text-anchor="middle" fill="#607968" font-size="12">${xLabel}${transform ? ' · Y轴对称对数' : ''}</text><text x="20" y="156" transform="rotate(-90 20 156)" text-anchor="middle" fill="#607968" font-size="12">${yLabel}</text>`;
    svg.innerHTML = content;
  }
  function renderEvaluationChart() {
    if (!state.run) return;
    const key = $("chart-metric").value, windowSize = Number($("evaluation-smooth").value), points = state.run.trajectory.map(f => ({x: f.time, y: f.metrics[key]}));
    const displayed = points.map((point, i) => { const history = points.slice(Math.max(0, i - windowSize + 1), i + 1).filter(p => finite(p.y)); return {x: point.x, y: history.length ? history.reduce((sum, p) => sum + p.y, 0) / history.length : undefined}; });
    const labels = {communication: "通信频谱效率 / bit·s⁻¹·Hz⁻¹", sensing_proxy: "感知代理指标", min_distance: "最小机间距 / m", reward: "当前奖励", speed_cost: "速度代价"};
    drawChart($("metric-chart"), [{points: displayed}], "回放时间 / s", false, labels[key] || key);
    $("evaluation-display-label").textContent = (windowSize === 1 ? "原始逐步值" : `${windowSize}点向后移动平均`) + " · 无置信区间";
    $("chart-caption").textContent = sourceLabel(state.run, state.imported) + "；" + (windowSize === 1 ? "原始逐步记录" : `${windowSize}点移动平均，原始数据不变`) + "，不是多次独立训练的置信区间。";
  }
  function renderProvenance() {
    const p = state.run.provenance, items = {"数据类型": sourceLabel(state.run, state.imported), "策略": p.policy, "训练种子": p.training_seed ?? "未记录 / 不适用", "回放场景种子": p.seed ?? "未记录", "已训练环境步": p.trained_environment_steps ?? "不适用", "检查点": p.checkpoint || "无（规则控制）"};
    const list = $("provenance-list"); list.replaceChildren();
    Object.entries(items).forEach(([label, value]) => { const group = document.createElement("div"), dt = document.createElement("dt"), dd = document.createElement("dd"); dt.textContent = label; dd.textContent = String(value); group.append(dt, dd); list.append(group); });
  }
  function attention() {
    const ids = ["neighbor", "user", "irrelevant"], scores = ids.map(id => Number($("score-" + id).value)), max = Math.max(...scores), exp = scores.map(x => Math.exp(x - max)), sum = exp.reduce((a, b) => a + b);
    ids.forEach((id, i) => { $("score-" + id + "-value").textContent = scores[i].toFixed(1); $("weight-" + id).textContent = (exp[i] / sum * 100).toFixed(1) + "%"; $("weight-" + id + "-bar").style.width = (exp[i] / sum * 100) + "%"; });
  }
  function setMode(training) {
    $("critic-lane").hidden = !training; $("learning-loop").hidden = !training; $("execution-note").hidden = training;
    $("mode-training").classList.toggle("selected", training); $("mode-execution").classList.toggle("selected", !training);
    $("mode-training").setAttribute("aria-pressed", String(training)); $("mode-execution").setAttribute("aria-pressed", String(!training));
    $("mode-description").textContent = training ? "训练阶段：Actor 使用局部观测，Critic 使用全局状态。" : "执行阶段：每架无人机只使用其 Actor 路径，根据当前可用观测输出动作。";
  }
  async function getJSON(path) { const response = await fetch(path, {cache:"no-store"}); if (!response.ok) throw Error(`${path}: HTTP ${response.status}`); return response.json(); }
  function safeRelative(path) { return typeof path === 'string' && path.length > 0 && path.length < 400 && !path.startsWith('/') && !path.includes('..') && !path.includes('\\') && !/^[a-z][a-z0-9+.-]*:/i.test(path); }
  async function loadDataset(path, label, fromSelector = false) {
    const requestVersion = ++state.requestVersion;
    $("dataset-feedback").textContent = "正在读取运行档案…";
    try {
      const data = await getJSON(path); if (requestVersion !== state.requestVersion) return;
      setRun(data);
      state.datasetPath = path;
      $("dataset-select").value = path;
      $("dataset-feedback").textContent = label || "归档运行已载入";
      if (!fromSelector) { location.hash = "playground"; toast("已载入 " + label); }
    } catch (error) { if (requestVersion !== state.requestVersion) return; $("dataset-feedback").textContent = "读取失败，可尝试重新选择"; toast("载入失败：" + error.message); }
  }
  async function loadManifest() {
    try {
      const manifest = await getJSON("data/manifest.json"); if (!Array.isArray(manifest.runs)) throw Error("manifest.runs 缺失");
      state.manifest = manifest; $("run-count").textContent = `${manifest.runs.length} 条归档运行`;
      $("manifest-status").textContent = manifest.notes || "已载入带来源的运行档案；短训练与历史单次运行不能替代完整多种子实验。";
      $("evidence-status-badge").textContent = "已接入真实档案";
      const list = $("run-list"); list.replaceChildren();
      manifest.runs.forEach(run => {
        if (!safeRelative(run.path)) return;
        const row = document.createElement("div"); row.className = "run-entry";
        const text = document.createElement("div"), heading = document.createElement("strong"), note = document.createElement("p"), button = document.createElement("button");
        heading.textContent = run.label; note.textContent = run.description || run.status; text.append(heading, note);
        button.className = "button secondary"; button.textContent = "载入并回放 →";
        const option = document.createElement("option"); option.value = run.path; option.textContent = run.label; $("dataset-select").append(option);
        button.addEventListener("click", () => loadDataset(run.path, run.label));
        row.append(text, button); list.append(row);
      });
      $("dataset-select").value = state.datasetPath;
      const defaultRun = manifest.runs.find(run => run.id === manifest.default_run);
      if (defaultRun && state.requestVersion === 0 && !state.imported && safeRelative(defaultRun.path)) await loadDataset(defaultRun.path, defaultRun.label, true);
      const artifacts = $("artifact-list"); artifacts.replaceChildren();
      (manifest.artifacts || []).forEach(artifact => {
        const div = document.createElement("div"), heading = document.createElement("strong"), p = document.createElement("p");
        heading.textContent = artifact.title; p.textContent = artifact.description || artifact.status; div.append(heading, p);
        if (safeRelative(artifact.path)) { const a = document.createElement("a"); a.href = artifact.path; a.target = "_blank"; a.rel = "noopener noreferrer"; a.textContent = "查看来源文件 ↗"; div.append(a); }
        artifacts.append(div);
      });
    } catch (e) { $("manifest-status").textContent = location.protocol === "file:" ? "直接打开时受浏览器文件访问限制。教学交互仍可用；使用本地服务器可读取完整实验档案。" : "尚未读取到实验manifest；不会生成虚构的实证结果。"; }
  }
  function addHistoricalPanel(data) {
    if (data.kind !== "legacy-training" || !Array.isArray(data.runs) || !data.runs.length || data.runs.length > 20) return;
    if (data.runs.some(run => typeof run.id !== "string" || !run.series || typeof run.series !== "object" || Object.values(run.series).some(points => !Array.isArray(points) || points.length > 100000 || points.some(point => !point || !finite(point.step) || !finite(point.value))))) return;
    state.legacy = data;
    const panel = document.createElement("section"); panel.className = "legacy-panel"; panel.id = "legacy-panel";
    panel.innerHTML = '<div class="section-heading"><h2>恢复的历史训练记录</h2><span>四组算法 · 各一条run1</span></div><p class="legacy-warning">原始历史日志；种子与完整配置待确认。不同奖励定义的纵轴不能直接作为公平优劣比较。</p><div class="legacy-controls"><label>运行 <select id="legacy-run"></select></label><label>指标 <select id="legacy-metric"><option value="episode_reward">训练回报</option><option value="value_loss">Value loss（机间均值）</option><option value="ratio">策略比值（机间均值）</option><option value="entropy">策略熵（机间均值）</option><option value="actor_grad_norm">Actor梯度范数</option></select></label><label>显示 <select id="legacy-smooth"><option value="1">原始记录</option><option value="5">5点移动平均</option><option value="15">15点移动平均</option></select></label><label><input type="checkbox" id="legacy-log">对称对数坐标</label></div><svg id="legacy-chart" viewBox="0 0 760 350" role="img" aria-label="有来源的历史训练曲线"></svg><p id="legacy-note" class="chart-caption"></p><div class="figure-actions"><span>保留原始记录 · 不生成置信区间</span><button class="copy-button" id="export-legacy-svg">导出历史曲线 SVG ↓</button></div><details><summary>数据追溯与解释</summary><p id="legacy-source"></p><p>移动平均仅用于辅助阅读，不能代替原始数据。多个智能体诊断的均值也不等于多训练种子均值。</p><a href="data/legacy-training.json" download>下载规范化曲线 JSON ↘</a></details>';
    $("page-evidence").insertBefore(panel, $("page-evidence").querySelector(":scope > .section-heading"));
    const select = $("legacy-run"); data.runs.forEach(run => { const option = document.createElement("option"); option.value = run.id; option.textContent = run.label; select.append(option); });
    const all = document.createElement("option"); all.value = "__all"; all.textContent = "四组同图（仅诊断比较）"; select.append(all);
    const legend = document.createElement("div"); legend.id = "legacy-legend"; legend.className = "chart-legend"; $("legacy-chart").after(legend);
    ["legacy-run", "legacy-metric", "legacy-smooth", "legacy-log"].forEach(id => $(id).addEventListener("change", renderHistorical)); renderHistorical();
    $("export-legacy-svg").addEventListener("click", exportHistoricalSvg);
  }
  function renderHistorical() {
    const chosen = $("legacy-run").value, key = $("legacy-metric").value, windowSize = Number($("legacy-smooth").value);
    const runs = state.legacy.runs.filter(run => chosen === "__all" || run.id === chosen), lines = [];
    const legend = $("legacy-legend"); legend.replaceChildren();
    runs.forEach(run => {
      const points = run.series[key] || [], color = COLORS[state.legacy.runs.indexOf(run) % COLORS.length];
      const displayed = points.map((p, i) => ({x: p.step, y: points.slice(Math.max(0, i - windowSize + 1), i + 1).reduce((sum, point) => sum + point.value, 0) / Math.min(windowSize, i + 1)}));
      lines.push({points: displayed, color});
      const label = document.createElement("span"), swatch = document.createElement("i"); swatch.style.background = color; label.append(swatch, document.createTextNode(run.label)); legend.append(label);
    });
    const yLabels = {episode_reward: "历史训练回报", value_loss: "Value loss（机间均值）", ratio: "策略概率比（机间均值）", entropy: "策略熵（机间均值）", actor_grad_norm: "Actor 梯度范数"};
    drawChart($("legacy-chart"), lines, "训练环境步（按历史日志）", $("legacy-log").checked, yLabels[key] || key);
    const total = runs.reduce((sum, run) => sum + (run.series[key]?.length || 0), 0);
    $("legacy-note").textContent = `${runs.length}条历史运行、共${total}个记录点；${windowSize === 1 ? '原始数值（含异常峰值）' : windowSize + '点向后移动平均'}。历史run1不是多种子实验。` + (chosen === "__all" && key === "episode_reward" ? " 不同奖励定义的纵轴不支持公平性能排名。" : "");
    $("legacy-source").textContent = runs.map(run => `来源：${run.source?.path || '未提供'}；SHA-256：${run.source?.sha256 || '未提供'}；重复副本一致：${run.source?.copies_identical ? '是' : '否或未确认，原副本分别保留'}。`).join("\n");
  }
  function runMetadata() {
    const p = state.run.provenance;
    return [
      `数据类型：${sourceLabel(state.run, state.imported)}；策略：${p.policy}`,
      `训练种子：${p.training_seed ?? '未记录/不适用'}；场景种子：${p.seed ?? '未记录'}；训练环境步：${p.trained_environment_steps ?? p.training_steps ?? '未记录/不适用'}`,
      `检查点：${p.checkpoint || '无/未提供'}；数据结构：${state.run.schema_version}`
    ];
  }
  function exportSvg(svg, filename, title, metadata, legend = []) {
    const NS = "http://www.w3.org/2000/svg", view = svg.viewBox.baseVal;
    const width = view.width || 760, figureHeight = view.height || 350;
    const textLines = [];
    metadata.forEach(value => {
      let line = "", units = 0;
      for (const character of String(value)) {
        const weight = character.charCodeAt(0) > 255 ? 2 : 1;
        if (units + weight > (width - 48) / 4.9) { textLines.push(line); line = ""; units = 0; }
        line += character; units += weight;
      }
      if (line) textLines.push(line);
    });
    const top = 58 + textLines.length * 14, bottom = 58 + Math.ceil(legend.length / 3) * 19;
    const output = document.createElementNS(NS, "svg");
    output.setAttribute("xmlns", NS); output.setAttribute("width", width); output.setAttribute("height", figureHeight + top + bottom); output.setAttribute("viewBox", `0 0 ${width} ${figureHeight + top + bottom}`);
    output.setAttribute("font-family", "Arial, Microsoft YaHei, sans-serif");
    const rect = document.createElementNS(NS, "rect"); rect.setAttribute("width", "100%"); rect.setAttribute("height", "100%"); rect.setAttribute("fill", "#ffffff"); output.append(rect);
    const addText = (content, x, y, size, fill = "#6b7b70") => { const text = document.createElementNS(NS, "text"); text.setAttribute("x", x); text.setAttribute("y", y); text.setAttribute("font-size", size); text.setAttribute("fill", fill); text.textContent = content; output.append(text); };
    addText(title, 24, 28, 16, "#172c32");
    textLines.forEach((line, index) => addText(line, 24, 48 + index * 14, 9));
    const group = document.createElementNS(NS, "g"); group.setAttribute("transform", `translate(0 ${top})`);
    [...svg.children].forEach(child => group.append(child.cloneNode(true))); output.append(group);
    legend.forEach((item, index) => { const x = 28 + (index % 3) * (width - 56) / 3, y = top + figureHeight + 12 + Math.floor(index / 3) * 19; const swatch = document.createElementNS(NS, "line"); swatch.setAttribute("x1", x); swatch.setAttribute("x2", x + 15); swatch.setAttribute("y1", y - 3); swatch.setAttribute("y2", y - 3); swatch.setAttribute("stroke", item.color); swatch.setAttribute("stroke-width", "2"); output.append(swatch); addText(item.label, x + 21, y, 9); });
    addText("AA-MAPPO Research Workbench · 数据类型与来源见图注 · 未绘制置信区间", 24, top + figureHeight + bottom - 17, 8, "#879482");
    const content = '<?xml version="1.0" encoding="UTF-8"?>\n' + new XMLSerializer().serializeToString(output);
    const url = URL.createObjectURL(new Blob([content], {type: "image/svg+xml;charset=utf-8"})), anchor = document.createElement("a");
    anchor.href = url; anchor.download = filename; document.body.append(anchor); anchor.click(); anchor.remove(); setTimeout(() => URL.revokeObjectURL(url), 3000);
    toast("已导出带来源标记的SVG矢量图。");
  }
  function exportSceneSvg() {
    const frame = state.run.trajectory[state.frame], legend = frame.positions.map((_, i) => ({label: `UAV ${i + 1}`, color: COLORS[i % COLORS.length]}));
    legend.push({label: "○ 起点 / △ 整段终点", color: "#778b75"}, {label: "◇ 地面用户", color: "#536c5d"});
    exportSvg($("scene-map"), "aa-mappo-trajectory.svg", "无人机集群轨迹回放 · 等比例坐标", [...runMetadata(), `当前回放：step ${frame.step}，t=${number(frame.time, 1)} s；航迹按当前显示导出。`], legend);
  }
  function exportMetricSvg() {
    const label = $("chart-metric").selectedOptions[0].textContent, windowSize = Number($("evaluation-smooth").value);
    exportSvg($("metric-chart"), `aa-mappo-${$("chart-metric").value}.svg`, `逐步评估指标 · ${label}`, [...runMetadata(), `显示：${windowSize === 1 ? '原始记录' : windowSize + '点向后移动平均'}；单次评估，不是独立训练种子统计。`]);
  }
  function exportHistoricalSvg() {
    const chosen = $("legacy-run").value, runs = state.legacy.runs.filter(run => chosen === "__all" || run.id === chosen), key = $("legacy-metric").value;
    const windowSize = Number($("legacy-smooth").value), legend = runs.map(run => ({label: run.label, color: COLORS[state.legacy.runs.indexOf(run) % COLORS.length]}));
    const source = runs.map(run => `${run.label}：${run.source?.path || '来源路径未提供'}；SHA-256 ${run.source?.sha256 || '未提供'}`);
    exportSvg($("legacy-chart"), `aa-mappo-history-${key}.svg`, `历史单次训练 · ${$("legacy-metric").selectedOptions[0].textContent}`, ["数据类型：historical-single-run；每算法单条run1；种子与完整配置未确认；不对应维护版新架构。", `显示：${windowSize === 1 ? '原始记录' : windowSize + '点向后移动平均'}${$("legacy-log").checked ? '；Y轴对称对数' : ''}；不同奖励定义不可按回报直接排名。`, ...source], legend);
  }
  async function loadPilotComparison() {
    try {
      const data = await getJSON("data/pilot-comparison.json");
      if (data.schema_version !== "1.0" || data.kind !== "pilot-comparison" || !Array.isArray(data.rows) || !data.rows.length || data.rows.length > 40) throw Error("pilot-comparison 格式不受支持");
      if (data.rows.some(row => !row || typeof row.label !== "string" || !row.metrics || typeof row.metrics !== "object")) throw Error("对照行缺少名称或指标");
      state.pilot = data;
      const p = data.provenance || {};
      $("pilot-badge").textContent = "探索性训练 · 固定场景"; $("pilot-badge").className = "badge blue";
      $("pilot-note").textContent = p.notes || "同一组固定评估场景下的探索性结果；不作为完整多随机种子结论。";
      const table = $("pilot-table"); table.replaceChildren(); const thead = document.createElement("thead"), tr = document.createElement("tr");
      const definitions = [
        {key: "label", label: "策略 / 方法"}, {key: "training", label: "训练步 / 种子"},
        {key: "communication", label: "通信频谱效率", unit: "bit/s/Hz", aliases: ["mean_communication"]},
        {key: "sensing_proxy", label: "感知代理", unit: "代理量", aliases: ["mean_sensing_proxy"]},
        {key: "collision_episode_rate", label: "碰撞回合比例", unit: "%", aliases: ["collision_rate"]},
        {key: "boundary_episode_rate", label: "触边回合比例", unit: "%"},
        {key: "min_distance", label: "机间距指标", unit: "m · 口径见来源", aliases: ["mean_min_distance", "minimum_distance"]},
        {key: "mean_return", label: "平均回合回报", unit: "当前统一奖励", aliases: ["reward", "episode_return"]}
      ];
      definitions.forEach(column => { const th = document.createElement("th"); th.textContent = column.label; if (column.unit) { const small = document.createElement("small"); small.textContent = column.unit; th.append(small); } tr.append(th); }); thead.append(tr); table.append(thead);
      const tbody = document.createElement("tbody");
      data.rows.forEach(row => {
        const entry = document.createElement("tr");
        definitions.forEach(column => {
          const cell = document.createElement("td");
          if (column.key === "label") cell.textContent = row.label;
          else if (column.key === "training") cell.textContent = `${row.trained_environment_steps ?? row.training_steps ?? '—'} / ${row.training_seed ?? '—'}`;
          else {
            const key = [column.key, ...(column.aliases || [])].find(candidate => row.metrics[candidate] !== undefined);
            const stat = key === undefined ? undefined : row.metrics[key], value = typeof stat === "object" && stat !== null ? stat.mean : stat;
            cell.textContent = finite(value) ? number(column.key.endsWith("_rate") ? value * 100 : value, column.key === "sensing_proxy" ? 5 : 3) : "未记录";
          }
          entry.append(cell);
        }); tbody.append(entry);
      }); table.append(tbody);
      $("pilot-table-wrap").hidden = false; $("pilot-footer").hidden = false;
      const seeds = Array.isArray(p.evaluation_seeds) ? p.evaluation_seeds : [];
      $("pilot-source").textContent = `来源：data/pilot-comparison.json${seeds.length ? '；固定测试种子：' + seeds.join(', ') : ''}；均值不是独立训练重复的置信区间。`;
    } catch (error) { if (!String(error.message).includes("HTTP 404")) $("pilot-note").textContent = "探索性对照档案尚未可用；保留空白状态，不填入示例数字。"; }
  }
  async function addFigures() {
    try {
      const data = await getJSON("data/historical-figures.json");
      const figures = (data.figures || []).filter(f => safeRelative(f.path) && !f.source.includes("更改命令"));
      const details = document.createElement("details"); details.className = "figure-archive";
      const summary = document.createElement("summary"); summary.textContent = `查看历史截图档案（${figures.length}张）`; details.append(summary);
      const grid = document.createElement("div"); grid.className = "figure-grid";
      figures.forEach(f => { const figure = document.createElement("figure"), img = document.createElement("img"), caption = document.createElement("figcaption"), link = document.createElement("a"); img.src = f.path; img.alt = f.caption; img.loading = "lazy"; caption.textContent = f.caption; link.href = f.path; link.target = "_blank"; link.rel = "noopener"; link.append(img); figure.append(link, caption); grid.append(figure); });
      details.append(grid); $("page-evidence").append(details);
    } catch (_) { /* Optional archive. */ }
  }
  function download(data, filename) {
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: "application/json"}));
    const a = document.createElement("a"); a.href = url; a.download = filename; a.click(); setTimeout(() => URL.revokeObjectURL(url), 2000);
  }
  async function copy(text) {
    try { if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text); else { const field = document.createElement("textarea"); field.value = text; field.style.position = "fixed"; field.style.opacity = "0"; document.body.append(field); field.select(); const copied = document.execCommand("copy"); field.remove(); if (!copied) throw Error("clipboard"); } toast("已复制"); }
    catch (_) { toast("浏览器未允许复制。可直接选中代码复制。"); }
  }
  $("play-button").addEventListener("click", () => {
    if (state.playing) return stop();
    if (state.frame === state.run.trajectory.length - 1) state.frame = 0;
    state.playing = true; state.elapsed = 0; $("play-button").textContent = "Ⅱ"; $("play-button").setAttribute("aria-label", "暂停轨迹");
  });
  $("reset-button").addEventListener("click", () => { stop(); state.frame = 0; renderFrame(); });
  $("timeline").addEventListener("input", () => { stop(); state.frame = Number($("timeline").value); renderFrame(); });
  ["show-trails", "show-safe"].forEach(id => $(id).addEventListener("change", renderFrame));
  $("scene-map").addEventListener("click", e => { const target = e.target.closest("[data-uav]"); if (target) { state.selected = Number(target.dataset.uav); renderFrame(); } });
  $("scene-map").addEventListener("keydown", e => { if ((e.key === "Enter" || e.key === " ") && e.target.dataset.uav) { e.preventDefault(); state.selected = Number(e.target.dataset.uav); renderFrame(); } });
  $("chart-metric").addEventListener("change", renderEvaluationChart);
  $("evaluation-smooth").addEventListener("change", renderEvaluationChart);
  $("export-scene-svg").addEventListener("click", exportSceneSvg);
  $("export-metric-svg").addEventListener("click", exportMetricSvg);
  $("dataset-select").addEventListener("change", () => {
    const path = $("dataset-select").value;
    if (path === "builtin") { state.requestVersion++; state.datasetPath = "builtin"; setRun(teachingDemo); $("dataset-feedback").textContent = "内置教学演示，非训练策略"; }
    else if (safeRelative(path)) loadDataset(path, $("dataset-select").selectedOptions[0]?.textContent, true);
  });
  $("mode-training").addEventListener("click", () => setMode(true)); $("mode-execution").addEventListener("click", () => setMode(false));
  ["neighbor", "user", "irrelevant"].forEach(id => $("score-" + id).addEventListener("input", attention));
  ["import-run-button", "import-evidence-button"].forEach(id => $(id).addEventListener("click", () => $("run-file-input").click()));
  $("run-file-input").addEventListener("change", async e => {
    const file = e.target.files[0]; if (!file) return;
    try { if (file.size > 10 * 1024 * 1024) throw Error("文件超过10MB，请先裁剪到一个评估回合。"); setRun(JSON.parse(await file.text()), true); toast("已在浏览器本地载入；来源仍需核实。"); }
    catch (error) { toast("导入失败：" + error.message); }
    e.target.value = "";
  });
  $("export-run-button").addEventListener("click", () => download(state.run, "aa-mappo-evaluation.json"));
  $("download-example").addEventListener("click", () => download(teachingDemo, "aa-mappo-teaching-example.json"));
  $("restore-demo").addEventListener("click", () => { state.requestVersion++; state.datasetPath = "builtin"; setRun(teachingDemo); $("dataset-select").value = "builtin"; $("dataset-feedback").textContent = "内置教学演示，非训练策略"; });
  $("view-current-run").addEventListener("click", () => { location.hash = "playground"; });
  $("copy-schema").addEventListener("click", () => copy($("schema-preview").textContent));
  document.querySelectorAll("[data-copy]").forEach(button => button.addEventListener("click", () => copy(button.dataset.copy)));
  $("copy-citation").addEventListener("click", () => copy('@inproceedings{gu2026aamappo,\n  author={Jiayi Gu},\n  title={UAV Swarm Path Planning Based on Attention Mechanism and Reinforcement Learning in 6G Networks},\n  booktitle={2026 IEEE 4th International Conference on Control, Electronics and Computer Technology (ICCECT)},\n  year={2026}, pages={147--153},\n  doi={10.1109/ICCECT68671.2026.11565262}\n}'));
  window.addEventListener("hashchange", route);
  document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); });
  // Expose pure validation for lightweight browser-console regression checks.
  window.AAMAPPO = Object.freeze({validateRun, makeTeachingDemo});
  setRun(teachingDemo); attention(); route(); requestAnimationFrame(animate);
  loadManifest(); getJSON("data/legacy-training.json").then(addHistoricalPanel).catch(error => {
    const note = document.createElement("p"); note.id = "legacy-load-error"; note.className = "chart-caption";
    note.textContent = "历史训练曲线暂未载入，请刷新页面；直接打开文件时，请通过本地服务器读取档案。";
    $("page-evidence").querySelector(".evidence-status").after(note);
    console.warn("Historical training archive could not be loaded:", error);
  }); addFigures(); loadPilotComparison();
  getJSON("data/demo.json").then(data => { if (!state.imported && state.run === teachingDemo && state.requestVersion === 0) { setRun(data); state.datasetPath = "data/demo.json"; $("dataset-select").value = state.datasetPath; $("dataset-feedback").textContent = "规则控制教学演示（归档文件）"; } }).catch(() => {});
})();
