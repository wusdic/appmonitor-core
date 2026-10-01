# AppMonitor Core · 组织信息化业务系统画像平台

在**不在主机安装任何软件**的前提下（以旁路流量为主、主动探测为辅），为组织内每个业务系统
构建完整画像。画像由**四个联动的库**组成：

1. **原始指标库**（Raw Metrics）— 基于流量/主动探测可获取、按分类尽量全的原始指标，含**获取方式**。
2. **次生指标库**（Derived Metrics）— 对原始指标分析/组合派生出的新指标。
3. **行为库**（Behavior）— 两部分：**渐进画像内核**（P00–P15）不遍历所有用户与服务器，从行为事件出发，
   由粗到细地学到“一类行为”乃至“某一个 IP 的行为”的模式（谁、何时、访问什么、提交内容的约束、绑定、流程、群组），
   随时间与行为变化而更新，并输出业务系统视角与用户（群组）视角的中英文画像陈述；**统计引擎库**（B01–B30：
   精确预测分布、节拍不变的双粒度特征、保形校准、融合与证据 CUSUM、身份/归因/链接、类即实体、可回滚的治理、
   忠实反事实解释）作为画像的侧面与检测器，决策链两者共用。实体粒度到 **IP / 一类 IP**（不定位到人）。
4. **行为特征库**（Behavior Signatures）— 指标组合 → 语义的**预设库**，识别“用户在干什么”。

> 设计与实现（中文）：**[`docs/组织业务系统画像平台设计.md`](docs/组织业务系统画像平台设计.md)**。
> 行为库逐引擎规格、契约与评估记录（英文）：[`docs/lib3/`](docs/lib3/)，
> 其中 [`progressive.md`](docs/lib3/progressive.md) 是渐进画像内核的规格与实测，[`summary.md`](docs/lib3/summary.md) 是现状摘要，
> [`integration.md`](docs/lib3/integration.md) 是集成与各轮评估记录。

## 现状（2026-10-01）

- 60 个引擎已实现；默认注册表 `full` 注册 44 个（不含渐进内核），`full+progressive` 注册全部 60 个。严格模式下 0 异常。
- **渐进画像内核**（组织包 O，21 天、种子 0–2，`reports/progressive/`）：已端到端运行并学到路由级动作、执行者、时间窗、大小区间、
  载荷语法、流程、行为群组与否定陈述；需求示例中的审批与 17 点报告在三个种子上都还原。但验收**未通过**：第 14 天模式召回 0.18–0.21（目标 0.90），
  需求示例异常 19/30 检出（目标 0.95），≥ MEDIUM 误报 0.025–0.029/实体·日（目标 0.02），陈述置信度未随时间上升且未校准；
  内存对 IP 数、服务器数次线性（达标），每事件耗时超目标 10–20 倍。
- **统计引擎库**（A–E 包第四轮，`reports/eval_report.json`）：15 项门限中第 15 项（鲁棒性）通过，第 12、13 项 n/a，其余未通过；
  期限内威胁召回 0.71（目标 0.95），对照实体 FAR ≥ LOW 0.156、≥ MEDIUM 0.042（均达标），单包墙钟与实时 p95 远超 360 s / 80 ms 目标。
- 结果、原因与未决问题见设计文档库三 §3.8、§3.10、§3.11 与 `docs/lib3/progressive.md` §16。

## 架构一览

```
数据源(被动/主动) → 原始指标库(8+P00) → 次生指标库(7+P01) → 行为库(渐进内核 14 + 统计引擎 27) → 行为特征库(2)
                  全部落入统一 MetricStore，引擎只按“指标名/模型名”读写，彼此低耦合、可独立增删换
```

| 库 | 引擎（`backend/app/engines/`） |
|---|---|
| 原始 `raw/` | l2l3 · l4flow · http · tls · dns · active_probe · action_token · client_stack；渐进内核 P00 event_builder |
| 次生 `derived/` | aggregation · periodicity · trend · ratio · entropy · graph · session；渐进内核 P01 event_context |
| 行为 `behavior/` · 渐进画像内核 | P02 attr_registry · P03 conformity · P04 pattern_tree · P05 attr_select · P06 content_bounds · P07 payload_grammar · P08 binding · P09 time_window · P10 workflow · P11 who_groups · P12 system_profile · P13 facets · P14 views · P15 resource_governor |
| 行为 `behavior/` · 统计引擎库 | B01 feature_vector · B02 peer_group · B03 baseline · B04 likelihood · B05 common_mode · B06 multivariate · B07 rhythm · B08 novelty · B09 client_identity · B10 sequence · B11 timing · B12 beacon · B13 budget · B14 changepoint · B15 identity_model · B16 attribution · B17 entity_link · B18 class_monitor · B21 cross_system · B23 feedback · B24 calibration · B25 fusion · B26 risk · B27 incident · B28 governor · B29 explain · B30 portrait |
| 特征 `signature/` | rule_match · correlation |

注册模式（`backend/app/pipeline/build.py::build_registry(progressive=…)`）：`full`（默认，44 个，后端服务与 A–E 包使用）、
`full+progressive`（60 个）、`progressive_only`（18 个，伸缩实验）、`progressive_decision`（24 个 = 渐进内核 + B24–B29 决策链，包 O 实测使用）。
后端服务（下文）默认运行 `full`；设置 `APPMON_PROGRESSIVE=decision` 时在组织包 O 上运行渐进内核，其视图经 API v3 与前端“画像模式”页展示。
P2 引擎 B19 mixture、B20 session_profile、B22 action_embedding 有规格但未实现、未注册（B21 cross_system 已在第四轮实现）。

## 安装

需要 Python 3.11+。

```bash
python -m venv .venv
.venv/bin/pip install -r backend/requirements.txt pytest
# 可选：scripts/render_check.py 需要 playwright 与 Chromium
```

以下命令都在仓库根目录执行。单线程 BLAS 更快也更可复现（引擎做大量小矩阵运算），
脚本会自行设置 `OMP_NUM_THREADS` 等变量；手动运行时建议加 `OMP_NUM_THREADS=1`。

## 运行后端与前端

```bash
cd backend
OMP_NUM_THREADS=1 ../.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8099
```

- 启动时先用合成流量预热（默认计划 120 × 3600 s + 192 × 900 s，预热在当前墙钟处结束，单核约 2 分钟；预热的 900 s 阶段缺少某种日类型时会打印告警），然后后台线程以 60 s 窗口持续运行实时拍。
- 浏览器打开 `http://localhost:8099/`（自动跳转 `/app/`）。前端为零依赖原生 JS，十个视图：
  总览 / 实体画像 / 类画像 / 事件队列 / 系统视图 / 行为事件 / 行为特征库 / 指标库 / 引擎拓扑 / 评估报告。
- 无需真实流量：内置合成流量源生成 3 个业务系统（erp-prod、oa-portal、api-gateway）的多类实体，并注入演示威胁。
- 前端渲染自检（需先启动后端）：`.venv/bin/python scripts/render_check.py`（`APPMON_URL`、`OUT_DIR` 可改）。

环境变量：

| 变量 | 作用 |
|---|---|
| `APPMON_WARMUP_TICKS` | 设置后使用 v2 预热计划 N × 900 s；不设则为 v2.1 默认计划 |
| `APPMON_LIVE_PERIOD_S` | 实时拍之间的真实等待秒数（默认 3.0） |
| `APPMON_SIGNATURE_DIR` | 特征库 YAML 目录（默认 `data/signatures`） |
| `APPMON_FRONTEND_DIR` | 前端目录（默认 `frontend`） |
| `APPMON_PROGRESSIVE` | `decision` / `only` / `full`：后端改在组织包上运行渐进内核（`progressive_decision` / `progressive_only` / `full+progressive`），供“画像模式”页使用；不设则为经典 `full` |
| `APPMON_PACK` / `APPMON_PROGRESSIVE_DAYS` / `APPMON_SEED` | 渐进内核所用组织包（默认 `O`）、900 s 预热天数（默认 7，每天约 35 s）、种子（默认 0） |
| `APPMON_PROGRESSIVE_RUNS` | “精度随时间”实测曲线读取的评估运行目录（默认 `reports/progressive/runs`） |
| `APPMON_EVAL_REPORT` | `/api/eval/report` 读取的报告路径（默认依次查 `reports/`、`eval_out/`） |

## 冒烟测试：`scripts/smoke.py`

```bash
.venv/bin/python scripts/smoke.py --strict
.venv/bin/python scripts/smoke.py --plan 120x3600,192x900 --live 16 --strict
.venv/bin/python scripts/smoke.py --warmup 180          # v2 计划：180 × 900 s
```

构建完整注册表的 Runtime，按计划预热（训练态），再以 60 s 窗口跑 `--live` 个实时拍（默认 16，即 900 → 60 s 节拍切换）。
输出粒度模式与预热计划（预热的 900 s 阶段缺少某种日类型时会告警）、各引擎耗时、类路径、可分性、风险前 10、打开的事件与最近事件。
`--strict` 让引擎异常直接抛出；有引擎错误时退出码非 0。

## 评估：`scripts/evaluate.py`

```bash
# 完整矩阵（5 包 × 5 种子，多进程），报告写到 --out 目录
.venv/bin/python scripts/evaluate.py --packs A,B,C,D,E --seeds 0,1,2,3,4 --workers 4 --out reports
# 单包单种子
.venv/bin/python scripts/evaluate.py --packs A --seeds 0 --out eval_out
# 门限 12：配对的模拟分析员反馈运行
.venv/bin/python scripts/evaluate.py --packs A --seeds 0,1 --feedback
# 门限 13：逐个关闭引擎的消融运行（须写完整引擎名或类名；短名如 likelihood 匹配不到任何引擎）
.venv/bin/python scripts/evaluate.py --packs A --seeds 0 --ablate behavior.likelihood,behavior.class_monitor
```

其它参数：`--smoke`（同时跑 smoke 包，用于门限 14 的冒烟预算）、`--time-budget S`（单个包-种子的墙钟上限）。
每个工作进程运行并评分一个 (包, 种子)，输出 `eval_report.json`（唯一的精度数据来源）与自包含的 `eval_report.html`。
门限定义见 `docs/lib3/eval.md`，数据包与场景见 `docs/lib3/generator.md`。
单个包-种子在单核上约 12–22 分钟（主机空闲时 A 包约 740 s、B 包约 1315 s；第四轮 4 进程并行时最长 2018 s）。最新结果（第四轮）在 `reports/eval_report.{json,html}`，历史各轮在 `reports/*/`。

## 渐进画像内核：组织包 O 评估与伸缩实验

```bash
# 包 O（需求中的组织：综合部/财务部/销售部/研发池、OA 与财务等 6 个系统、21 天、900 s 拍），种子 0–2，三进程并行
.venv/bin/python scripts/progressive_report.py --seeds 0,1,2 --workers 3 --registry progressive_decision --no-series --out reports/progressive
# 只根据已保存的 JSON 重新生成 HTML
.venv/bin/python scripts/progressive_report.py --render reports/progressive
# PG4 伸缩实验（IP 数、属性数、服务器数；--days 缩短天数）
.venv/bin/python scripts/progressive_scale.py --days 3 --workers 2 --out reports/progressive/scale
```

- `progressive_report.py` 用 `backend/app/eval/pmetrics.py` 评分（门限 PG1–PG11，`docs/lib3/progressive.md` §12），并从每天的快照中抽取需求示例的
  系统视角与群组视角陈述、逐条核对真值，写 `progressive_report.{json,html}` 与每个种子的 `runs/O_<种子>.json`。
- 包 O 的默认注册模式是 `full+progressive`，但它在约 700 个来源上需要每进程 7 GB 以上内存，因此实测使用 `--registry progressive_decision`
  （每次运行约 28 分钟、峰值约 1.4 GB）。其它参数：`--bounded`（统计引擎库有界模式）、`--keep-res`/`--rescore`（保存/重评运行结果）、`--assemble`、`--skip-existing`。
- 生成器 `backend/app/pipeline/orggen.py`，评估包 `backend/app/eval/packs.py`（O、O60、O-real、O-red、O-servers、O-scale）。

## 测试

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests
```

- 全套约 16 分钟；渐进内核集成后为 `2769 passed, 4 skipped`（983 s）。
- 目录：`tests/engines/`（逐引擎规格测试，渐进内核为 `test_p00_*` … `test_p15_*` 与 `test_progressive_integration_fixes.py`）、`tests/lib/`、`tests/core/`、`tests/eval/`（含包 O 生成器、PG 指标与收敛实验）、`tests/api/`（API v2），
  以及顶层的流水线端到端、tick 模式黄金、节拍不变性等测试。
- 引擎/库/核心单元测试通过各目录的 `conftest.py` 以 v2 的 `tick` 模式运行；流水线、Runtime、评估与脚本默认 `canonical`（双粒度）模式。
- 慢测试：`APPMON_SLOW=1` 启用 `tests/test_cadence_invariance.py` 的 Part B（整条流水线，数分钟）与
  `tests/test_b07_automation_generator.py`。`test_part_b_live_60_equals_900` 目前**失败**（60 s T 流校准问题，见 integration.md §10.4），
  `test_part_b_null_levels` 为 xfail。

## API

所有端点前缀为 `/api`，时间按 `ctx.config.tz` 显示。

**旧版（`backend/app/api/routes.py`，兼容保留）**

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 运行时状态 |
| GET | `/api/overview` | 总览 KPI |
| GET | `/api/engines` | 引擎拓扑（产出/消费/说明） |
| GET | `/api/catalog` | 原始与次生指标目录 |
| GET | `/api/signatures` | 行为特征库 |
| GET | `/api/systems` | 业务系统列表 |
| GET | `/api/systems/{system}/entities` | 实体列表，按 `behavior.risk` 排序 |
| GET | `/api/systems/{system}/entities/{entity}` | 实体详情：画像摘要、风险、身份、状态、特征行（每粒度当前值与 p5/p50/p95、z/zr） |
| GET | `/api/systems/{system}/entities/{entity}/metrics` | 最新原始/次生指标 |
| GET | `/api/systems/{system}/entities/{entity}/series?name=` | 指定序列的历史 |
| GET | `/api/events` | 行为事件 |
| GET | `/api/matches` | 特征命中 |
| POST | `/api/step` | 手动推进一拍 |

**API v2（`backend/app/api/routes_v2.py`，规格见 `docs/lib3/api_ui.md`）**

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/systems/{system}/entities/{entity}/portrait?version=` | 版本化画像（json、中英文文本） |
| GET | `/api/systems/{system}/entities/{entity}/portrait/diff?from=&to=` | 两个画像版本的差异 |
| GET | `/api/systems/{system}/entities/{entity}/timeline?since=` | 实体时间线（事件、风险带变化、状态、版本） |
| GET | `/api/systems/{system}/entities/{entity}/scores?since=` | 各检测器 p、p_family、q_inst、q_all、e_day、evidence、trust、regime 序列 |
| GET | `/api/systems/{system}/entities/{entity}/identity` | 混淆行、特质、召回与 EER（含 CI）、T99、归因后验序列 |
| GET | `/api/systems/{system}/classes` | 角色/子类/静态/池类，含成员、风险、打开的类事件 |
| GET | `/api/systems/{system}/classes/{cid}` | 类画像、类监测带、活跃比例热图、采用史、可辨识度、谱系 |
| GET | `/api/systems/{system}/summary` | 系统视图：共模与一致变化、活动、健康摘要 |
| GET | `/api/incidents?system=&entity=&status=&severity=&kind=` | 事件单列表（kind = entity \| class） |
| GET | `/api/incidents/{incident_id}` | 事件单详情：按族/轴证据、e_day、解释、反事实（有效性与范围）、叙述、活动、父子关系 |
| POST | `/api/incidents/{incident_id}/feedback` | 写入标签 `{verdict, scope, ttl_s, note}` |
| POST | `/api/events/{event_id}/feedback` | 对单个事件写入标签 |
| GET | `/api/label-queue` | 待标注队列 |
| GET | `/api/detectors/health` | 每引擎运行/错误/陈旧度；每检测器 KS、实际率、weight_mult、降级占比与原因 |
| GET | `/api/eval/report` | 最新评估报告（JSON） |
| GET | `/api/store/memory` | 存储内存报告 |

**API v3：渐进画像内核（`backend/app/api/routes_v3.py` + `progressive_views.py`，规格 `docs/lib3/progressive.md` §9.4）**

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/v3/status` · `/api/v3/systems` | 内核是否运行、注册模式；每系统节点（按状态）、动作、陈述、谁的粒度、档位 |
| GET | `/api/v3/systems/{s}/view?fresh=&flat=` | 系统视角：动作 → 谁 → 何时 → 内容 → 绑定 → 流程，置信度/支持度/版本 |
| GET | `/api/v3/systems/{s}/precision?days=` | 精度随时间：每日已确认模式、特化深度、分裂、漂移、违例；评估实测召回/精确率曲线 |
| GET | `/api/v3/systems/{s}/lattice?kind=&root=&depth=&limit=` | 模式格（有界广度优先） |
| GET | `/api/v3/patterns/{pid}` | 模式详情：路径、父子、例外、约束、陈述、谱系、生命周期/漂移、违例 |
| GET | `/api/v3/groups` · `/api/v3/groups/{g}/view` | 群组列表；群组（用户）视角：系统 → 动作 + 否定陈述 |
| POST | `/api/v3/groups/{g}/name` | `{name}`：给群组命名（写入 `who_group_names`，P11 下次聚类匹配） |
| GET | `/api/v3/systems/{s}/entities/{ip}/view` | IP 视角：继承的群组模式、例外、绑定、点名陈述、违例 |
| GET | `/api/v3/violations?system=&entity=&type=&severity=&since=&limit=` | 类型化违例（中英文原因与标记释义） |
| GET | `/api/v3/systems/{s}/facets` · `/groups/{g}/facets` · `/systems/{s}/entities/{ip}/facets` | P13 多维刻面组成 + 刻面注册表 |
| GET | `/api/v3/systems/{s}/strategy` | P12 系统刻画、选择的策略与引擎、理由、历史、系统族 |
| GET | `/api/v3/systems/{s}/attributes` | P02 属性注册表与 P05 角色 |
| GET | `/api/v3/budget` | P15 档位/上限、降级阶梯、活跃/挣得集合、`ops.budget` 序列、P 引擎耗时 |

## 目录

```
backend/app/
  models/schema.py           统一数据模型（Observation、RawMetric、DerivedMetric、EntityProfile、BehaviorEvent、Incident、Label、SignatureMatch）
  core/{store,engine}.py     时序/模型存储 + 引擎框架/注册表
  engines/raw|derived|behavior|signature/   四库引擎
  engines/behavior/lib/      行为库共享数学（特征表、检测器注册表、预测分布、校准、融合、序贯阈值、门控、双粒度、模型访问器）
  engines/behavior/lib/p*.py 渐进画像内核共享数学（事件批、泛化层次、草图、e 过程、模式树、打分、语法、函数依赖、时间窗、流程、MinHash/Louvain、策略、侧面、渲染）
  pipeline/                  编排器 · 合成流量源 · 组织生成器 orggen.py · 注册表装配与 Runtime
  eval/                      评估包、真值、指标、运行器、报告；渐进内核门限 pmetrics.py 与伸缩实验 pscale.py
  api/                       FastAPI：routes.py（旧版）、routes_v2.py + views.py（v2）、routes_v3.py + progressive_views.py + progressive_runtime.py（v3 渐进内核）
data/
  catalog.yaml               指标库目录（原始 55 + 次生 24，含获取方式）
  signatures/*.yaml          行为特征库（原子 24 + 组合 5）
frontend/                    零依赖原生 JS 仪表盘（内联 SVG 图表）
docs/                        设计文档（中文）与 docs/lib3/ 行为库规格
reports/                     评估报告（最新 + 历史各轮）；reports/progressive/ 为渐进内核（包 O 与伸缩实验）
scripts/                     smoke.py、evaluate.py、progressive_report.py、progressive_scale.py、render_check.py、gen_beacon_null.py
tests/                       单元、集成、API、评估测试
```
