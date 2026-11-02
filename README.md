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

## 现状（2026-11-02）

- 60 个引擎已实现；默认注册表 `full` 注册 44 个（不含渐进内核），`full+progressive` 注册全部 60 个。严格模式下 0 异常。
- **渐进画像内核**（组织包 O，21 天、种子 0–4，第十轮最终代码，`reports/progressive/`）：第十轮继续只修第九轮列出的退步，门槛是“包 O 至少与第九轮一样好，并把列出的退步拉回”，**大部分达到，没有全部达到**。
  第 14 天模式召回 **0.895**（同第九轮；目标 0.90，差 0.005；开发种子 0–2 的均值 0.930），第 21 天 0.905；精确的主指标（只评判“它所描述的真值行为已可观测满一个工作日”的陈述）第 14 天 0.941 → **0.967**（高于第八轮 0.955），第 21 天 0.958 → **0.971**（负责人要求 ≥ 0.975，差 0.004；
  5 个种子池化第 21 天 338 条中失败 10 条，与第八轮 339 条中 9 条持平）。
  **越用越准**：可观测陈述的精确率第 5 天 0.88 → 第 10–21 天 0.97–0.98，陈述置信度中位数 0.79 → 0.98；“不变模式的置信度不降”（符号检验）5/5；但池化校准误差 0.053 → **0.071**（目标 0.05；偏保守：陈述 0.940、实际成立 0.952）。
  需求示例逐条核对 50 项中 46 项通过（第九轮 47；核对的审批条款改为按当天有效的真值路由，按第九轮的条款为 45）——5/5 个种子还原“综合部 3 个 IP 的登录节点 + `username=` 语法与封闭集 {jack, mike, mike.w, rose} + 3 条 IP→用户名绑定”、.21 的审批（改名后在 `/flow/` 下）、17 点报告、“财务部（192.168.2.10）”独占财务审批与用户视角“综合部在财务系统中从未执行写操作”，
  差种子 1、3 的 90 % 带（1–2.5 KB、1–3 KB）、种子 4 的登录窗（IoU 0.68）与种子 1 的范围（1–2.8 KB，新）。
  示例异常 50/50，≥ MEDIUM 误报 0.0005/实体·日（≥ LOW 0.0013 → 0.0011），PG3、PG6、PG7 通过；漂移 D2、D3、D4 5/5，D5 4/5，D1 3/5，第 14 天 6 条绑定 5/5；两个散列种子下评分与示例 0 处差异。
- **变体**：**红队包 O-red** 的目标全部达到——≥ MEDIUM 误报 0.0027 → **0.0002**、D1–D5 全部通过、A4 检出（10/10）、第 14 天精确 0.891 → 0.982（门户按会话位置的年轻子节点不再出现、P06 / P03 的范围下限循环已修）。
  **真实扰动包 O-real**（35 天，3 个种子）：第 35 天召回 0.633 / 0.646 / 0.608 → **0.709 / 0.709 / 0.620**；**财务出纳换址后，新地址第 11 天起进入学到群组，财务账簿模式第 11–16 天起还原**（第八轮第 30–31 天；P11 的签名不再乘 P03 的降权），但财务登录换址后的模式仍丢失；
  示例核对 5 / 6 / 6（第八轮 5 / 6 / 7），种子 2 第一次恰好学到综合部 3 个地址的登录者；种子 2 的改名晚一天被陈述、第 14 天绑定 0/6（新退步）；校准误差 0.086–0.156。O60 第 8 天一致性 0.077（超过 0.05，新退步）。
- **资源**：内存对 IP 数、服务器数次线性；内存对属性数 0.202 → **0.150**、每事件 CPU 对属性数 0.191 → **0.165**，都以余量达标（340 个属性的点 49.4 → 40.0 MB，最大单棵树 18.5 → 11.6 MB）；每事件 p95 打分 0.75 ms、学习 1.73 ms，仍远超 100 / 250 µs 的目标；同等负载下 P 内核为第九轮的 0.96 倍。
  **PG9** 只差一项：A 包有界模式多 1 个 MEDIUM；A / E 包库三 CPU p95 为完整的 0.85 / 0.87，1 000 个临时 IP、84 小时的点 0.43。
  验收仍**未通过**；真实日志试点未做。
- **统计引擎库**（A–E 包第四轮，`reports/eval_report.json`）：15 项门限中第 15 项（鲁棒性）通过，第 12、13 项 n/a，其余未通过；
  期限内威胁召回 0.71（目标 0.95），对照实体 FAR ≥ LOW 0.156、≥ MEDIUM 0.042（均达标），单包墙钟与实时 p95 远超 360 s / 80 ms 目标。
  第五、六轮改了共用的决策链，只在 A、E 包种子 0 上复测；第七轮没有改 B 引擎；第八轮只改了 B14；第九轮改了 B13、B28 与 `lib/gating`（有界模式的补齐、按（来源, 凭据）对的信任），A、E 包完整模式与第八轮每个数字都相同；第十轮没有改 B 引擎，A、E 包完整模式与第九轮每个非耗时的叶都相同（设计文档 3.5、3.10.2）。
- 结果、原因与未决问题见设计文档库三 §3.0（逐句需求状态）、§3.6（O-real，含财务换址后的各行）、§3.7（资源与成本）、§3.8.0（第十轮实测）、§3.8.0a（逐字学到的画像，两种视角，含 O-real 第 35 天）、§3.8.0b（修订汇总）、§3.8.0c（如实结论与每一项退步的诊断）、§3.8.0d（第九轮，历史）、§3.10、§3.11 与 `docs/lib3/progressive.md` §3.1、§16.18。

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
`full+progressive`（60 个）、`progressive_only`（18 个，伸缩实验）、`progressive_decision`（24 个 = 渐进内核 + B24–B29 决策链，组织包 O/O60/O-red/O-real 的默认与全部实测）；
`lib3.resource_mode = bounded` 时 `full` 另注册 P15（45 个）。
后端服务（下文）默认运行 `full`；设置 `APPMON_PROGRESSIVE=decision` 时在组织包 O 上运行渐进内核，其视图经 API v3 与前端“画像模式”页展示。
P2 引擎 B19 mixture、B20 session_profile、B22 action_embedding 有规格但未实现、未注册（B21 cross_system 已在第四轮实现）。

## 安装

需要 Python 3.11+。

```bash
python -m venv .venv
.venv/bin/pip install -r backend/requirements.txt pytest
# 可选：scripts/render_check.py 需要 playwright 与 Chromium
```

`backend/requirements.txt` 中的 `numba>=0.60` 是**可选**加速：模式树分裂统计的 numba 核（`backend/app/engines/behavior/lib/pnumba.py`）与纯 numpy 路径逐位相同；
不安装 numba，或设置 `APPMON_NO_NUMBA=1`，就运行 numpy 路径（结果相同、更慢）。JIT 缓存写在 `lib/__pycache__`，每份检出第一次编译约 3–6 s。

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
| `APPMON_NO_NUMBA` | 设为 1 时不用 numba 核（`lib/pnumba.py`），运行逐位相同的纯 numpy 路径 |

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
单个包-种子在单核上约 12–22 分钟（主机空闲时 A 包约 740 s、B 包约 1315 s；第四轮 4 进程并行时最长 2018 s）。最新的全矩阵结果（第四轮）在 `reports/eval_report.{json,html}`，历史各轮在 `reports/*/`；第五至十轮只在 A、E 包种子 0 上复测（第五轮 `reports/progressive/round5_eval/ae/`；第六至八轮见 `docs/lib3/progressive.md` §16.14.8、§16.15.9、§16.16.9；第九轮 `reports/progressive/round9_diag/{A,E}_0_full.json`、§16.17.7；第十轮 `reports/progressive/round10_diag/{A,E}_0_full.json`、§16.18.7）。

## 渐进画像内核：组织包 O 评估与伸缩实验

```bash
# 包 O（需求中的组织：综合部/财务部/销售部/研发池、OA 与财务等 6 个系统、21 天、900 s 拍），种子 0–4（第十轮每个约 28–32 分钟，至多 2 个并行）
.venv/bin/python scripts/progressive_report.py --seeds 0,1,2,3,4 --workers 3 --no-series --out reports/progressive
# 可续跑：每约 600 s 在日末写检查点，跑满约 2 400 s 后在下一个检查点停下；再次执行同一命令从检查点继续（容器重启后也可）
.venv/bin/python scripts/progressive_report.py --seeds 0 --workers 1 --no-series --checkpoint /tmp/pckpt --segment-s 600 --stop-after 2400 --out reports/progressive
# 变体（红队 O-red、60 秒事件模式 O60、35 天的真实扰动包 O-real），写 runs/<包>_<种子>.json（也会覆盖 progressive_report.json，最后用 --assemble 重建）
.venv/bin/python scripts/progressive_report.py --pack O-red --seeds 0 --no-series --out reports/progressive
.venv/bin/python scripts/progressive_report.py --pack O60 --seeds 0 --no-series --out reports/progressive
.venv/bin/python scripts/progressive_report.py --pack O-real --seeds 0,1,2 --workers 2 --no-series --checkpoint /tmp/pckpt --segment-s 900 --out reports/progressive
# 不重跑，只把已保存的运行（含变体）汇总成报告
.venv/bin/python scripts/progressive_report.py --assemble --seeds 0,1,2,3,4 --variants O-red,O60,O-real --scale reports/progressive/scale_r10 --pg9 reports/progressive/pg9_r10/pg9.json --out reports/progressive
# 只根据已保存的 JSON 重新生成 HTML
.venv/bin/python scripts/progressive_report.py --render reports/progressive
# PG4 伸缩实验（IP 数、属性数、服务器数；--days 指定天数，规格为 7 天）
.venv/bin/python scripts/progressive_scale.py --days 7 --workers 2 --out reports/progressive/scale_r10
# PG9（有界 / 完整，负责人第七轮的定义）：A、E 包成对运行（同时启动）、大规模临时 IP 点（先后运行，8.5 GB RSS 时干净停止），再汇总成 pg9.json
.venv/bin/python scripts/progressive_pg9.py pairs --packs A,E --out reports/progressive/pg9_r10
.venv/bin/python scripts/progressive_pg9.py large --ips 1000 --hours 84 --sequential --out reports/progressive/pg9_r10
.venv/bin/python scripts/progressive_pg9.py summary --out reports/progressive/pg9_r10
```
- 代码快照（例如评估时把 `backend/`、`scripts/` 复制到别处运行）必须带上 `data/`：库四的签名库按代码位置查找（或设 `APPMON_SIGNATURE_DIR`），第八轮起评估运行（严格模式，或设 `APPMON_REQUIRE_SIGNATURES=1`）缺库时报错 `SignatureLibraryMissing`，其他情况告警并在存储中标出，`progressive_report.py` / `progressive_pg9.py` 拒绝在缺库时运行带库四的注册表（第七轮评估中的陷阱，设计文档 3.8.0f、3.3.13）。

- `progressive_report.py` 用 `backend/app/eval/pmetrics.py` 评分（门限 PG1–PG11，`docs/lib3/progressive.md` §12），并从每天的快照中抽取需求示例的
  系统视角与群组视角陈述、逐条核对真值，写 `progressive_report.{json,html}` 与每个种子的 `runs/O_<种子>.json`。
- 组织包的默认注册模式是 `progressive_decision`（`full+progressive` 在约 700 个来源上需要每进程 7 GB 以上内存；`--registry` 可覆盖）；
  第十轮最终代码每次运行约 28–32 分钟（至多 2 个并行；未改动的引擎用了第九轮 1.11 倍的时间）、峰值 RSS 约 1.7 GB（随检查点的序列化而变，内存只在不写检查点的运行之间比较），O-real 每次约 70–75 分钟、约 3.1–3.3 GB（一台机器上同时只跑一个 O-real：检查点序列化的峰值接近 7 GB）；第五轮起运行不再依赖散列顺序（`PYTHONHASHSEED` 0 与 1 评分与完整结果都相同），第六轮起续跑与一次跑完相同，评估入口仍固定散列种子 0。第一轮（提交 6001027）的运行保存在 `reports/progressive/round1/runs/`，第二轮在 `round2/runs/`（原样）与 `round2/rescored/`（用第三轮的真值与评分器重评），第三轮加 V10 之前的最终运行在 `round3_eval/pre_v10_runs/`，第三轮伸缩点在 `round3_eval/scale7/`，第三轮的运行与报告在 `round3/`，第四轮评估/成本/伸缩负责人的记录在 `round4_eval/`、合并后的伸缩点在 `scale_r4/`，第四轮的运行与报告在 `round4/`（用第五轮评分器重评的在 `round4/rescored_r5/`），第五轮的伸缩点在 `scale_r5/`、评估者的可复现 / O-real / A、E / PG9 记录在 `round5_eval/`，第五轮的运行与报告在 `round5/`（用第六轮评分器重评的在 `round5/rescored_r6/`），第六轮的伸缩点在 `scale_r6/`、运行与报告原样在 `round6/`，第七轮的伸缩点在 `scale_r7/`、PG9 在 `pg9_r7/`、评估工具负责人的 PG9 / 输入成本 / B25 诊断在 `round7_harness/`、运行与报告原样在 `round7/`，第八轮的伸缩点在 `scale_r8/`、PG9 在 `pg9_r8/`，评估工具负责人用第八轮评分器重评的第七轮（`rescore_r7.json`）、节假日变体、P02 剖析与空闲实体测量在 `round8_harness/`、运行与报告原样在 `round8/`，第九轮的伸缩点在 `scale_r9/`、PG9 在 `pg9_r9/`、评估者的诊断导出（O-real 财务、O-red 门户、A / E 包的 `scripts/evaluate.py` 结果，只有 JSON 与文本）在 `round9_diag/`、运行与报告原样在 `round9/`，第十轮的伸缩点在 `scale_r10/`、PG9 在 `pg9_r10/`、评估者的诊断导出与脚本（包 O 种子 2 门户的留出记录、O-real 种子 0 的逐日财务陈述与 P11 群组、A / E 包结果，见 `README.txt`）在 `round10_diag/`。其它参数：`--bounded`（统计引擎库有界模式）、`--keep-res`/`--rescore`（保存/重评运行结果）、`--assemble`、`--skip-existing`、`--checkpoint`/`--segment-s`/`--stop-after`（可续跑，`backend/app/eval/resumable.py`）。
- 生成器 `backend/app/pipeline/orggen.py`，评估包 `backend/app/eval/packs.py`（O、O60、O-real、O-red、O-servers、O-scale）。

## 测试

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests
```

- 全套约 20–45 分钟；第十轮最终代码上为 `3456 passed, 4 skipped`（21 分 14 秒，旁边同时跑一个 O-real）；第九轮为 `3436 passed, 4 skipped`（19 分 47 秒）；第八轮为 `3397 passed, 4 skipped, 1 failed`（`test_b13_budget_edges::test_perf_forty_entities_generous_bound` 在负载下 83 ms 对 80 ms 上限，第九轮改为按交错运行的参照工作量计、上限未放宽）。耗时类测试读进程 CPU（`tests/helpers.cpu_s`）、交错比较配置、取 N 次中最好的一次、断言操作次数或以参照工作量为单位，不受机器负载影响。提速类改动各有在录制的包 O 事件流上与旧实现逐位比较的等价测试（`tests/lib/test_*_equivalence.py`、`tests/engines/test_p04_p05_stream_equivalence.py` 等），散列顺序测试在多个 `PYTHONHASHSEED` 下起子进程运行（`tests/engines/test_registry_hash_order.py`、`test_round5_hash_order_views.py`）。
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
  engines/behavior/lib/p*.py 渐进画像内核共享数学（事件批、泛化层次、草图、e 过程、模式树、打分、语法、函数依赖、时间窗、流程、MinHash/Louvain、策略、侧面、渲染、价格表 pcost、可选 numba 核 pnumba）
  pipeline/                  编排器 · 合成流量源 · 组织生成器 orggen.py · 注册表装配与 Runtime
  eval/                      评估包、真值、指标、运行器、报告；渐进内核门限 pmetrics.py、伸缩实验 pscale.py 与可续跑运行器 resumable.py
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
