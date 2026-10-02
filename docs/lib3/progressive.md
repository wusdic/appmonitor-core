# lib-3 v3 — Progressive Profile Core (PPC): specification

Status: **implemented and integrated** (specification of 2026-09-29, base commit 482c534; integration and
round-1 results 2026-09-30 / 10-01 in §16.1–§16.8 with the deviations M1–M25; round 2 of 2026-10-01 in §16.9
(scenario adaptation, bounded decision chain, cost: A1–A11) and §16.10 (results on the final code, 5 seeds,
O-red, O60; deviations M26–M46, H1–H6, G1–G7, C1–C2, E1–E11)). Sections §0–§15 are the specification as written;
where the code differs, §16.2, §16.9, §16.10.1 and the as-built cards at the end of `engines.md` say so.
Measured status per requirement sentence: §3.1.
Scope: library 3 (行为库) profile core, redesigned per the user requirement quoted
in §3, plus the changes that make the existing B01–B30 engines resource-bounded
(§10). Library 1/2 get a capture extension (§5.1) and two event engines (P00, P01);
library 4 is unchanged except for one optional input (§9.3).

Reading guide. §0 is the Chinese summary for the design doc. §2 says what the
current data can and cannot support. §3 maps every sentence of the requirement to a
mechanism. §5 is the data model, §6 the algorithms (formulas, pseudo-code,
complexity), §7 the budgets, §8 the engine cards, §9 the integration with
B01–B30, §10 the library-wide scalability change list, §11 the generator
extension, §12 the gates, §13 the work breakdown, §14 decisions and risks,
Appendix A the review changes. §6.20 (servers: system families), §6.21 (real-world
situations) and §6.22 (a worked trace of the requirement's example) were added by the
review.

Conventions. "IP" is the subject of every profile (users are IPs or IP classes,
never persons; a `username=` value is a *content attribute of an IP's events*,
not an identity). `s` = system, `ip` = client IP, `n_eff` = effective evidence
(§6.5.4). Half-lives: H_s = 1 d (short), H_m = 7 d (medium), H_l = 30 d (long; the
*confidence channel*, reset on an accepted change, §6.9.4). Three quantities are kept
apart everywhere (§6.5.4): **mass** (how many real events a row stands for; used for
distributions and rendering), **evidence** (how many independent observations the
learner has seen; used for every test, bound and confidence), and **cost** (µs, bytes).
Every number called a *default* is a design constant with a reason in §7.1; every
number called an *estimate* must be replaced by a measurement (gate PG4) before it
is quoted anywhere else. No accuracy is claimed in this document. Appendix A lists
the changes made by the adversarial review of 2026-09-29 and why.

---

## 0. 中文摘要（主设计文档库三 §3.0–§3.12 是完整的中文版，含实测结果）

**要解决的问题。** 现有行为库（B01–B30）以“每个 (系统, IP) × 固定 52 维特征 × 每拍”为单位建模：
每个实体独立维护基线、检测器、校准环、检查点，成本与内存随“IP 数 × 指标数”线性增长
（性能专项实测 `integration.md` §9.2：约 9.4–11.7 MB/实体、约 15 ms/实体·拍），指标集合写死在 `FEATURE_SPEC` 中，
画像的粒度（个体 / 角色类）也是预先规定的；新设计若按“每个系统一套固定内存与周期任务”实现，
成本又会随服务器数线性增长。需求要求的是相反的东西：**不遍历所有用户与服务器**，
而是**由粗到细、随时间逐步变准**地学到“一类行为”乃至“某一个 IP 的行为”，并随行为演化；
指标不写死、可动态增加；从多个侧面叠加出完整画像；并能按场景自动选择算法。

**核心思路（渐进画像内核，PPC）。**
1. **行为事件 + 开放属性空间**：学习单元是一次请求/事务/流/会话步骤（行为事件），
   携带任意引擎产出的开放属性表（HTTP 方法、路由模板、状态、上下行字节、请求体键值、
   客户端栈、SNI、会话位置、本地时刻……以及任意原始/次生指标）。属性注册表在线推断
   类型（类别/数值/序数/文本载荷/IP/时间/集合）、基数、稳定性，并为每个属性指定
   **泛化层次**（IP → /24 → /16 → 学到的 IP 群组/区域 → *；路由 → 模板 → 路径前缀 → *；
   时刻 → 分钟 → 学到的时间窗 → 时段 → 日类型 → *；数值 → 学到的分位箱 → *；
   载荷 → Drain 模板 → 键集合 → 取值形状（字符集/长度）→ *）。新指标出现即自动登记，无需改代码。
2. **渐进式、资源有界的模式树（格上的一条路径）**：每个系统（或系统族，见第 12 点）从最泛的根模式开始；
   只有当流式统计**证明**“按某属性某层级细分后，其余属性变得更可预测”时才细分。
   证明分两部分：一是**随时可停的有效性检验**（逐目标属性计算“前序预测码长”与“最大似然码长”之差，
   得到 e 值，按目标取平均后与阈值比较；由 Ville 不等式，无论检查多频繁，每个叶节点误分裂概率 ≤ 2⁻¹⁰ × 候选数的调和级数（约千分之几），
   且对目标之间的相关性、对样本加权都成立）；二是**MDL 收益**（所有目标的总码长节省须超过分裂自身的描述长度）。
   候选之间用时间一致的经验 Bernstein 界保证选择稳定（VFDT/EFDT 式的平局打破）。
   于是**观察越久，模式越细、置信度越高**；子模式不再有差别时自动合并、剪枝。
   计数全部用**带多尺度指数衰减的有界草图**（Space-Saving、分层重击者 HHH、Count-Min、HLL、t-digest），
   内存受预算约束，**与 IP 数、指标数无关，与服务器数近似无关**。单个 IP 的专属模式只作为“例外”存在——
   只有当它被证明与所在群组不同（同样的 e 值检验，零假设用“去掉该 IP 后的群组分布”）才建立，其余 IP 继承群组模式。
   **“越久越准”与“随行为变化”的统一**：分布形状用 7 天半衰期跟随变化；置信度用单独的“置信通道”
   （30 天半衰期，一旦确认发生了合法变化就从新状态重新累积），因此行为稳定时置信度持续上升，
   行为变化被接受后从变化点重新积累，而不是永远停在一个由衰减决定的上限。
   **证据单位**：同一 IP 连续突发的事件按调和级数折算，聚合记录与抽样权重只影响“量”（分布估计），
   不会虚增“证据”（检验与置信度），避免抽样/聚合把 1 条观测当成 1000 条证据。
3. **自动特征选择**：每个系统、每个模式节点按覆盖率、稳定性、可预测性（条件熵压缩率）、
   冗余（近似函数依赖 g3）决定“用哪些指标做细分、保留哪些做约束、哪些只保留形状、哪些丢弃”，
   结果本身随时间修订；常量变成“不变式”，标识符只保留形状，噪声被丢弃。
4. **内容约束学习**：数值的 90% 区间与“全部落在 [a,b]”的硬界（保形秩界 + 极值 GPD 尾部），
   载荷语法（如 `username=[a-z]{3,8}`，长度上下界来自观测并带覆盖率，而不是预设的“≤10”）、键集合、取值闭集，
   以及近似函数依赖/绑定（`192.168.1.21 → username=jack`；共用终端则学到集合绑定），全部带置信度与支持度。
5. **时间窗与业务流程**：在环形日时轴上用 Bayesian Blocks 分段学到活动窗（如工作日 09:00–09:21）；
   在会话可识别时，用直接跟随图 + 启发式挖掘（依存度、时间约束）学到
   “登录 → 审批页 → … → 17:00 提交报告”。
6. **“谁”的发现**：IP × 模式的二部图（加权 MinHash + LSH + Louvain）得到行为群组
   （如“综合部” = 共享一组模式的 IP），IP 无规律时退化到网段/区域，IP 无信息时直接不用 IP；
   群组的**成员与模式是自动学到的，“综合部”这个名字则必须有来源**：运维配置、IPAM/DHCP 作用域描述、
   资产台账或目录服务导入（按成员重合度自动匹配），没有来源时显示自动名（G7·“OA 登录+审批”）。
7. **两种视角**：同一个模式库投影出“用户/群组视角”（群组 → 系统 → 动作 → 时间 → 内容）
   与“业务系统视角”（系统 → 路由/动作 → 谁 → 时间 → 内容约束），渲染成中英文陈述句，
   附支持度、置信度、首末次出现与版本。
8. **一致性检测**：事件按“最具体的已确认模式”回退打分，违规分型为
   谁（如综合部 IP 去财务系统审批）、时间、内容（大小/语法/绑定，如用户名不符）、序列、新动作，
   校准为 p 值后作为新的检测器族进入现有校准→融合→风险→事件链（B24–B27），并供 B29 解释、B30 画像。
   “谁”的违规区分三种情况：**已知属于别的群组的 IP**（如综合部 IP 访问财务审批，高危候选）、
   **从未见过的 IP**（在敏感的封闭模式上为中危；若它提交的用户名所绑定的原 IP 当天未出现，判为疑似 DHCP/VPN 换址，降为低危；
   若原 IP 同一小时内仍在活动，判为凭据并发使用，中危，与“谁”违规叠加时为高危候选）、
   **群组内成员**（正常）。离散告警按“每 IP·每模式·每类型·每天”做多重检验校正，控制每实体日误报。
9. **多侧面画像**：时间节律、网络/空间、功能（路由-动作）、内容、序列/流程、关系/社会（群组、共访）、
   技术（客户端栈/TLS）、量/预算、身份/连续性等侧面由不同引擎产出并登记到动态侧面注册表，
   像描述一个人的外观、生物学特征、社会属性那样组合成画像。
10. **场景自适应**：系统刻画器度量每个业务系统的用户规模、IP 稳定性/DHCP 翻转、路由基数、
    载荷可见性（是否加密）、会话可识别性、量级与周期性；策略选择器依据引擎声明的适用前提与成本，
    以及**实测效用**（前序预测码长增益减去计算成本，统一以“比特/事件”计；可并行度量的维度用全信息 Hedge，
    需要实际运行的维度用成本感知的多臂老虎机）为每个系统选择引擎与粒度，定期重评。
11. **全库可扩展**：现有 B01–B30 改为只处理活跃集；逐实体模型只给“挣得”它的 IP
    （逐实体模型的前序似然相对群组模型的增益 ≥ 阈值），其余 IP 由类/群组模型覆盖（§10 变更清单）。
12. **服务器维度同样不遍历（系统族）**：运行同一应用的多台服务器（负载均衡集群、各分支的同构系统）
    按路由模板/SNI/客户端栈的 MinHash 相似度自动归为“系统族”，共用一棵模式树，`net.dst`（哪台服务器）
    只是一个普通的可细分属性——与“一类用户 → 某一个用户”对称，得到“一类服务器 → 某一台服务器”。
    每个系统的内存与周期任务按其活跃度分配（空闲系统只占极小的 XS 档），周期拟合只处理“有新证据的节点”，
    新上线的服务器加入已知系统族即可立即继承画像（冷启动）。
13. **真实环境适配**（§6.21）：NAT/代理/负载均衡源地址转换（优先取可信代理的 X-Forwarded-For，否则 IP 退化为类别）、
    DHCP/VPN/IPv6 临时地址（网段/群组层级 + 换址判定）、加密流量（只用大小/时序/SNI/JA3，内容引擎自动关闭）、
    多人共用终端与服务账号（“集合绑定”：某 IP 只用这几个用户名 / 某用户名只从这几台服务器来）、
    批处理突发（证据单位折算）、月末/季末任务（日历上下文 + 休眠模式记忆）、节假日（不计入陈旧与漂移计时）、
    指标消失（模式结构不被误判为行为变化）、扫描器（离群事件降权学习）、传感器抽样与丢包（按抽样率加权）。
    生成器新增对应的扰动包 O-real 与门限 PG11 逐项验证。

**验证**：生成器新增组织包（综合部/财务部/销售部/研发 DHCP 池、OA 与财务系统、大用户量门户、
加密邮件系统、运行中新增指标、合法行为漂移、需求示例中的各类异常），并发布模式真值；
新门限 PG1–PG11 度量模式还原的精确率/召回率、随时间的收敛曲线与置信度校准、达到的具体度（群组还是 IP）、
内存/CPU 对 IP 数、指标数与服务器数的次线性、漂移适应延迟、示例异常（A1–A10，含慢速投毒）的检出与误报、
以及真实环境扰动（O-real）下的稳健性。

**实测状态（第二轮，2026-10-01，§16.10；包 O 种子 0–4、21 天）**：第 14 天模式召回 0.58、精确 0.53（目标 0.90；第一轮 0.21 / 0.34），
召回随时间上升（第 5 天 0.22 → 第 21 天 0.60）；需求示例在 4/5 个种子上还原到“综合部 3 个 IP 的登录节点 + `username=` 语法（种子 0 另有封闭集）+ 3 条绑定”，
5/5 个种子还原 .21 的审批、17 点报告、“财务审批只有 192.168.2.10”与用户视角“综合部在 finance 中从未执行写操作”；“全部在 0.5–3 KB”只有 1/5；
示例异常 48/50 检出，≥ MEDIUM 误报 0.004/实体·日；PG7 通过；陈述置信度（留出检验通过频率）ECE 0.30、不随时间上升；场景策略 0.83；
内存对 IP 数与属性数次线性，每事件耗时仍超目标；O-real 与真实日志试点未做。逐句状态见 §3.1，中文完整版见主设计文档库三 §3.0、§3.8.0。

---

## 1. Terms

| Term | Meaning |
|---|---|
| Behaviour event (BE) | One request, transaction, flow, DNS query, or one (IP, H-grain) metric window, with an open attribute map (§5.2). |
| Attribute | A named field of a BE: `http.route`, `body.kv.username`, `m.derived.upload_dominance`, … Any producer may add one. |
| Generalisation hierarchy (GH) | For an attribute, an ordered list of levels ℓ = 0 (exact) … L (= `*`) and a map `gen(a, ℓ, v)` (§5.4). |
| Context | A conjunction of constraints `(a, ℓ, V)` meaning `gen(a, ℓ, value) ∈ V`, or `∉ V` for an "other" branch. |
| Pattern | A node of the pattern tree: context + content model (distributions and constraints of the node's target attributes) + who/when summaries + lifecycle. |
| Pattern tree (PT) | Per (system, event kind) tree whose root context is empty; each edge adds one context constraint. The PT is one path-structured sample of the generalisation lattice (all contexts), grown only where evidence pays (§6.5). |
| Target | An attribute whose distribution a node models (at most m_t per node). |
| Split candidate | An attribute level on which a node may be specialised. |
| Exception | A child of a confirmed node whose context adds `ip = x` for one IP x, holding only the targets on which x differs (§6.7). |
| Binding | An approximate functional dependency X → Y inside a node, e.g. `net.src → body.kv.username` (§6.12). |
| Facet | A named aspect of a portrait (temporal, spatial, functional, content, sequential, relational, technical, volume, identity, …) produced by one or more engines (§6.17). |
| Earned | A per-IP model (in P04 or in B01–B30) is *earned* when its prequential log-likelihood gain over the covering group/class model exceeds a threshold (§6.7, §10.2). |
| Mass / evidence | Mass m_e = how many real events a learned row stands for (aggregation × HT weight × sensor sampling); used for distributions. Evidence ω_e ∈ (0, 1] = how much independent information the row carries (§6.5.4); used for every test, bound and confidence. |
| Confidence channel | The H_l-decayed evidence of a node or constraint, reset to the H_m state when a change is accepted (§6.9.4). All closedness, binding, coverage and statement confidences use it. |
| System family | A set of systems (servers) running the same application, sharing one pattern tree in which `net.dst` is an ordinary split attribute (§6.20). |

---

## 2. What the data supports today, and what is missing

Inventory from `models/schema.py::Observation`, `engines/raw/*`, `engines/derived/*`,
`lib/template.py`, `lib/m_template.py`, `pipeline/generator.py`.

| Needed by the requirement | Exists today | Gap and the fix in this spec |
|---|---|---|
| Server identity (192.168.100.100:8080) | `Observation.peer`, `dst_port`; `Observation.system` assigned by the adapter | None. P00 emits `net.dst = peer:dport`; P02 learns it as a system-root *invariant*, which renders as the system address. |
| HTTP method, raw path, status, UA, content type | `http_method`, `http_path`, `http_status`, `user_agent`, `content_type` | None. |
| Route template | R2 Drain-lite templater per system (`model.template@(s,__system__)`, read via `lib/m_template.templater`); its `template_path` learns as it templates | Small gap: P00 must not mutate R2's templater. Additive read-only `Templater.apply_path(host, method, path) -> str` in lib/template.py (same walk, no count update); P00 runs after R2, so route templates agree with `act.stream`. |
| Per-event request / response sizes | `bytes_up/bytes_down` per Observation; in aggregated mode (Δt ≥ 900 s) only the group total and `extra.ts_sample` | **Gap in aggregated mode.** Generator and adapters add `extra.ev_sample` (§5.1): up to 64 per-event rows (offset, up, down, status, body ref). Without it, P00 emits one event per record with the mean and flags it `approx` (§6.2). |
| Request body / form / query key=value, e.g. `username=jack` | Not captured. `lib/template.py` keeps query *names* only; `http_path` still carries query values in raw form | **Gap.** Capture extension `extra['l7'] = {body, body_len, body_trunc, query, headers}` (§5.1). P00 parses form / JSON / multipart key-value structure into `body.kv.<key>` and `q.kv.<key>`. Value retention is a policy (§5.1.3). |
| Headers (new metrics appearing at runtime) | Not captured | `extra['l7']['headers']`, and every scalar leaf of `extra['meta']`, become attributes automatically (open by default). |
| Encrypted payload visibility | TLS-only records have `tls_sni`, `ja3`, sizes, no `http_*` | None; P12 measures visibility per system and disables content engines where it is ~0. |
| Client stack | R3 `lib/stack.stack_token` | None; P00 emits `client.stack`. |
| Local time, day type (workday/holiday/调休) | `lib/timebins.tctx`, `day_type` | None; P01 emits `ctx.tod_min`, `ctx.daytype`. |
| Sessions | D2 session engine (gap = B10's `session_gap`, else 30 min) over `act.stream` (capped 512 rows/tick, 1 h retention) | P01 re-derives session ids on the full event batch with the same gap rule (reads `model.seq` gap via `lib/m_seq`); no cap. |
| Departments / cross-system users (same IP in OA and finance) | Generator personas are per (system, IP); IP ranges are disjoint per system; no department concept | **Gap in the generator.** Org pack (§11) with departments whose IPs use several systems. |
| Arbitrary metrics as learnable attributes | 52 fixed features (`lib/features.py`) | P01 builds `win` events from every fresh scalar raw/derived metric name per active IP and H grain (§6.2.3), capped and prioritised by P05. |
| Operator names for groups (综合部) | `ip_classes` (static CIDR classes) | New config `who_group_names` (§6.15, item 5), fed by an operator, or by an IPAM / DHCP-scope / asset / directory import that writes the same structure. A name cannot be learned from traffic; membership can. |
| Real client IP behind a reverse proxy / load balancer with source NAT | `Observation.src` is the proxy | **Gap.** `trusted_proxies` config + `hdr.x-forwarded-for` resolution in P00 (§5.1.4); without it P05 finds the IP uninformative and the system runs in who = none. |
| Session identity under NAT / shared terminals | IP-keyed sessions only | Optional `sess.key` = HMAC of the session cookie or bearer token (§5.1.4), never stored in clear. |
| Holidays vs weekends, month-end | `day_type` ∈ {workday, nonworkday}; `Calendar.holidays` exists | P01 adds `ctx.dayclass` ∈ {workday, weekend, holiday, makeup} and `ctx.dom`, `ctx.mend` (§6.2.3). |
| Many servers running one application | one `Observation.system` per adapter mapping | System families (§6.20). |

---

## 3. Requirement-to-design traceability

The requirement (verbatim, split into sentences S1–S21). Every row names the
mechanism, where it is specified, the engines, and the gate that measures it.

| # | Requirement sentence (verbatim) | Mechanism | § | Engines | Gate |
|---|---|---|---|---|---|
| S1 | 第三个库的算法，注意不能是遍历所有的用户和服务器的行为(一旦参数多或数据量大就会把资源消耗死) | No per-(IP × metric) enumeration anywhere in the core: one pattern tree per (system, event kind) with a node budget; all counts in bounded decayed sketches; per-event cost O(depth × touched nodes); only events of the tick are touched (active set); learning on a budgeted, stratified sample with Horvitz–Thompson weights; attribute work per event capped (A_ev); per-IP state only for exceptions, bindings' heavy hitters and (optionally) MinHash signatures, every per-IP structure an LRU or heavy-hitter sketch with a cap. **Servers too:** systems running one application share one tree (system families), per-system memory and periodic work are allocated by activity (tier XS for idle systems) and periodic fits touch only nodes with new evidence (§6.20). B01–B30 move to active-set + earned-model processing. | 6.2, 6.5, 6.20, 7, 10 | P00, P04, P12, P15, all | PG4 |
| S2 | 而是通过算法不断提升优化画像的准确度，慢慢得到一类行为或精准到一个用户的行为 | Top-down specialisation: root → route/action → group/prefix/time → single-IP exception, each step taken only when an anytime-valid e-value test passes (false specialisation ≤ 2⁻¹⁰ per candidate under continuous monitoring), the total code length shrinks (MDL), and the choice is stable under a time-uniform empirical-Bernstein bound. Specificity is an outcome, not a setting. | 6.5, 6.7 | P04 | PG1, PG3 |
| S3 | 用的时间越长越精准 | Confidence is measured on the confidence channel (§6.9.4), which keeps growing while behaviour is stationary and restarts from the new state after an accepted change (a fixed exponential decay alone would cap confidence at rate × H/ln 2 forever). Every constraint carries a confidence that grows with evidence (exchangeability coverage, Good–Turing closedness, Beta posterior bounds, e-values); lifecycle candidate → confirmed → stable. The plateau that remains (rate × H_l / ln 2 evidence units if nothing ever changes) is stated in §7.1. | 6.8, 6.9.4, 6.10–6.13 | P04, P06–P09 | PG2 |
| S4 | 并且会随着行为的动态发展而动态变化 | Forward-decayed statistics at three half-lives; ADWIN on per-node log-loss; Page–Hinkley on content; Hoeffding-Adaptive-Tree alternates; EFDT split revision; prune/merge; lifecycle stale → retired; versioning and lineage; trust-gated acceptance so that attacks are not learned. | 6.6, 6.8, 6.9 | P04, P06–P10 | PG5 |
| S5 | 下面我举一个例子便于理解，但不限于此：比如通过流量慢慢学习到早上9点到9点21分综合部的用户会访问OA系统(192.168.100.100的8080端口)的登录页面 | System address = learned root invariant `net.dst`; "login page" = route node `POST /login`; "09:00–09:21" = P09 activity window (Bayesian Blocks on the node's local arrival minutes, per day type); "综合部的用户" = P11 group used as a split level of the who hierarchy. | 6.13, 6.15 | P02, P04, P09, P11 | PG1, PG3 |
| S6 | ip清单为192.168.1.21、192.168.1.23、10.168.7.121 | Who summary per node = hierarchical heavy hitters over the IP hierarchy; rendered as an explicit IP list when ≤ 8 IPs cover ≥ 95 % of mass with small unseen mass. | 5.5.3, 6.17 | P04, P14 | PG3 |
| S7 | 提交的数据量90%为1KB-2KB，100%不会小于0.5KB或大于3KB | P06: central band from the node's quantile sketch, rounded to a 1-2-5 grid while keeping coverage in [0.88, 0.95]; hard bounds = observed range with exchangeability coverage (n−1)/(n+1) plus a GPD tail for scoring. | 6.10 | P06 | PG1 |
| S8 | 提交的内容里都是包含'username='字样且后边内容不超过10个字符 | P00 parses the body into `body.keys` and `body.kv.<key>`; P07 learns required keys (present in ≥ 99 %) and a value grammar per key by anti-unification of character-class shapes; length bounds are the observed range with a coverage bound, so "不超过10个字符" is learned as `[a-z]{4}` for jack/rose/mike and becomes `{…,10}` only if 10-character values occur (or an operator pins the business rule via B23). | 6.11 | P00, P07 | PG1 |
| S9 | 且192.168.1.21对应的是'username=jack'、192.168.1.23对应的是'username=rose'、10.168.7.121对应的是'username=mike' | P08 approximate functional dependencies `net.src → body.kv.username` with per-IP purity (empirical-Bayes Beta lower bound) and global g3 error. | 6.12 | P08 | PG1, PG6 |
| S10 | 后续192.168.1.21访问业务审批页面、特征是……，下午5点提交报告数据到生成报告页面…… | P10 workflow mining on sessions (directly-follows graph, heuristics-miner dependency, time-delta bands, required predecessors) with P09 time anchors per action (17:00). | 6.13, 6.14 | P09, P10 | PG1 |
| S11 | 按照这些行为举例自动学习到这就是综合部的3个用户的日常工作特征，也是这个综合部OA业务系统的日常行为画像 | One pattern store, two projections (group view, system view), statements in zh/en. | 6.17 | P13, P14 | PG1 |
| S12 | 但是有可能面对有些业务系统特别是大量用户使用的业务系统，用户IP没有规律的时候，不论哪个IP都会访问登录页面，这样学习到的就不会是某一个IP，可能是某几个IP段或者某几个IP区域，或者直接IP不作为特征 | The who hierarchy has prefix, learned-region and `*` levels; splits and summaries pick the level that pays; P05 drops IP as a split candidate/target when it carries no information; P12 sets the system's who-granularity (ip / prefix / group / none) by measured utility. | 5.4.1, 6.4, 6.18 | P04, P05, P11, P12 | PG3, PG8 |
| S13 | 所有的用户都是访问的某个页面系统某个路由时会执行什么动作且动作的详细特征或聚类出动作的特征约束条件是什么，也是这个业务系统的特征画像 | Route/action nodes with content constraints; content attributes (key-set, size bin, status) are split candidates too, so the tree performs divisive predictive clustering of actions into variants, each with its own constraints. | 6.5.2 | P04, P06–P08 | PG1 |
| S14 | 按照这个算法，比如最终我可以得出，站在用户行为视角：综合部访问哪几个业务都干什么，如果访问财务系统去审批就是异常 | Group view (group → systems → actions → when → content) and who-violation typing in P03. | 6.16, 6.17 | P03, P14 | PG6 |
| S15 | 站在业务系统视角，OA服务器的某类人会在哪个时间段访问我什么页面干什么事，财务系统的审批只有财务部某一个IP来访问且行为特征是什么样 | System view (system → route/action → who → when → content), with the who summary's closedness confidence. | 6.17 | P14 | PG1, PG3 |
| S16 | 注意！注意！以上只是举了一个例子，不局限于这些分类这些指标，我们有非常多的指标并且指标可能还会动态增加，不要写死用哪些指标 | Open attribute space: generic extraction of every Observation field, every scalar leaf of `extra.l7`/`extra.meta`, every raw/derived metric name (window events); dynamic registry with schema inference; no attribute list in code. | 5.2, 6.2, 6.3 | P00, P01, P02 | PG7 |
| S17 | 而是动态的学习到并确认用到什么指标、保留什么指标、保留什么特征等等各种细节 | P05 roles (split candidate / target / invariant / shape-only / redundant / dropped) per system and per node, with hysteresis and periodic re-probing; the kept set is versioned. | 6.4 | P05 | PG7 |
| S18 | 并且有可能不是一个引擎一个视角来考虑这些算法，有可能会是多个算法，从不同方面叠加起来，才能把一个业务系统画像看全面 | 16 focused engines (P00–P15) plus the reused B-engines, each contributing a facet; composition in P13. | 6.17, 8 | all | PG1 |
| S19 | 就像看一个人有外观维度、生物学特征、社会属性等等，外观维度又包括身高、体重、五官、常穿的衣服，生物学特征又包括血型、右位心、……，社会属性又包括出生地、父母关系、在哪上班、什么职务、常交往人群等等 | Dynamic facet registry with nested sub-facets (facet tree) per subject (system, group, IP, class); engines declare facets at runtime. | 6.17 | P13 | PG1 |
| S20 | 同时，要考虑真实环境下各种情况的适用性，不同的场景怎么自动适配哪些算法引擎 | P12 system characteriser + strategy selector: declared preconditions and cost, measured prequential utility in one currency (bits/event minus λ_c × µs/event), Hedge (full-information dimensions) and budgeted UCB (dimensions that must run), daily re-evaluation; B-engine applicability by the same characteristics. Real-world situations (NAT, proxies, DHCP/VPN, encryption, shared terminals, service accounts, batch jobs, holidays, month-end, cold start, schema change, scanners, sensor sampling) each have a detection signal and an automatic response (§6.21). | 6.18, 6.20, 6.21 | P12, P00, P01, P04, P08 | PG8, PG11 |
| S21 | 从而在真实环境中得到想要的精准行为画像集合 | Versioned pattern store as the profile set; gates PG1–PG11 on a generator whose truth is the persona program itself, plus a perturbation pack (O-real) of real-world conditions and a red-team variant never used for tuning; a real-log pilot is required before any accuracy claim (§14.2). | 11, 12 | all | PG1–PG11 |

Standing constraints: users are IPs / IP classes (conventions; §5.1.3);
legal factors out of scope (value retention is a switch, §5.1.3);
libraries raw → derived → behaviour → signatures (P00 raw, P01 derived,
P02–P15 behaviour; lib-4 optional input §9.3).

### 3.1 Measured status per sentence (round 2, final code; §16.10)

Pack O, seeds 0–4, 21 days, `progressive_decision`; "a/5" = seeds passing; medians [min–max]; round 1
(HEAD 6001027, seeds 0–2) in parentheses. Status: **met** (the gate's check passes), **partial**
(mechanism works on the example, gate not passed), **not met**, **not measured**.

| # | Measured (round 2) | Status |
|---|---|---|
| S1 | 7-day points: memory slope vs IPs 0.036, vs attributes 0.147 (0.23), CPU/event vs IPs −0.06 — met; vs servers 0.21 (round-1 3-day point, not re-run); CPU/event vs attributes 0.236 and absolute cost (scoring p95 0.42–0.70 ms per scored event, learning p95 1.8–2.7 ms per learned event; targets 100 / 250 µs) not met; bounded decision chain on pack O: identical gates, chain CPU −63 % | partial |
| S2 | recall@14 0.58 [0.53–0.58], precision 0.53 [0.52–0.57] (0.21 / 0.34); 综合部 login node = the 3 IPs 4/5 (1/3); finance approval = 192.168.2.10 only 5/5 (0/3); portal login at prefix/region level 5/5 (0/3) | not met (large gain) |
| S3 | median recall day 5 / 10 / 21: 0.22 / 0.55 / 0.60 (0.16 / 0.23 / 0.29); recall non-decreasing (±0.05) 2/5; ECE day 14 0.30 [0.28–0.34], day 21 0.36 (0.39 / 0.33); median stated confidence does not rise (day 7 0.43 → day 21 0.35) | partial (recall rises; confidence neither rises nor is calibrated) |
| S4 | D2 rename rebound 5/5 (3/3); .21 still `jack` on day 21 5/5 (0/3); D1 new window 3/5 (0/3); D3, D4, D5 0/5 | partial |
| S5 | before D1 (truth 09:00–09:21): seeds 1–4 state on day 11 "工作日 09:00–09:21 / 09:00–09:20 / 09:03–09:20 / 09:03–09:20 … 综合部（10.168.7.121、192.168.1.21、192.168.1.23）访问 POST /login" (the department's part of the shared login node); day-21 window (truth 08:30–08:51) IoU passes 3/5 | partial |
| S6 | day 21: the 综合部 login node names exactly the 3 IPs 4/5 (1/3) | partial |
| S7 | 90 % in 1–2 KB 3/5; all in 0.5–3 KB 1/5 (the department node is split off on day 10–15 and starts with an empty numeric summary) | not met |
| S8 | `username=` required and `[a-z]{4}(\.[a-z])?` 5/5 (2/3); seed 0 also the closed set {jack, mike, mike.w, rose} | met (example clause) |
| S9 | 3 bindings on day 21 4/5 (1/3 each seed); GA + FIN 6 bindings at day 14 2/6 (seeds 0–3), 0/6 (seed 4) | partial |
| S10 | .21 approvals 5/5; 17:00 report by .23 / .121 with the form page as required predecessor 5/5; A5 detected 5/5 (0/3) | met (example clause) |
| S11 | both views 5/5; department view `class:grp:dept:综合部` lists the 3 IPs and who does what 5/5; the name comes from `who_group_names` | met (via the department view) |
| S12 | portal login who at prefix / region 5/5, no portal bindings, no single-IP exceptions; DEV stated as its configured pool 5/5; DEV pool one learned group 0/5 | partial |
| S13 | content component of recall 0.61 [0.58–0.63] (0.53); required keys missing where `body.keys` is not a P05 target | partial |
| S14 | department view states "综合部 在 finance 中从未执行写操作 …；192.168.1.23 的尝试被判定为越权（未学习）" 5/5; A1 detected 5/5 (3/3) | met |
| S15 | "192.168.2.10访问 POST /fin/approval/{num}/approve" on day 21 5/5 (0/3); GET /docs stated per department ("某类人") | met |
| S16 | registered in the first tick 1.0; role within 24 h 1.0 | met |
| S17 | type correct 1.0 (0.67); informative kept 1.0; noise dropped and constants → invariants not measured | met (measured items) |
| S18 | 16 P engines + B24–B29 run together on pack O | met (structure) |
| S19 | 10 top facets, 22 default sub-facets; B15/B16/B13/B10 sub-facets not registered | met (structure) |
| S20 | arms in truth 0.83 [0.67–0.83] (0.50; target 0.95); switches after day 7 ≤ 2; who level within 5 % of the offline best 2/5; O-real not run | not met |
| S21 | 5 seeds and O-red (never tuned on: recall@14 0.41, anomalies 7/10) run; O-real and a real-log pilot not done | not measured (real world) |

---

## 4. Architecture overview

### 4.1 Principles (additive to architecture.md §0)

- **PPC-1 Events, not entity rows.** The core learns from behaviour events; per-(IP, tick) vectors remain for the existing spine but are not the unit of profiling.
- **PPC-2 General first, specific when proven.** Every system starts at the root pattern. A specialisation exists only while its evidence exceeds its description cost (MDL). Specific-IP models are exceptions.
- **PPC-3 Bounded by construction.** Every structure has a declared cap (nodes, learning leaves, sketch sizes, attributes, samples, signatures). Caps are enforced by eviction rules with explicit utility, never by silent truncation. Memory is O(budget), not O(#IPs × #metrics).
- **PPC-4 Open schema.** No code lists which attributes exist. Types, hierarchies and roles are inferred and versioned.
- **PPC-5 Prequential everywhere.** Every model is scored on an event *before* it learns from it; the same prequential log-loss is the split evidence, the drift signal, the attribute utility, the earned-model test and the strategy utility. One currency: bits per event.
- **PPC-6 Learn only what is trusted.** Learning uses the event weight × trust of its IP (B28, previous tick); quarantined IPs' events are held, not learned (§6.9.3).
- **PPC-7 Store-only coupling.** P engines follow the lib-3 rule: no engine imports another engine module; shared maths in `engines/behavior/lib/p*.py`; store names in §5.6.
- **PPC-8 Explain in natural units.** Every learned constraint can be rendered as a sentence with support, confidence, first/last seen and version.
- **PPC-9 Mass is not evidence.** Aggregation weights, HT weights and sensor sampling rates scale *mass* (distribution estimates); they never scale *evidence*. A learned row carries at most one evidence unit (§6.5.4). Every predictive used for a test or a p-value is evidence-scaled: `p(v) = (share_mass(v) · n + α · p_parent(v)) / (n + α)` with n = evidence, so a row with HT weight 1 000 does not make a node look 1 000 times more certain.
- **PPC-10 Servers are not enumerated either.** Systems that run one application share a tree (§6.20); per-system memory, LRU caps and periodic work are allocated by activity and bounded by a global budget, and periodic fits touch only nodes that received evidence since their last fit.
- **PPC-11 Anytime validity.** Every structural decision that is re-checked as data arrives (split, exception, closure used for HIGH severities) uses a test that stays valid under continuous monitoring (e-values and Ville's inequality, time-uniform bounds), so "check every 32 units forever" does not inflate its error.

### 4.2 Data flow per tick

```
Observations ──► [raw]  R1..R3 ──► P00 event_builder ──► evt.batch (per system)
                                   (open attrs, body kv parse, ev_sample expansion,
                                    stratified learning sample, HT weights)
             ──► [derived] D0..D2 ──► P01 event_context ──► evt.ctx (aligned: tod, daytype,
                                   session id/pos, prev action, think) + evt.win (metric windows)
             ──► [behaviour]
                 P15 resource_governor (budgets, active set, earned sets; reads t−1)
                 B01..B18, B19..B21 (bounded mode, §10)
                 P02 attr_registry ─► model.attr          (schema, stats, hierarchies)
                 P05 attr_select   ─► model.attrsel       (roles, kept sets; hourly)
                 P03 conformity    ─► pat.assign, behavior.score[conf_*], pattern_violation events
                     (routes with the model as of t−1: prequential)
                 P04 pattern_tree  ─► model.ptree         (learn t−D events with trust; split/merge/
                                                           exceptions/lifecycle/drift)
                 P06 content_bounds, P07 payload_grammar, P08 binding, P09 time_window
                                   ─► model.pbounds / pgrammar / pbind / pwin (fitted constraints)
                 P10 workflow      ─► model.pflow
                 P11 who_groups    ─► model.who_groups   (daily; feeds the IP hierarchy)
                 P12 system_profile─► model.sysprof      (characteristics, strategy; daily)
                 B23 feedback, B24 calibration (conf_* included), B25 fusion, B26 risk,
                 B27 incident, B28 governor, B29 explain,
                 P13 facets, P14 views, B30 portrait (embeds the facet tree)  (every 2 h, on read)
             ──► [signature] rule_match, correlation (optional pat.* inputs, §9.3)
```

Closed loops (each converges because its producer uses hysteresis and stable ids):
P11 groups → IP hierarchy level `grp` → P04 splits on `grp` → sharper patterns →
P11 signatures; P05 kept sets → P04 targets → node entropies → P05 utilities;
P12 strategy → which engines run → measured utility → P12.

Which batch each learner reads (the poisoning rule of §6.9.3 applies to all of them):
P03 and P10's *session state* use the current tick (scoring needs it); P02's statistics,
P04, P10's DFG *counts* and P11's signatures use the learned rows of tick t − D with trust.
Only the *registration* of a new attribute name in P02 happens on the current tick.

### 4.3 Engines (summary; cards in §8)

| Id | Name (store name) | Layer | Cadence | Focus |
|---|---|---|---|---|
| P00 | `raw.event` EventBuilder | raw | every tick | Observations → open-attribute event batches; body/query parsing; sampling |
| P01 | `derived.event_context` EventContext | derived | every tick | local time, day type, sessions, previous action, think time; metric-window events |
| P02 | `behavior.attr_registry` AttributeRegistry | behaviour | every tick (sampled) | schema inference, stats, hierarchies |
| P03 | `behavior.conformity` Conformity | behaviour | every tick | route every event to its most specific confident pattern; typed violation p-values |
| P04 | `behavior.pattern_tree` PatternTree | behaviour | every tick | counting, evidence, split/merge/revise, exceptions, lifecycle, drift |
| P05 | `behavior.attr_select` AttributeSelection | behaviour | 1 h | roles and kept sets per system and node |
| P06 | `behavior.content_bounds` ContentBounds | behaviour | 1 h | numeric bands and hard bounds |
| P07 | `behavior.payload_grammar` PayloadGrammar | behaviour | 1 h | key sets, value grammars, closed value sets |
| P08 | `behavior.binding` Binding | behaviour | 1 h | approximate FDs (IP → value, value → IP) |
| P09 | `behavior.time_window` TimeWindow | behaviour | 6 h | activity windows per node and system |
| P10 | `behavior.workflow` Workflow | behaviour | every tick (count) / 6 h (mine) | directly-follows graphs, workflows, required predecessors |
| P11 | `behavior.who_groups` WhoGroups | behaviour | every tick (signatures) / 24 h (cluster) | behavioural groups, prefix covers, names |
| P12 | `behavior.system_profile` SystemProfile | behaviour | 24 h | system characterisation, strategy selection, system families (§6.20) |
| P13 | `behavior.facets` Facets | behaviour | 2 h / on read | facet registry, portrait composition |
| P14 | `behavior.views` Views | behaviour | 2 h / on read | group and system views, zh/en statements |
| P15 | `behavior.resource_governor` ResourceGovernor | behaviour (first) | every tick | budgets, active set, earned sets, degradation |

---

## 5. Data model

### 5.1 Capture extension (library 1 input contract)

All three additions travel in `Observation.extra`, so `models/schema.py` is not
changed and every existing raw engine ignores them (they read only `count`,
`bytes_*_total`, `ts_sample`, `retransmits_total`).

#### 5.1.1 `extra['l7']` — application payload view (optional, per record)

```
extra['l7'] = {
  'body':       str,            # request body prefix, at most BODY_CAP = 4096 bytes, decoded utf-8 'replace'
  'body_len':   int,            # full request body length in bytes (Content-Length or measured)
  'body_trunc': bool,           # body was cut at BODY_CAP
  'body_type':  str,            # adapter hint: 'form' | 'json' | 'multipart' | 'xml' | 'text' | '' (unknown)
  'query':      str,            # raw query string without '?', values included
  'headers':    {name: str},    # request headers the adapter decodes (lower-case names), cookies excluded
  'resp_len':   int,            # response body length (optional)
}
extra['meta'] = {name: scalar}  # any further adapter / WAF / proxy field (e.g. 'waf.score': 3)
```

Sources: SPAN decode of cleartext HTTP (the decoder already reassembles the
request), reverse-proxy / WAF logs with request-body logging, or ICAP. For
TLS-terminated traffic seen after the terminator the same fields apply; for opaque
TLS they are absent (that absence is what P12 measures as payload visibility).

#### 5.1.2 `extra['ev_sample']` — per-event rows inside aggregated records

Aggregated records (generator Δt ≥ 900 s, NetFlow-style logs) lose per-event sizes,
which the requirement's "90 % of submissions are 1–2 KB" needs. Adapters that
aggregate attach up to EV_SAMPLE_MAX = 64 rows, drawn uniformly without
replacement from the w events of the record:

```
extra['ev_sample'] = [ {'o': float,          # offset from obs.ts in seconds (as ts_sample)
                        'up': int, 'down': int,
                        'st': int,           # HTTP status (optional)
                        'l7': {...},         # optional per-event l7 view (§5.1.1)
                        'meta': {...}}, ... ]  # optional per-event adapter / WAF fields (meta.*)
```

P00 expands each row into one event with weight w / len(ev_sample). Without
`ev_sample`, P00 emits one event per record with the record means and sets the
event flag `approx = 1`; P02 records the share of approx events per attribute and
P06 does not publish hard bounds for a node whose approx share exceeds 0.2 (it
publishes the band with `approx` in the statement).

#### 5.1.3 Value retention policy

`ctx.config['progressive']['value_policy']` maps attribute-name globs to one of
`clear` (the value is kept), `hmac` (HMAC-SHA256 with a deployment key, first 12 hex
chars; equality, bindings and closed sets still work, rendering shows `<h:3fa2…>`),
`shape` (only the value-shape of §5.4.5 is kept). Defaults:

| Glob / rule | Default (production and eval) | Why |
|---|---|---|
| `body.kv.*pass*`, `*pwd*`, `*token*`, `*secret*`, `*otp*`, `*captcha*`, `hdr.authorization`, `hdr.cookie` (globs are config, `value_policy.secret_globs`) | shape | secrets must never be stored by the platform (a security property of the platform itself, not a legal one) |
| any value that `lib/template._looks_random` classifies as random, or with ≥ 3.5 bits/char and length ≥ 16 | shape | catches secrets and nonces under unrecognised key names; their identity carries no behaviour, their shape does |
| any value longer than V_len = 64 characters | shape (+ exact length) | bounds memory; long free text is profiled by shape, length and Drain template |
| `body.kv.*`, `q.kv.*`, `hdr.*` otherwise | clear | the requirement's own example renders `username=jack` |
| everything else | clear | |

`hmac` is available per glob for deployments that want equality without clear values.
The policy is applied in P00, before anything is stored. For the PPC this supersedes the
lib-3 privacy line "query strings keep parameter names only" (decisions.md), because the
requirement asks for the values and legal factors are out of scope by the standing
instruction; the lead confirms this default (Appendix A, open item). The subject of every
profile stays the IP: a statement says "192.168.1.21 submits username=jack", never
"jack is …".

#### 5.1.4 Who resolution, session key, sensor sampling

- **Who resolution.** `config['progressive']['trusted_proxies']` lists proxy / load-balancer
  addresses (CIDRs). For a request whose `Observation.src` is trusted and that carries
  `hdr.x-forwarded-for` (or `forwarded`, `x-real-ip`, configurable), P00 sets `net.src` to the
  right-most address in the chain that is not trusted and keeps the transport source as
  `net.peer_src`. Without the config, P02 reports `net.src` with one dominant value that
  carries most of a system's traffic (share ≥ 0.9 over ≥ 1 d with ≥ 50 distinct
  `client.stack` or `sess.key` values behind it) as a `snat_suspect` characteristic; P12 then
  runs the system in who = none (§6.18) and the view says "来源地址被代理转换，未配置可信代理，
  IP 不作为特征" rather than profiling the proxy as a user.
- **Session key.** When the adapter decodes cookies or bearer tokens, it may pass
  `extra['l7']['sess']` = the value of the session cookie named in
  `config['progressive']['session_cookies']` (default `JSESSIONID`, `PHPSESSID`,
  `ASP.NET_SessionId`, `sid`, `session`). P00 stores only `sess.key` =
  HMAC-SHA256(deployment key, value)[:12]. P01 then keys sessions by (ip, sess.key), which
  separates users behind one NAT address or one shared terminal (§6.21).
- **Sensor sampling.** `extra['sample_rate']` (packets or flows sampled 1:k at the sensor)
  multiplies the row mass by k; evidence is unchanged (PPC-9). Claims such as "100 % in
  [a, b]" are then about observed rows and say so (§6.10).

### 5.2 Behaviour event and event batch

#### 5.2.1 Event kinds

| Kind | Code | One event per | Producer |
|---|---|---|---|
| `txn` | 0 | HTTP request, TLS connection without L7, DNS query, L4 flow without L7, active-probe result is excluded | P00 |
| `win` | 1 | (system, IP, H-grain decision tick) with any activity | P01 |

Each (system, kind) has its own pattern tree. The channel (`ev.ch` ∈ {http, tls,
dns, l4}) is an ordinary attribute and usually the first split of the `txn` tree.

#### 5.2.2 Attribute naming

`<ns>.<path>`, lower-case, `.`-separated, at most 96 characters (longer names are
cut and suffixed with an 8-hex blake2b digest). Namespaces:

| ns | Content | Examples |
|---|---|---|
| `net` | Observation L3/L4 fields | `net.src` (IP, the who), `net.dst` (`peer:dport`), `net.bytes_up`, `net.bytes_down`, `net.dur_ms`, `net.rtt_ms`, `net.l4`, `net.ttl` |
| `http` | L7 request fields | `http.method`, `http.host`, `http.path` (masked per policy), `http.route` (R2 template), `http.status`, `http.sclass`, `http.ua`, `http.ctype` |
| `tls`, `dns` | handshake / query fields | `tls.sni`, `tls.ver`, `tls.ja3`, `dns.qname`, `dns.qtype`, `dns.rcode` |
| `client` | R3 stack token | `client.stack` |
| `body` | parsed request body | `body.fmt`, `body.len`, `body.keys` (set), `body.kv.<key>` (value), `body.tpl` (Drain template of non-structured bodies) |
| `q` | parsed query string | `q.keys` (set), `q.kv.<key>` |
| `hdr` | request headers | `hdr.x-client-ver`, `hdr.referer` (route-templated) |
| `meta` | `extra['meta']` leaves | `meta.waf.score` |
| `ctx` | P01 context | `ctx.tod_min`, `ctx.dow`, `ctx.daytype`, `ctx.dayclass`, `ctx.dom`, `ctx.mend`, `ctx.sid`, `ctx.sess_pos`, `ctx.prev_route`, `ctx.prev_act`, `ctx.think_s`, `ctx.sess_age_s` |
| `sess` | session key (§5.1.4) | `sess.key` (HMAC) |
| `m` | window-event metrics (kind `win` only) | `m.http.requests`, `m.derived.upload_dominance`, `m.feature.bytes_up`, any new metric name |
| `ev` | event bookkeeping (not a target) | `ev.ch`, `ev.approx` |

Nested structures flatten with `.`; JSON arrays flatten to `<key>[]` holding the
set of scalar elements (at most 16) plus `<key>[].n` (length). `net.src` is the resolved
client (§5.1.4); `net.peer_src` the transport source when they differ.

**Absence is a value.** An attribute that a registered attribute's events usually carry
but this event lacks is read as the reserved value `⊥` at every level of its hierarchy
(`gen(a, ℓ, ⊥) = ⊥`). Absence is therefore learnable where it is informative (GET requests
have no body keys) and routable (§5.5.1), and a system-wide disappearance of an
attribute is recognised as a schema change rather than a behaviour change (§6.3).

#### 5.2.3 `EventBatch` (lib/pevent.py)

Columnar and sparse; one batch per (system, kind, tick).

```python
@dataclass(slots=True)
class EventBatch:
    system: str
    kind: int                       # 0 txn, 1 win
    t0: float; t1: float            # tick bounds (t1 = ctx.now)
    n: int
    ts: np.ndarray                  # float64[n], event time
    ip: np.ndarray                  # int32[n], index into ips
    ips: List[str]
    w: np.ndarray                   # float32[n], row mass before HT: count share x sensor sample rate
    pi: np.ndarray                  # float32[n], learning inclusion probability (1 = always learned)
    learn: np.ndarray               # bool[n], selected into the learning sample
    flags: np.ndarray               # uint8[n], bit0 approx, bit1 body_trunc
    cols: Dict[str, Col]            # open attribute map
@dataclass(slots=True)
class Col:
    rows: np.ndarray                # int32[k], sorted row indices where the attribute is present
    vals: np.ndarray                # float64[k] (numeric / time) or object[k] (str, frozenset)
```

Row index = event id within the batch; `(system, kind, t1, row)` is the global event
id. P01's `evt.ctx` batch carries only `cols` and is row-aligned with the `evt.batch`
of the same (system, kind, t1). P03's `pat.assign` batch is row-aligned as well.
Horvitz–Thompson: a learned event contributes mass `w / pi`; its evidence is computed by
P04 (§6.5.4) and never exceeds 1. After P03, P04 (learning of t − D) and P10 have run,
the store compacts a batch to its learned and held rows and to the columns that are
split candidates, targets or context (`compact_batch`, W-P0), so the retained batches cost
`(Δt + D) · min(rate, e_rate) · ≈ 0.2 KB` per system (§7.2), not all events of 2 h.

### 5.3 Attribute registry record (P02; `model.attr@(s,'__system__')`)

```
attr[a] = {
  'name': a, 'ns': str, 'first_seen': ts, 'last_seen': ts, 'version': int,
  'type': 'categorical'|'numeric'|'ordinal'|'ip'|'time'|'set'|'text'|'unknown',
  'type_evidence': {'n': float, 'num': float, 'int': float, 'ip': float, 'set': float,
                    'kv': float, 'json': float, 'mean_len': float},     # decayed H_m
  'locked': bool,                 # type locked after n >= 500 and >= 1 day
  'parse_as': None|'form'|'json', # P00 re-parses a text attribute found to be structured
  'coverage': float,              # present / events, decayed H_m
  'card': HLL(p=10),              # distinct values (on the level-1 generalisation for text)
  'top': DecayedSpaceSaving(k=32),# of level-1 generalised values
  'num': {'qs': TDigest(delta=50), 'log': bool, 'min': float, 'max': float} | None,
  'entropy': float,               # bits, Chao–Shen on top + other mass
  'stability': float,             # 1 - JSD(p_Hs, p_Hl) on top-k (0..1)
  'approx_share': float,
  'hier': {...},                  # §5.4, learned levels versioned separately
  'role_sys': str,                # written by P05 (split / target / invariant / shape / redundant / dropped / probe)
  'cost_us': float,               # measured P04 update cost per event for this attribute (P15)
}
```

Type inference rules (evaluated on the decayed evidence; ties resolved in this order):

1. `ip` if ≥ 99 % of values parse with `ipaddress.ip_address`.
2. `time` if the attribute is `ts`-like (name ends with `_ts` or `.ts`) and ≥ 99 % are finite floats in [1e9, 4e9].
3. `set` if values are frozensets / JSON arrays.
4. `numeric` if ≥ 98 % parse as finite floats and HLL distinct ≥ 16; `ordinal` if they parse and distinct < 16. Codes are *categorical*: a name hint (suffix list `type_hints.code`, default `status`, `code`, `port`, `qtype`, `rcode`, `method`; configuration, not code) proposes it, and the data decide when there is no hint: a parsed integer attribute is categorical when the numeric order carries no information about the targets (P05's U_s at the ordinal ℓ1 split is < 0.1 × U_s at the categorical value-group level).
5. `categorical` if distinct ≤ 256 or distinct / n ≤ 0.05.
6. `text` otherwise. A text attribute with `kv` share ≥ 0.8 (contains `=` separated by `&` or `;`) or `json` share ≥ 0.8 gets `parse_as` set; P00 then emits `<a>.keys` and `<a>.kv.<key>` from the next tick on.

`log` is set for a numeric attribute when all values are > 0 and the skewness of
log(v) is below the skewness of v (sizes, durations, counts).

### 5.4 Generalisation hierarchies (lib/phier.py)

`gen(a, ℓ, v)` maps a value to its level-ℓ representative; level 0 is the value
itself (after policy), the last level is `*`. Levels marked *learned* come from a
model and are versioned; a level whose model is not ready is skipped (maps to the
next coarser level). Each hierarchy exposes `levels(a)`, `gen(a, ℓ, v)`, and
`card_hint(a, ℓ)`.

#### 5.4.1 IP (`net.src`, any `ip` attribute)

| ℓ | Level | Source |
|---|---|---|
| 0 | /32 (/128) | value |
| 1 | /24 (/64) | fixed |
| 2 | /16 (/48) | fixed |
| 3 | `grp` behavioural group id | learned: P11 `model.who_groups.ip2g` (members only; non-members map to `grp:∅`) |
| 4 | `reg` region | config `ip_classes`, `dhcp_scopes` (first match), else P11 prefix covers, else `reg:∅` |
| 5 | `*` | |

`grp` and `reg` are not nested in the prefix levels; the hierarchy is a list of
alternative coarsenings ordered by typical cardinality. Splits may use any level;
the who summary (§5.5.3) tracks all. IPv6 uses /128 → /64 → /48; the /64 level absorbs
privacy (temporary) addresses that rotate daily, the same way `grp`/`reg` absorb DHCP
and VPN pools. An IP that B17 flags `shared_ip` (NAT, VDI, terminal server) or P00 flags
`snat_suspect` maps at level 0 to `shared:<ip>`: it stays visible as a who item but is
excluded from exceptions and per-IP bindings (§6.7, §6.12), because several users are
behind it.

#### 5.4.2 Route / path (`http.path`, `http.route`, `hdr.referer`)

ℓ0 raw path (policy-masked) → ℓ1 R2 template `http.route` (`{METHOD} host template`
without the status class) → ℓ2 first two path segments (`/fin/approval/*`) → ℓ3 first
segment → ℓ4 host → ℓ5 `*`.

#### 5.4.3 Time (`ts` of every event, via `ctx.*`)

ℓ0 local minute of day (`ctx.tod_min`, 0..1439) → ℓ1 15-min slot → ℓ2 learned system
window id (P09 system-root windows, e.g. `w:0900-0921`; outside every window →
`w:off`) → ℓ3 daypart (lib/timebins: wd_day, wd_night, nwd_day, nwd_night) → ℓ4 day
type (workday / non-workday) → ℓ5 `*`. Day type and time of day are also separate
attributes (`ctx.daytype`, `ctx.tod_min`) so a split may use either. Calendar context
is open in the same way: `ctx.dayclass` (workday / weekend / holiday / makeup workday,
from the configured calendar), `ctx.dom` (day of month, ordinal) and `ctx.mend` (one of
the last 3 workdays of the month) are ordinary split candidates, so a month-end batch
job or a make-up-Saturday schedule is learned when it pays and ignored otherwise.

#### 5.4.4 Numeric

ℓ0 value → ℓ1 decile-like bin: 8 bins from the registry's global t-digest at quantiles
{1/8, …, 7/8} on the log scale if `log`, frozen per registry version → ℓ2 quartile bin
(4) → ℓ3 median split (2) → ℓ4 `*`. Bin edges are recomputed by P02 at most daily
and only when the JSD between the old and new bin occupancy exceeds 0.05 (so
split statistics keyed by bin stay comparable).

#### 5.4.5 Text / payload values (`body.kv.*`, `q.kv.*`, `hdr.*`, `text`)

ℓ0 value → ℓ1 shape: the run-length class sequence, classes L = [a-z], U = [A-Z],
D = [0-9], literal punctuation and space kept, any other code point class X; runs
keep their exact length (`jack` → `L4`, `mike.w` → `L4 . L1`, `3fa2c9` → `D1 L2 D1 L1
D1` … mixed alnum runs longer than 8 collapse to `A{n}`) → ℓ2 shape with length
buckets {1, 2–3, 4–7, 8–15, 16–31, 32+} → ℓ3 charset set + length bucket
(`{L,.}:4–7`) → ℓ4 length bucket → ℓ5 `*`. Whole bodies that are not key-value
structured use Drain-lite (lib/template.py) templates as ℓ1 (`body.tpl`).

#### 5.4.6 Categorical, ordinal, set

Categorical: ℓ0 value → ℓ1 value group (learned: agglomerative merge of values
whose conditional target distributions in the PT root are within JSD 0.02 bits,
refreshed by P05 daily; ungrouped values map to themselves) → ℓ2 `*`.
Ordinal: ℓ0 value → ℓ1 ranges from the value order split at the median → ℓ2 `*`.
HTTP status has a fixed ℓ1 status class. Set: ℓ0 the frozenset → ℓ1 its template
(the set minus elements present in < 1 % of events with the attribute; so optional
rare keys fold away) → ℓ2 cardinality bucket → ℓ3 `*`.

### 5.5 Pattern tree and node layout (lib/pnode.py, lib/ptree.py)

#### 5.5.1 Tree

`model.ptree@(s,'__system__')` = `{'fmt': 1, 'version': int, 'kinds': {kind:
Tree}}`, where `Tree = {'root': nid, 'nodes': {nid: Node}, 'next_id': int,
'budget': {...}, 'lineage': Ring(4096)}`. Node ids are integers, never reused.
A node's split, when present, is `Split(attr, level, groups: List[FrozenSet[value]],
other: nid)`: child `i` holds the values of `groups[i]` at `(attr, level)`, and the
`other` child receives every value not in a group (so unseen values always land
in a general pattern). Exceptions hang from a node as `exc: {ip: nid}`.

Routing is `route(tree, event) → path [nid_0 … nid_leaf]`: at each node with a split,
compute `g = gen(attr, level, event[attr])` (`⊥` when the attribute is absent, §5.2.2)
and follow the group containing g, else `other`. `⊥` is grouped like any other value,
so absence gets its own child when it is informative. When P02 has declared the split
attribute `gone` (schema change, §6.3), P04 collapses the split (§6.6) instead of letting
every event fall into `other`, and P03 routes by the heaviest child until the collapse.
Depth ≤ D_max = 8.

#### 5.5.2 Node

```
Node:
  id, parent, depth, kind
  ctx: Tuple[(attr, level, values: frozenset, negated: bool)]  # conjunction from the root
  split: Split | None
  exc: {ip: nid}                        # IP exceptions (§6.7); exception nodes carry is_exc = True
  state: 'candidate'|'confirmed'|'stable'|'evolving'|'stale'|'retired'
  created, first_seen, last_seen, days (uint64 bitmap of the last 64 local dates + total distinct dates), version, cver
  mass: float32[3]                      # forward-decayed weight at H_s, H_m, H_l
  n_eff: float32[3]                     # decayed evidence units (§6.5.4) at H_s, H_m, H_l;
                                        # "n_eff" alone means H_m; n_c = the H_l entry (confidence channel)
  seg_start: ts                         # start of the current confidence segment (§6.9.4)
  who: WhoSummary                       # §5.5.3
  when: WhenSummary                     # §5.5.4
  targets: {attr: Summary}              # ≤ m_t (§5.5.5); chosen by P05
  rate_iph: TDigest                     # events per (IP, local hour) at this node (§6.16.4); route-level nodes
  inv: {attr: (level, value)}           # invariants (constant at this node)
  split_stats: SplitStats | None        # only on learning leaves (§6.5)
  xstats: {ip: ExcStats}                # only on confirmed nodes, ≤ k_x heavy IPs (§6.7)
  pairs: {(X, Y): PairSketch}           # binding pair counts requested by P08 (§6.12)
  adwin: ADWIN; ph: {attr: PageHinkley} # drift (§6.9)
  alt: nid | None                       # alternate subtree under evaluation (HAT)
  ref: RefSnapshot | None               # daily reference snapshot of fitted constraints (§6.8.3)
```

#### 5.5.3 Who summary (hierarchical heavy hitters)

Per IP level ℓ ∈ {/32, /24, /16, grp, reg}: `DecayedSpaceSaving(k=8)` (H_m), plus
`HLL(p=6)` of distinct /32, and per level the running prequential code length of the
IP under that level's model (§6.18.2). Conditioned counts (HHH semantics) are computed at
render/score time: `cond(p) = count(p) − Σ_{q HHH descendant of p} count(q)`.
Unseen mass per level (Good–Turing with an eviction correction):
`U_ℓ = (N1_ℓ + E_ℓ + 0.5)/(N_ℓ + 1)`, where N_ℓ is the evidence on the confidence
channel (units ω of §6.5.4 at H_l with segment reset, §6.9.4 — not raw mass, so bursts
and HT weights do not make a set look closed, and not H_m, whose plateau would cap
closedness at rate × 10 d forever), N1_ℓ the number of tracked items whose evidence is
< 1.5 (seen about once), and E_ℓ the evidence of arrivals that evicted an item from the
full sketch. The SpaceSaving counters therefore carry an evidence entry next to the
three mass entries. While the sketch is not
full, E = 0 and U is the Good–Turing estimate; under churn E grows, so U over-estimates the
unseen mass — the conservative direction for closedness (a churning population never
looks closed). The same convention is used for every unseen-mass estimate in this
document (keys, shapes, values, actions), written `N1` in formulas for brevity.

Across the route-level nodes, these per-node summaries *are* the system's two-dimensional
(route × IP-prefix/group) hierarchical heavy hitters: the tree supplies the route (and
time, content) dimension, each node's summary the IP dimension, so no separate
multi-dimensional HHH structure is kept.

#### 5.5.4 When summary

`hist96[daytype∈{wd,nwd}][96]` float32 (15-min slots, H_m); a weighted reservoir of
(daytype, local minute) of size R_t = 256 with time-decayed Efraimidis–Spirakis keys
(`key = u^(1/(w·2^((t−L)/H_m)))`, landmark L per node, rescaled when the exponent
exceeds 60); the reservoir is kept only when P09 asks for minute resolution
(`model.pwant`, §5.6), i.e. when ≥ 50 % of mass lies in ≤ 4 slots.

#### 5.5.5 Target summaries by type

| Type | Summary | Size (float32 unless stated) |
|---|---|---|
| categorical / ordinal / ip (non-who) | DecayedSpaceSaving(k=16) with counts at H_s, H_m, H_l; `other` mass per half-life | ≈ 16 × (key ref + 12 B) + 12 B |
| numeric | t-digest(δ = 50) at H_m (forward-decayed centroid weights), moments (n, mean, M2) of the transformed value at H_s, H_m, H_l, daily min/max ring (30 days), lower/upper exceedance reservoirs (64 each, for GPD) | ≈ 1.2 KB |
| text | DecayedSpaceSaving(k=8) of ℓ1 shapes, SpaceSaving(k=16) of exact values (if policy allows), length histogram (log2 buckets, 16), charset-class counts (8), unseen-shape N1 | ≈ 0.6 KB |
| set | DecayedSpaceSaving(k=8) of ℓ1 templates, per-element presence counts for the union of the top templates (≤ 32 elements) | ≈ 0.5 KB |

All numeric storage is packed into one float32 array per node (`lib/pnode.py`
offsets) to keep Python object overhead O(1) per node.

### 5.6 Store names

New store capability (`core/store.py`, contract addition "batch series"):
`add_batch(system, name, ts, obj)`, `batch_at(system, name, ts)`,
`batches_since(system, name, since) -> List[(ts, obj)]`, retention by age through
the retention table. Batches are not per-entity metrics, so they bypass the raw
pseudo-entity guard and `first_seen/last_seen`.

| Name | Kind | Key | Writer | Readers | Retention |
|---|---|---|---|---|---|
| `evt.batch` | batch (EventBatch, kind txn) | s | P00 | P01, P02, P03, P04, P10, P11 | D + 1 tick, compacted after the tick (§5.2.3) |
| `evt.win` | batch (EventBatch, kind win) | s | P01 | P02, P03, P04 | D + 1 tick, compacted |
| `evt.ctx` | batch (aligned cols) | s | P01 | P02, P03, P04, P09, P10 | D + 1 tick, compacted |
| `pat.assign` | batch: per event `leaf`, `conf_node`, `act` (action id, §6.14), `prev_act`, `act_node`, `exc` flag, `p_who`, `p_when`, `p_content`, `p_seq`, `p_novel`, `p_ev`, `vtype` bitmask, `damp` (outlier damping, §6.9.3) | s | P03 | P04, P10, P11, P12, P14 | D + 1 tick, compacted |
| `pat.rate` | batch: finished per-(ip, node) hourly counts (§6.16.4), not row-aligned | s | P03 | P04 | D + 1 tick |
| `model.attr` | model | (s, `__system__`) | P02 | P00, P03–P09, P12, P14 | persistent |
| `model.attrsel` | model: roles, kept sets, node overrides, redundancy map, versions | (s, `__system__`) | P05 | P02, P03, P04, P12 | persistent |
| `model.ptree` | model (§5.5) | (s, `__system__`) | P04 | P03, P05–P11, P13, P14 | persistent; daily checkpoint `ptree` |
| `model.pwant` | model: statistics requested from P04 per node (`pairs`, `minute_reservoir`, extra targets) | (s, `__system__`) | P06–P10 (each its own sub-key) | P04 | persistent |
| `model.pbounds` / `model.pgrammar` / `model.pbind` / `model.pwin` | model: fitted constraints per node id, versioned | (s, `__system__`) | P06 / P07 / P08 / P09 | P03, P04, P13, P14 | persistent |
| `model.pflow` | model: DFG sketches, workflows, required predecessors | (s, `__system__`) | P10 | P03, P13, P14 | persistent |
| `model.who_groups` | model: groups, `ip2g`, prefix covers, names, mode per system | (`__org__`, `__org__`) | P11 | lib/phier (via P03/P04), P13, P14, B02 (optional) | persistent |
| `model.sysprof` | model: characteristics, arms, utilities, chosen strategy, history | (s, `__system__`) | P12 | P00–P11, P15 | persistent |
| `model.sysfam` | model: system families, member → family, tree key per system, signatures (§6.20) | (`__org__`, `__org__`) | P12 | P00, P03, P04, P13, P14, P15 | persistent |
| `model.facets` | model: facet registry (declarations) | (`__org__`, `__org__`) | any engine (declare), P13 (own) | P13, P14, API | persistent |
| `model.pviews` | model: rendered views (JSON + zh/en statements) | (s, `__system__`), (`__org__`, `class:grp:<g>`) | P14 | API, B30 | persistent + `put_profile_version` |
| `model.budget` | model: caps, active set, earned sets per system | (`__org__`, `__org__`) | P15 | every bounded engine | persistent |
| `ops.budget` | derived series: measured ms / MB per engine and system | (s, `__system__`) | P15 | API | 8 d |
| `behavior.score[conf_*]`, `behavior.pm[conf_*]`, `behavior.axes` | existing vector/dict series, 5 new detector columns | (s, ip) | P03 | B24, B25, B29 | existing |
| events `pattern_violation`, `pattern_absent`, `pattern_confirmed`, `pattern_retired`, `pattern_replaced`, `pattern_drift`, `pattern_revived`, `binding_changed`, `group_formed`, `group_changed`, `attribute_new`, `attribute_gone`, `attribute_role`, `strategy_changed`, `family_changed` | BehaviorEvent kinds | (s, ip) or (s, `__system__`) | P03 / P04 / P08 / P11 / P02 / P05 / P12 | B25–B27, B29, API | existing |

---

## 6. Algorithms

### 6.1 Bounded, decayed, mergeable sketches (lib/psketch.py)

**Forward decay** (Cormode, Shkapenyuk, Srivastava, Xu, ICDE 2009). An item arriving at
time t with weight w is stored with weight `w · 2^((t − L)/H)` relative to a landmark L;
the decayed value at time T is `stored · 2^(−(T − L)/H)`. Decay is therefore an add at
update time and a multiply at read time, sums stay linear (mergeable, subtractable for
release/reject), and every sketch below accepts forward-decayed weights unchanged.
When `(T − L)/H > 60` the structure is rescaled: L ← T, all stored weights multiplied by
`2^(−(T − L_old)/H)`. Each sketch keeps the three half-lives H_s, H_m, H_l as a
length-3 weight vector per counter (one landmark per structure). Counters that feed a
test or a confidence also carry the evidence of §6.5.4 on the confidence channel
(one more float), per PPC-9.

| Sketch | Use | Guarantee | Size |
|---|---|---|---|
| `DecayedSpaceSaving(k)` (Metwally 2005; with forward decay, Cormode 2009) | categorical targets, who levels, route/value tops, DFG edges, per-IP signatures | decayed count error ≤ N_decayed / k; every item with share > 1/k is present; `count − error` is a guaranteed lower bound | k × (key + 3 mass floats + 1 evidence float (H_l) + 1 error float) |
| `HLL(p)` (lib/sketch.HyperLogLog reused; no decay: two epochs of 7 d rotated) | distinct IPs per node, attribute cardinality | σ ≈ 1.04/√(2^p) | 2^p bytes |
| `TDigest(δ)` (Dunning & Ertl 2019), forward-decayed centroid weights | numeric targets, registry bins | rank error O(q(1−q)/δ) | ≈ δ centroids × 2 floats |
| `CountMin(w=1024, d=4)` (Cormode & Muthukrishnan 2005), decayed | first-seen tests for high-cardinality text values in P02 (N1 estimates) | overestimate ≤ εN w.p. 1−δ, ε = e/w, δ = e^−d | 16 KB |
| `ADWIN` (Bifet & Gavaldà 2007), M = 5 buckets per row | per-node log-loss drift | false positive ≤ δ per test | O(M log W) |
| `PageHinkley(λ, δ)` | per-target mean shift | – | 4 floats |
| `WeightedReservoir(R)` A-ExpJ with decayed keys (Efraimidis & Spirakis 2006) | minute-of-day samples (P09), probe sample (P05), held events | exact weighted sampling without replacement | R rows |

All sketches serialise to plain dict/ndarray form (`to_dict/from_dict`) for
checkpoints and are mergeable (`merge(a, b)`), which prune/merge (§6.6) relies on
(SpaceSaving merge per Agarwal et al., "Mergeable summaries", PODS 2012).

### 6.2 Event construction and bounded sampling (P00, P01)

#### 6.2.1 P00 extraction

```
for obs in observations:                              # raw layer, after R2 and R3
    if obs.method is active or pseudo(obs.entity/system): continue
    s = obs.system
    tpl = m_template.templater(store, s)              # read-only; R2 already updated it this tick
    rows = obs.extra.ev_sample or [None]              # §5.1.2
    w_row = count(obs) * obs.extra.get('sample_rate', 1) / len(rows)   # mass only (PPC-9)
    base = generic_fields(obs)                        # every non-default Observation field -> attribute
    base['net.src'], base['net.peer_src'] = resolve_who(obs, trusted_proxies)   # §5.1.4
    base['net.dst'] = f"{obs.peer}:{obs.dst_port}"
    base['http.route'] = tpl.apply_path(host, method, path) if http   # read-only (§2)
    base['client.stack'] = stack_token(obs.ja3, obs.user_agent, obs.ttl, obs.win_size)
    base |= flatten('meta', obs.extra.get('meta'))
    for r in rows:
        e = dict(base); e.ts = obs.ts + (r.o if r else 0)
        if r: e['net.bytes_up'], e['net.bytes_down'], e['http.status'] = r.up, r.down, r.st
        l7 = (r and r.l7) or obs.extra.get('l7')
        if l7: e |= parse_l7(l7, registry_hints(s))   # body.fmt/len/keys/kv.*, q.keys/kv.*, hdr.*
        apply_value_policy(e)                         # §5.1.3 (globs, randomness, V_len)
        e.flags = approx if (rows == [None] and count(obs) > 1) else 0
        append(batch[s], e, w_row)
select_learning_sample(batch[s])                     # §6.2.2 (batches stay per system; P03/P04 route them into
                                                     #  the system's tree, which may be its family's, §6.20)
store.add_batch(s, 'evt.batch', now, batch[s])
```

`parse_l7`: form (`application/x-www-form-urlencoded` or sniffed `k=v&k=v`), JSON
(first 64 leaves, depth ≤ 4), multipart (part names + sizes; file parts give
`body.kv.<name>.len` and `.ctype`), XML (element path set only), otherwise Drain-lite
template `body.tpl`. At most K_BODY = 64 keys per event; keys beyond are counted in
`body.keys_extra`. Keys are lower-cased and templated (`items[3].name` →
`items[].name`). Registry hint `parse_as` (§5.3) makes any text attribute parse the
same way (a header that carries `k=v;k=v` becomes structured automatically).

#### 6.2.2 Learning sample (bounded learning cost, unbiased statistics)

Scoring (P03) sees every event. Learning (P02, P04, P10, P11) sees a stratified
sample so that the per-tick learning cost is at most `E_learn(s) = e_rate(s) · Δt`
events (default e_rate = 10 events/s per system, a cap that P15 lowers by budget share;
most systems never reach it). Stratum `k = (ev.ch, key)` where key is the value, at
its split level, of the attribute that the system tree's root splits on (the tree's own
first answer to "what separates behaviours here"); before the root has split, the
bootstrap key `http.route | tls.sni eTLD+1 | dns qname eTLD+1 | net.dst` is used. The
bootstrap is a seed, not a requirement: a system whose events carry none of these falls
back to `ev.ch` alone. With c_k
events of stratum k in the tick, inclusion probability is threshold sampling
(Duffield, Lund & Thorup 2005 "priority/threshold sampling", water-filling form):

```
π_i = min(1, τ / c_k(i)),   τ solves  Σ_k min(c_k, τ) = E_learn     (τ = ∞ when Σ c_k ≤ E_learn)
```

Every stratum keeps min(c_k, τ) events in expectation, so rare actions are always
learned while dominant ones are thinned; a learned event carries HT weight
`w / π`, which keeps every count, quantile and heavy-hitter estimate unbiased.
Selection uses `seeded_uniform(s, ts, row)` (lib/combine) so runs are reproducible.

#### 6.2.3 P01 context and window events

- `ctx.tod_min`, `ctx.dow`, `ctx.daytype` from `lib/timebins.local_datetime` and
  `day_type` with the configured calendar; `ctx.dayclass` from the same `Calendar`
  (`holidays`, `makeup_workdays`, weekday), `ctx.dom`, `ctx.mend`.
- A per-system *normal-day* flag for each local date (system mass within its H_l band of
  the same day class, and not a holiday): stale timers (§6.8.1) and drift acceptance
  (§6.9.2) count only normal days, so a seven-day national holiday neither makes a daily
  pattern stale nor is accepted as a new regime.
- Sessions: per (s, ip, sess.key) LRU state `(last_ts, sid, pos, last_route, last_act,
  start_ts)` (sess.key = `∅` when absent) with cap S_sess per system allocated by P15
  (default min(65 536, 4 × distinct session keys seen in 7 d)); least recently active is
  evicted, and an evicted key starts a new session on its next event, which is what a
  gap would do anyway.
  A new session starts when `ts − last_ts > G(s)`; G(s) is B10's `session_gap`
  (lib/m_seq) when present, else P01's own valley: a 64-bin histogram of log10
  inter-event gaps per system (decayed H_m), threshold by Otsu's method between the
  two largest modes, clamped to [60 s, 2 h]; default 30 min.
  Attributes: `ctx.sid`, `ctx.sess_pos`, `ctx.prev_route`, `ctx.think_s` (gap to the
  previous event of the session), `ctx.sess_age_s`.
- Window events (`evt.win`): for IPs active in the H grain that closes at this
  decision tick (canonical mode; every tick in tick mode), one event with
  `m.<name>` = the IP's fresh scalar value of each raw/derived metric name and each
  `feature.nat.h` column, read through `store.snapshot(s, ip, now, names=N)` where
  N = P05's kept `win` targets ∪ a rotating slice of the other registered names
  (A_win = 96 per event in total; the slice rotates by a crc32 phase so every name
  is probed at least once a day). New metric names therefore enter the registry
  the day they appear, without code changes. **Bounded per grain:** at most W_max(s)
  window events per system and grain (default 512, allocated by P15). When more IPs are
  active, the IPs are chosen by priority sampling (Duffield–Lund–Thorup) with priority
  weight = 1 for ordinary IPs and 10 for IPs that are earned, flagged by P03 in the grain,
  quarantined, or new to the system; each chosen event carries HT mass 1/π. Per-IP
  volume deviations of unchosen IPs are still scored by the B-library row detectors and
  by P03's intensity check on txn events (§6.16.4), so the cap loses learning speed, not
  coverage of the IPs that matter.

### 6.3 Schema inference and attribute statistics (P02)

Per learned event, P02 updates at most A_ev = 32 attribute records: every attribute
P05 marks `split` or `target` plus a uniform random subset of the rest with
inclusion probability `q = (32 − n_kept)/n_rest` (HT weight 1/q). Registration of
a new name is always done (a dict insert and an `attribute_new` INFO event). Type
rules are those of §5.3; the entropy estimate is Chao–Shen on the SpaceSaving top
plus other mass; stability `S_a = 1 − JSD(p_{H_s}, p_{H_l})` over the top-k (with
other). Hierarchy models (numeric bins, categorical value groups) are refreshed at
most daily and only if the bin occupancy JSD between old and new edges exceeds
0.05; a refresh bumps `attr.version`, and P04 re-keys split statistics that use
the attribute (split statistics are reset for that candidate only).

Statistics are updated from the learned rows of tick t − D with trust (the poisoning rule
of §6.9.3: an attacker's first hour must not move bin edges or value groups); only
registration uses the current tick.

**Schema change.** An attribute whose system coverage falls below 5 % of its H_l coverage
for one full normal day while the system's event mass is within its H_l band is declared
`gone` (`attribute_gone` INFO): P05 removes it from every role, P04 collapses splits on it
(§6.6) and marks its target summaries frozen, and P03 stops scoring it. A reappearance
revives it with its old hierarchy. This separates "the adapter stopped sending header X"
from "users stopped doing X", which would otherwise flood `other` branches and look like
drift everywhere.

Cap: A_max = 512 registered attributes per system (per family, §6.20). When full, a new name is
counted in `__overflow__` (HLL of names, total count) and admitted when an
attribute has been `dropped`, `gone` or unseen for 30 d (lowest coverage evicted first).

### 6.4 Attribute and feature selection (P05)

P05 answers "which metrics are used, kept, and at which detail" per system and per
node, every hour, from three sources: the registry (§6.3), node summaries in
`model.ptree`, and a probe sample `R_p` = 4096 learned events per (system, kind) with
all their attributes (R_p scaled down with the system's tier, §6.20). The probe is
*stratified* by the stratum key of §6.2.2 with allocation ∝ √(stratum mass) and at least
min(32, stratum size) rows per stratum (reservoirs per stratum with H_m keys), and every
row keeps its HT mass: a mass-weighted reservoir would leave three daily GA logins out
of a probe dominated by portal page views, and P08 could then never propose their binding.

Hourly work is bounded, not a sweep of the registry: each run evaluates every attribute
that currently holds a role other than `dropped`, plus a rotating slice of A_probe = 64
others (crc32 phase), so every registered attribute is evaluated at least once a day and
the run costs O((n_kept + 64) · levels · m_t · R_p) numpy work.

For attribute a at level ℓ (the finest level with ≤ 64 distinct values; numeric at
bins, text at shape), on the probe with HT weights:

```
H(a)          plug-in entropy with Miller–Madow correction
C0            context candidates: top-5 split-role attributes by I(·; targets) plus the
              seeds http.route@ℓ1, net.src@(grp | /24), ctx.tod_min@slot, ev.ch (seeds
              absent from the system are skipped; they only bootstrap the first hour)
CR(a)         = max_{c ∈ C0} (1 − H(a | c) / H(a))                 predictability, 0..1
U_t(a)        = cov(a) · S_a · (H(a) − min_c H(a | c)) − λ_c · cost_us(a)      bits/event
U_s(a, ℓ)     = Σ_{b ∈ targets} Î(b ; gen(a, ℓ)) − (card(a, ℓ) − 1) · K̄ · log2(n) / (2n)
g3(a → b)     = 1 − Σ_x max_y c(x, y) / n                           (Kivinen & Mannila 1995)
```

`K̄` = mean target alphabet size, λ_c = 0.001 bits per µs (default; one bit saved
per event is worth 1 ms of CPU per event). Roles, with hysteresis (promote at
U ≥ u_hi = 0.05 bits/event, demote below u_lo = 0.02 for 3 consecutive hourly runs):

| Role | Condition | Effect |
|---|---|---|
| `invariant` | H(a) ≤ 0.05 bits and cov ≥ 0.99 (system level); per node, detected by P04 when the top value has ≥ 99.5 % of mass with n_c ≥ 30 (confidence channel; at H_m a small node would never reach it) | Stored once as `(level, value)`, checked for violations, not modelled, not split on. "Always contains username=" and the server address are invariants. |
| `split` | max_ℓ U_s(a, ℓ) ≥ u_hi and card(a, ℓ) ≤ 64 | Candidate for specialisation at the best 1–2 levels. |
| `target` | U_t(a) ≥ u_hi, cov ≥ 0.05, S_a ≥ 0.7 | Modelled in node summaries (top m_t per node). |
| `shape` | distinct/n ≥ 0.5 at ℓ0 and CR at ℓ0 < 0.05 | Generalised to the shape level (identifiers, nonces, free text); may still be a target at that level. |
| `redundant` | g3(a → b) ≤ 0.01 with H(b) ≥ 0.1 for a kept a | b is not a target (its information is in a; counting both would double the split evidence); the FD is kept as a fact for P08/P14. The member of the pair with lower cost and higher coverage is kept. |
| `dropped` | U_t < u_lo and max_ℓ U_s < u_lo | Registry-only (presence, HLL); re-probed for one evaluation every 7 days (`probe`). |

Who and when are always summarised (§5.5.3–5.5.4; they are not counted in m_t),
but they are *split candidates* only while U_s ≥ u_lo at some level. When IP carries
no information at any level (CR(net.src) < 0.02 at /32, /24, grp and reg), `net.src`
is removed from split candidates and targets for that system — the "直接 IP 不作为特征"
case — and the who summary is rendered as population statistics only.

Per node: targets are the system targets re-ranked by node-local
`H_node(a) · cov_node(a)` from the node summaries (an attribute that is constant at
the node becomes a node invariant and frees its slot); the top m_t = 8 are written
as `node_overrides[nid]`. P04 applies overrides at the next update of the node
(new target summaries start empty and back off to the parent's).

Output `model.attrsel@(s,__system__) = {version, roles{a: role}, levels{a: [ℓ…]},
targets_sys{kind: [a…]}, split_cands{kind: [(a, ℓ)…]}, redundant{b: a},
node_overrides{nid: [a…]}, who_mode}`; every change emits `attribute_role` INFO.

### 6.5 The pattern tree: counting, evidence, specialisation (P04)

#### 6.5.1 What P04 does per learned event

```
def learn(tree, e, m_e, trust):                     # m_e = w/π (mass, §5.2.3); trust = trust(ip, t−1) × damp(e)
                                                    # (§6.9.3); evidence ω_e ∈ (0, 1] is derived inside (§6.5.4)
    path = route(tree, e)                           # same function P03 used
    leaf = path[-1]
    ω = trust · evidence_unit(e, leaf)              # §6.5.4
    for depth, N in enumerate(path):
        N.update_core(e, mass = m_e·trust, evidence = ω)       # mass, n_eff, who, when: every node, O(levels)
        ρ = 1 if depth >= len(path) - 2 else min(1, r_target / N.rate_Hs)   # ancestor sampling
        if seeded_uniform(s, N.id, e.t1, e.row) < ρ:
            N.update_targets(e, mass = m_e·trust/ρ, evidence = ω)   # targets, invariant checks
    if leaf.split_stats: leaf.split_stats.update(e, ω)                # §6.5.3 (evidence-weighted)
    if leaf.state >= confirmed:
        leaf.update_exception_stats(e, m_e·trust, ω)                  # §6.7
        leaf.update_pairs(e, m_e·trust, ω)                            # §6.12
        leaf.adwin.add(clip(loss(leaf, e), 0, 20)); leaf.ph_update(e)  # §6.9
    if leaf.split_stats and leaf.since_check >= n_g: try_split(leaf)
```

`r_target` = 1 event/s of mass at H_s (`rate_Hs` = H_s-decayed mass per second): a general
node busier than that updates its *target* summaries on a thinned stream whose mass is
re-weighted by 1/ρ (distributions stay unbiased); its evidence count, who and when
summaries are updated on every learned event (they are cheap, and who-closedness must not
miss a new source), and its targets' confidence uses the evidence actually observed. The
leaf and its parent always update everything because they decide splits and exceptions.

#### 6.5.2 Split candidates

A leaf in *learning mode* (at most L_max = 64 per system, chosen by P15 as the leaves
with the largest `mass_Hm · (1 − purity)`, purity = mean over targets of the top-value
share) tracks up to C = 6 candidates `(a, ℓ)`:

1. for each `split`-role attribute a (P05), the coarsest level ℓ at which a is not
   constant at this leaf, and the next finer level (two candidates for who/when/route,
   one for others), ranked by P05's U_s and truncated to C;
2. attributes already constrained in the leaf's context contribute only levels finer
   than the constrained one;
3. content attributes (body key set, size bin, status class) are split candidates like
   any other: splitting a route on them divides one route into *action variants*
   (predictive clustering, Blockeel, De Raedt & Ramon 1998), which is how "聚类出动作的
   特征约束条件" is realised.

A candidate that is replaced (re-ranking by P05) starts from empty statistics. The leaf
keeps `C_ever`, the number of distinct candidates it has ever tracked; it is the
multiplicity charge of rule (V). If no candidate is accepted within R_learn = 14 d or
2 000 evidence units, or the leaf's ADWIN fires, the split statistics restart from empty
(stale accumulations under a drifting leaf would otherwise mix regimes); `C_ever` keeps
counting, so repeated attempts pay for themselves.

#### 6.5.3 Evidence: prequential code lengths with hierarchical smoothing

Every node predicts each target value with an evidence-scaled hierarchical Dirichlet
(Teh et al. 2006, single-level form; PPC-9) that backs off to its parent:

```
p_N(v) = (s_N(v) · n_N + α · p_parent(N)(v)) / (n_N + α),   α = 2
         s_N(v) = mass share of v at N (H_m), n_N = evidence at N (H_m channel for shape,
         confidence channel where a confidence is stated); p_root backs off to the registry's
         system distribution of the attribute at that level, whose own unseen mass is the
         Good–Turing estimate.
```

For a candidate c = (a, ℓ), the leaf keeps, per value slot j (a SpaceSaving(k_v = 8)
over gen(a, ℓ, ·) plus `other`; an evicted slot folds its counts into `other`), per
target t ∈ targets(leaf) ∪ {who@ℓ_w, when@slot} with t not derived from the same source
field as a, **evidence-weighted** counts n_{c,j,t,b} over k_b = 9 bins (the leaf's top-8
values of t at learning start plus other; ℓ_w = the system's who level chosen by P12,
numeric targets at their ℓ1 bin). For each learned event e with evidence ω_e:

```
ℓ_leaf,t(e) = −log2 p_leaf,t(bin_t(e))
ℓ_c,t(e)    = −log2 [ (n_{c,j,t,b} + α · p_leaf,t(b)) / (n_{c,j,t,·} + α) ],  j = slot_c(e), b = bin_t(e)
L1_{c,t}   += ω_e · ℓ_c,t(e)                          # per-target prequential code length under the split
d_c(e)      = ω_e · Σ_t (ℓ_leaf,t(e) − ℓ_c,t(e))      # bits this event saves under the split (all targets)
G_c        += d_c(e);  S1_c, S2_{c,c'}, R_{c,c'} as running sums for rule (S)
then n_{c,j,t,b} += ω_e for every t
```

Both predictors are evaluated *before* they see e (prequential MDL, Dawid 1984; Grünwald
2007), so G_c already charges the split's parameters: a child with little data predicts
like the leaf and saves nothing. G_c is the *selection and gain* statistic (MDL). It is
not used as a significance test on its own, for two reasons found in review: (i) the
leaf's own predictive is a mixture over unknown parameters, so the ratio of two mixtures
is not a test martingale under a composite null; (ii) summing savings over *dependent*
targets (a body size and a key set that move together) counts the same information twice.
Rule (V) of §6.5.5 is the significance test; it is valid under both.

The split's own description is `L_split(c) = log2 C + g · log2(card_hint(a, ℓ))` bits for
g value groups (an Occam charge for the post-hoc value grouping; the test in (V) does not
depend on the grouping). The split attribute's own hierarchy is excluded from the targets
(splitting on IP trivially predicts the IP).

#### 6.5.4 Evidence units

Consecutive events of one source on one leaf are not independent evidence of a
population pattern. P04 keeps an LRU keyed (ip, sess.key, leaf) (cap allocated by P15,
default min(65 536, 4 × active sources in 7 d)) with a run counter r of *learned rows*:
if the previous learned row of the key was more than τ_burst = 300 s ago, r ← 0. A
learned row contributes

```
ω_e = trust(ip) · damp(e) / (r + 1),   then r ← r + 1        (ω_e ≤ 1 always)
```

so a burst of k observed rows contributes H(k) ≈ ln k + 0.58 units. The row's mass m_e
(aggregation share × HT weight × sensor sampling rate) does **not** enter ω: a row that
stands for 40 unobserved events carries the information of one observation, and an HT
weight of 1 000 in a thinned stratum is a statement about how many events the row
represents, not about how much the learner has seen. (The first draft used
ω = H(r + m_e) − H(r), which let one aggregated row count as ln m units and a heavily
thinned stratum count as ≈ 7 units per row; with weights above 1 the e-value argument of
§6.5.5 also fails, because E[(q/θ)^ω] ≤ 1 needs ω ≤ 1.) Aggregated mode sees fewer rows
than event mode and correspondingly has less evidence; pack O60 checks that the two
agree on what they learn within the stated confidence (PG1).

`n_eff` sums ω_e at H_s, H_m and H_l (§5.5.2). Mass drives distributions (shares, bands,
windows, rendering); evidence drives every threshold, test, bound and confidence.

#### 6.5.5 Split rule

Every n_g = 32 evidence units at a learning leaf, with c1, c2 the best two candidates by
G and k the index of this check at the leaf:

```
(V) validity:  log2 e_c1 ≥ τ0 + log2 C_ever,        τ0 = 10
               e_c = (1/T_c) Σ_t 2^( L0_{c,t} − L1_{c,t} ),   T_c = number of targets of c
               L0_{c,t} = Σ_b n_{c,·,t,b} · log2( n_{c,·,t,·} / n_{c,·,t,b} )   (pooled counts, ML code length)
(G) gain:      G_c1 − L_split(c1) ≥ 0
(S) stability: μ_c1 − μ_c2 ≥ ε_k  or  ε_k ≤ τ_tie,                            τ_tie = 0.05 bits/unit
               ε_k = sqrt(2 V_Δ ln(3/δ_k) / n) + 3 R_Δ ln(3/δ_k) / n,  δ_k = δ / (k(k+1)),  δ = 1e-4
               (empirical Bernstein, Maurer & Pontil 2009, made time-uniform by the δ_k union bound;
                V_Δ, R_Δ of d_c1 − d_c2, n = evidence units since the candidates started)
(D) diversity: the evidence spans ≥ 2 distinct local dates, or ≥ 200 evidence units since the candidates started
(M) mass:      ≥ 2 value groups with ≥ n_child_min = 5 evidence units each in the split statistics
```

**Why (V) is anytime-valid.** Fix a target t and suppose the null holds: at this leaf,
t's distribution θ0 does not depend on the value of c, and evidence units are
conditionally independent given the past. The pooled maximum-likelihood code length is
never longer than the code length under θ0, so for every n,
`2^(L0 − L1) ≤ Π_e (q_e(x_e)/θ0(x_e))^(ω_e)`, where q_e is the split predictive (which is
predictable: it uses only the past). Each factor has conditional expectation
`E[(q/θ0)^ω] ≤ (E[q/θ0])^ω = 1` because ω ≤ 1 (Jensen), so the right side is a
non-negative supermartingale that starts at 1, and by Ville's inequality
`P(sup_n e_{c,t} ≥ 2^τ) ≤ 2^−τ` ("universal inference", Wasserman, Ramdas & Balakrishnan
2020, in its sequential form). An average of such e-processes over targets is again one,
**whatever the dependence between targets**. Candidate number i at the leaf (in order of
first tracking) is always tested at a threshold ≥ τ0 + log2 i, because C_ever only grows;
the union bound therefore gives: the probability that this leaf ever splits on a candidate
that is independent of all its targets is ≤ 2^−τ0 · H(C_ever) (harmonic number;
≈ 0.5 % for 150 candidates over a year of restarts), however often it is checked. The check cadence n_g therefore costs nothing statistically; it only bounds CPU.

What (V) does not cover, stated so that nobody relies on it: an attribute that tracks the
identity of a few heavy sources (a client stack used by one department) *is* dependent
with the targets; splitting on it is a correct statement about the population, and P05's
redundancy map and EFDT revision (§6.6) later replace it by the more natural attribute
(`grp`) when that predicts better. Within-source dependence beyond τ_burst (one IP's
logins on successive days are the same person) is not independence either; the
population claims (who-closedness) therefore count sources as well as events (§6.16.2).

**Power, and why not Hoeffding alone.** The Hoeffding bound on a gain with range
R ≈ m_t · log2 k_b ≈ 25 bits needs n ≈ R² ln(1/δ)/(2ε²) ≈ 10⁴ events to resolve
0.5 bit/event — a department of 3 IPs logging in once a day would never split. Rule (V)
needs only that *one* target's code length shrinks by ≈ τ0 + log2 C_ever + log2 T_c
≈ 16 bits in total: a target that saves 2–3 bits per event once each child has a few
events (a department's usernames or login minutes against a mixed population) crosses it
after a few dozen evidence units, i.e. within one to two weeks for a daily 3-IP pattern
(estimate; P04 unit test (a) measures it). (S) still guards the *choice* between
candidates, and τ_tie breaks ties VFDT-style (Domingos & Hulten 2000) when two candidates
are equally good.

**System-level false splits.** A per-leaf bound does not make the *system* free of false
splits: a system can expect about (leaves that enter learning mode per year) ·
2^−10 · H(C_ever), i.e. a few per system per year at the defaults (estimate: ≈ 500 leaves ×
1e-3 × 5). A false split is not an
alarm (its children back off to the parent and predict like it) and is pruned by the
decayed-saving rule of §6.6 within days; PG2 reports the false-split count on the
red-team pack.

**Value grouping.** Before committing, the k_v + 1 slots are merged greedily: the
pair (j, j') with the smallest static code-length increase
`ΔL = L_KT(j ∪ j') − L_KT(j) − L_KT(j')` (L_KT = KT code length of the pooled
evidence-weighted target counts) is merged while `ΔL ≤ log2 card_hint(a, ℓ)`; slots with
< n_child_min evidence units join `other`. The result is value *groups*: e.g. {192.168.1.21,
192.168.1.23, 10.168.7.121} as one child and every other IP as `other`, or {GA} vs {FIN}
vs other at the grp level. New children start as `candidate`; their categorical target
summaries are seeded from the split statistics (evidence counts, with mass = evidence ×
the leaf's mass/evidence ratio) and everything else starts empty (backing off to the
parent until they have data).

#### 6.5.6 Complexity

Per learned event: routing O(D); who/when/evidence updates O(D · 7); target updates
O((2 + ρ̄(D − 2)) · m_t); split
statistics O(C · (m_t + 2)) (vectorised gather over a float32 `[C, k_v+1, m_t+2, k_b]`
array plus the `[C, m_t+2]` accumulators L1); exception and pair statistics O(1) hash
lookups each; drift O(log W). With the defaults (D ≤ 8, m_t = 8, C = 6) that is at most
~110 counter updates and ~60 logarithms per learned event: amortised O(depth × touched
nodes), independent of the number of IPs and of the number of registered attributes.
Split checks cost O(C · (m_t + 2) · k_b + C²) per n_g units. Estimate (Python, packed
arrays): 40–120 µs per learned event; to be measured (PG4).

### 6.6 Generalisation: prune, merge, revise, budget

- **Prune.** An internal node N keeps, on the events it routes, the decayed (H_m)
  prequential saving of its split `G_N^m = Σ ω_e (ℓ_N(e) − ℓ_child(e))`. If G_N^m < 0 at
  3 consecutive daily checks, N collapses: children summaries are merged into N (all
  summaries are mergeable), N becomes a leaf, children are retired with lineage
  `merged_into`. A split on an attribute that P02 declared `gone` (§6.3) collapses at
  the next daily check without waiting.
- **Sibling merge.** Two children with mass-weighted JSD over all targets < τ_merge =
  0.02 bits (H_m) for 3 consecutive daily checks are merged (their value groups
  unite). This is how two departments that converge stop being separate patterns.
- **Revision (EFDT, Manapragada, Webb & Salehi, KDD 2018).** Up to R_max = 16 internal
  nodes per system (largest mass, n_eff ≥ 200) keep a split-statistics block for
  alternative candidates. Daily, if the best alternative's gain exceeds the current
  split's gain on the same events by a margin of τ0 bits (both prequential, §6.5.3) and
  rule (S) prefers it, the node is re-split: the new children are created as candidates
  while the old subtree keeps scoring; when the new children are confirmed (≤ 7 d) the
  old subtree is retired (`replaced_by`). Revision chooses between two splits that both
  passed (V), so it is a model-selection margin, not a new significance test. This is how
  a split made early on /24 (before P11 groups existed) is replaced by a split on `grp`.
- **Budget.** When the node count exceeds N_max(s) (P15), the leaf with the lowest
  utility `u = (its share of the parent's G^m per day) × (2 if confirmed else 1)` is
  pruned first, stale leaves before anything else; a parent all of whose children are
  below the median utility collapses. Learning mode is granted to at most L_max leaves.

### 6.7 Exceptions: when one IP earns its own pattern

At a confirmed node N, the heavy sources of the who summary (share ≥ φ_x = 0.02, at most
k_x = 16; `shared:<ip>` items excluded, §5.4.1) get `ExcStats`. An exception is a split
of N on the indicator `net.src = x`, tested with rule (V) of §6.5.5:

```
per target t (numeric targets at their ℓ1 bin), since x's ExcStats started:
  counts: c_x,t[b] (x's evidence counts), c_N,t[b] (N's evidence counts, pooled)
  L1_x,t = Σ_{e from x} ω_e · (−log2 p_x,t(b_e))  +  (L_N,t − L_N,t^start − Σ_{e from x} ω_e · (−log2 p_N,t(b_e)))
           p_x,t(b) = (c_x,t[b] + α p_N,t(b)) / (c_x,t[·] + α)       (x under its own predictive, backing off to N;
                                                                     everyone else under N's own prequential code L_N,t)
  L0_x,t = Σ_b c_N,t[b] · log2(c_N,t[·] / c_N,t[b])                  (pooled ML code length)
  e_x    = (1/T) Σ_t 2^(L0_x,t − L1_x,t)
exception ⇔ log2 e_x ≥ τ0 + log2(i_x)  (i_x = ordinal of x among the IPs ever tested at N),
            n_c,x ≥ 10 (x's evidence on the confidence channel; a once-a-day IP has only ≈ 7 at H_m)
            and x's evidence spans ≥ 2 days
```

Coding the other sources under N's predictive (which also learned from x) can only
lengthen L1, so the test is conservative and needs one accumulator per target at N, not
one per (x, event). The exception child `N@ip=x` holds only the targets with per-target
saving ≥ 2 bits; all other attributes are inherited from N. It is removed (the IP
returns to its group) when its H_m-decayed saving is < 0 at 3 daily checks. Exceptions
count toward N_max; at most 10 % of N_max may be exceptions (lowest e_x evicted first).

Bindings (§6.12) are not exceptions: "each of the three IPs submits its own username"
is one FD at the group node, not three nodes. An exception is for an IP whose
*distribution* differs (e.g. one member whose login bodies are 2–3 KB where the rest
are 1–2 KB). An IP with at least one exception or one binding is *distinctive*; P15
uses this in the earned sets of §10.2. In a large open population (a portal) no single
IP reaches φ_x at a route node, so single-IP exceptions do not arise there; this is the
intended "IP 不作为特征" outcome, and per-IP abuse is caught by intensity (§6.16.4).

### 6.8 Lifecycle, versions, lineage, reference snapshots

#### 6.8.1 States

| From → to | Condition |
|---|---|
| (new) → candidate | created by a split, revision, alternate or exception |
| candidate → confirmed | n_c ≥ n_conf = 20 (confidence channel), ≥ 3 distinct local dates (kind `win`: 2), and every applicable fitter (P06–P09) has published a constraint or declared none applicable |
| confirmed → stable | ≥ 7 d since confirmed, ≥ 5 distinct dates, no drift alarm in 7 d |
| confirmed/stable → evolving | an ADWIN or Page–Hinkley alarm (§6.9); back to stable after acceptance or replacement |
| any → stale | expected arrivals `E = rate_Hl × (active time of the node's day types and windows since last_seen, counting normal days only, §6.2.3) ≥ 3` with none observed (P(0 \| Poisson(E)) = e^−E < 0.05); emits `pattern_absent` |
| stale → confirmed | the pattern reappears |
| stale → retired | stale for 30 d, or pruned/merged/replaced |
| retired → dormant | the node's date bitmap shows ≥ 3 occurrences with a regular spacing (monthly, quarterly: the gaps' coefficient of variation ≤ 0.2) |
| dormant → confirmed | an event routes to the dormant node's context again within 400 d (`pattern_revived` INFO; scoring uses the dormant constraints at once instead of treating the event as novel) |

`pattern_absent` is INFO, or LOW when the node is stable, daily-periodic, and its whole
who-set is active elsewhere in the system on that day (the report that was not
submitted while the people were at work).

Dormant records keep only the context, the reference snapshot and the date bitmap
(≈ 1 KB); at most 10 % of N_max, oldest first out. They are how a month-end close or a
quarterly audit, which recurs less often than the H_l memory, is remembered without
keeping its full node alive.

#### 6.8.2 Versions and lineage

`version` increments on a structural change (split set or replaced, exception added or
removed, groups merged); `cver` on a material content change (a band endpoint moves
> 25 %, a window endpoint > 10 min, a grammar changes, a binding value changes, the
rendered who-list changes). Pattern id `p:<tree key>:<kind>:<nid>@<version>.<cver>`. The
lineage ring (4096 entries per tree) records `(ts, op, nid, parents, children, detail)`
for op ∈ {create, split, merge, prune, replace, exc_add, exc_del, content, retire,
dormant, revive}; retired node summaries (context, last statements, dates) stay 90 d for
diffs, at most N_max / 2 of them per tree (FIFO).

#### 6.8.3 Reference snapshot

Daily at 04:00 local, each confirmed or stable node that had no drift alarm, no held
events and no open incident on any of its top who IPs during the previous day copies
its fitted constraints and compact target distributions into `N.ref`. P03 scores
against both: `p = min(1, 2 · min(p_cur, p_ref))`, the dual-anchor rule of B04
(`lib/bayes`), so a slow creep of the current statistics cannot hide a deviation from
the reference. Before a node has a snapshot, or when p_ref is NaN for a type, `p = p_cur`.

### 6.9 Drift and multi-timescale evolution

#### 6.9.1 Detectors

- ADWIN (δ = 0.002) on each confirmed node's per-event clipped log-loss ℓ_N(e) ∈ [0, 20]:
  a significant increase is structural drift. Over many nodes and a long run ADWIN will
  raise false alarms; a false alarm costs one alternate (capped at 4 per system) and is
  resolved by the alternate losing (§6.9.2), never by an alert.
- Page–Hinkley per numeric target (λ = 5σ, drift allowance 0.5σ, σ from H_l moments),
  per when (signed circular difference of the arrival minute from the H_l circular
  mean), and a Bernoulli CUSUM on "IP new to this node" (who churn).

#### 6.9.2 Responses

- **Structural** (ADWIN): an alternate subtree (Hoeffding Adaptive Tree, Bifet &
  Gavaldà 2009) rooted at a copy of N's context learns only new events; after ≥ 3 days
  and ≥ 100 evidence units it replaces N's subtree if its prequential code length on the
  same events is shorter by τ0 bits and rule (S) agrees; otherwise it is dropped after
  14 d. At most 4 alternates per system.
- **Content** (Page–Hinkley): N → evolving; P06/P09 fit provisional constraints from the
  H_s statistics. The change is *accepted* when it persists for T_persist, counted in
  normal days (§6.2.3) (numeric bands 1 d, time windows 3 workdays, who-set 2 d, binding
  5 events over ≥ 2 d) and either ≥ max(2, ⌈0.5 · |who_top|⌉) distinct IPs of N show it (a
  coordinated change, e.g. a department's new schedule) or, for a single-IP node, it
  persists for 5 d; and none of those IPs had an open incident or quarantine in the
  period. On acceptance the constraints switch (cver + 1), the confidence channel of the
  changed attribute is reset (§6.9.4), and `pattern_drift` INFO is emitted. Until
  acceptance, a coordinated change's violations of the drifting attribute are capped at
  LOW (the class-level reading rule of architecture.md §5); a single IP's are not capped.
- **Who-set changes are special.** Persistence alone never makes a foreign source a member
  of a who-closed node: that is exactly what a patient intruder would rely on (anomaly A9).
  A new source is absorbed into a who-closed node's who set only when (i) it belongs to a
  group that already holds mass at the node (a department colleague), (ii) it is a
  `readdress_candidate` of a member (§6.12), (iii) an operator labels it (B23 `fp` with scope
  pattern, or `expected_change`), or (iv) the node has σ < 2 and ≥ 2 new sources of one
  group show up and persist for T_persist (a team being given access). Until then its events
  are learned with outlier damping (§6.9.3), so its mass share stays far below the heavy set's
  95 % boundary, and every day it appears it is scored as a who violation again.

Because every statistic exists at three half-lives, "动态变化" has an explicit
timescale: a change is visible in H_s within hours, becomes the pattern after
acceptance (days), and from then on accumulates confidence from zero-ish on the reset
confidence channel; a slow change that no detector flags is followed by the H_m shape
within one to two weeks.

#### 6.9.3 Trust gating, outlier damping and delayed learning

P04 learns the batches of tick t' ≤ t − D (D = max(4 ticks, 600 s), the lib-3 rule;
batches are retained D + 1 ticks in compacted form, §5.2.3). An event's learning mass is
`w/π · trust(ip, t−1) · damp(e)` and its evidence ω_e (§6.5.4) carries the same
`trust · damp` factor, with `behavior.trust` from B28. In bounded mode (§10) an IP
without a B28 row has trust 1 unless it has an open incident or is quarantined. The same
rule applies to every learner (P02 statistics, P10 counts, P11 signatures, §4.2).

**Outlier damping.** An attacker who is not yet in an incident still has trust 1. So an
event that P03 scored at p_ev ≤ 1e-4 against a confirmed or stable node is learned with
`damp = 0.1` unless that node is `evolving` on the violated attribute (then 1). A single
source therefore needs ten times as long to move a confirmed constraint; a genuine
coordinated change still goes through the acceptance path of §6.9.2 at full weight.
Scanners and random-path floods, whose events land as novel at p ≤ 1e-4, are damped the
same way and cannot fill the tree faster than the node budget prunes them.

Events of a quarantined IP go to its held reservoir (`ptree.held[ip]`, ≤ 256 events per
IP and ≤ H_max events per system (default 65 536, allocated by P15), FIFO; only the
attributes that are targets or split candidates are kept): `model.control` release
learns them with trust_prov, reject or 7-day expiry discards them. There is no lattice
rollback (§14.1); resistance to slow poisoning rests on delayed learning, trust gating,
outlier damping, multi-day confirmation and the reference snapshot, and is measured by
anomaly A9 (§11.4).

#### 6.9.4 The confidence channel: "longer is more precise" without freezing

A fixed exponential decay makes evidence saturate: a node with r evidence units per day
never holds more than r · H / ln 2 units at half-life H. At H_m = 7 d, the finance
approval node of pack O (≈ 3–4 units per workday after burst damping) would plateau
near 30 units and its unseen-IP mass near 0.5/31 ≈ 0.016, so it could never be
"closed at U ≤ 0.01" however long the system ran — the opposite of 用的时间越长越精准.
The design therefore separates:

- **shape** (distributions, bands, windows, value sets): estimated on H_m (and H_s for
  provisional constraints), so it follows behaviour within one to two weeks;
- **confidence** (closedness U, binding lower bounds, coverage bounds, the n used in
  Dirichlet concentrations for p-values, statement confidence): computed on the
  **confidence channel**, the H_l = 30 d evidence entry, which (i) is reset to the H_m
  state of the attribute (or of the whole node, for structural changes) when a change is
  accepted (§6.9.2) or a node is replaced, and (ii) while the attribute is `evolving` is
  capped at its H_m value. `seg_start` records the last reset.

While behaviour is stationary, confidence therefore grows for about two months before
the H_l plateau (rate × 43 d) is reached; after an accepted change it starts again from
the new behaviour's H_m evidence instead of mixing regimes. The plateau is a stated
design constant (§7.1): raising H_l (config) buys more confidence for patterns that never
change, at the cost of slower forgetting of changes that no detector flags.

### 6.10 Numeric constraints (P06)

From each node's numeric target summaries (t-digest H_m, moments, daily ring of
`(min, max, n_obs)` over 30 days where n_obs = evidence units observed that day,
exceedance reservoirs); transformed scale y = log v when the registry says `log`.

```
band90  = [Q(0.05), Q(0.95)]                       # "90 % in …"  (shape: H_m)
band98  = [Q(0.01), Q(0.99)]
range   = [min over the ring's daily mins, max over its daily maxes]   (days of the current confidence segment only)
n_rng   = Σ n_obs over the same days                 # the observations the range was taken from
cover   = P(next ∉ range) ≤ 2 / (n_rng + 1)          # exchangeability: rank of the next observed row
tails   = GPD(ξ, σ) fitted by lib/evt.gpd_pwm_fit on exceedances over Q(0.90) (and of −y
          under Q(0.10)) when ≥ 30 exceedances, else none
```

The first draft paired a 30-day observed range with the H_m-decayed n_eff, which is a
different (smaller, differently weighted) sample; the rank bound is only valid for the
number of observations the extremes were actually taken over, hence n_rng. Using evidence
units rather than rows makes the bound conservative under within-burst dependence. The
bound is about *observed* rows: with HT thinning or sensor sampling the unobserved events'
extremes are covered by the tail fit, and the statement says "观测到的请求" in that case.

Display rounding: the band endpoints are rounded outward to the coarsest 1-2-5 × 10^k
grid (in the attribute's unit, bytes shown in KB/MB) whose rounded band still has
empirical coverage in [0.88, 0.95]; otherwise the next finer grid. "100 % in [a, b]" is
rendered only with n_rng ≥ 30 (bound ≤ 6.5 %) and approx share ≤ 0.2, as the rounded
observed range with its coverage bound ("n = 183, 下一次落在区间外的概率 ≤ 1.1 %"), the bound
tightening as n_rng grows; otherwise "observed range". Scoring p for a value y (P03): inside band98 → two-sided mid-p from the
t-digest CDF; beyond → `0.1 · GPD_sf(y − u)` doubled, or the conformal rank
`(1 + #{y_i at least as extreme}) / (n + 1)` without a tail fit, n on the confidence
channel. Output per node and attribute: `{band90, band98, range, n_rng, cover, tail_lo,
tail_hi, n_eff, n_c, approx, cver}`.

### 6.11 Payload grammar induction (P07)

For set attributes (`body.keys`, `q.keys`, JSON key sets):

```
required  R = {k : presence(k) ≥ 0.99}                optional O = {k : 0.01 ≤ presence < 0.99}
p(new key) = (N1_keys + 0.5) / (N + 1)                p(missing required k) = (misses_k + 0.5) / (n + 1)
```

For text attributes (`body.kv.*`, `q.kv.*`, `hdr.*`, `body.tpl`), from the shape
SpaceSaving, the length histogram and the charset counts:

1. Group shapes by *skeleton* (class/literal sequence without run lengths).
2. If ≤ 3 skeletons cover ≥ 99 % of mass, the grammar is their alternation with
   per-run length ranges (anti-unification). Skeletons that share a prefix are factored
   into the prefix plus an optional suffix: `L4`, `L3`, `L5 . L1` → skeletons `L` and
   `L . L` → `[a-z]{3,5}(\.[a-z])?`.
3. Otherwise generalise to the charset union and total length range (`[a-z.]{1,10}`).
4. Closed value set: when the exact-value SpaceSaving covers ≥ 99 % of mass, N1/N ≤ 0.01
   and n_c ≥ 50 (policy clear or hmac), the value set V is closed; p(new value) =
   (N1 + 0.5)/(N + 1).

Length ranges are the observed ranges of the current confidence segment, with the
coverage bound of §6.10 on the length; nothing is pre-set. The requirement's "后边内容不超过
10 个字符" is therefore an *output* when observed lengths reach 10, and otherwise the
learned bound is tighter (three users jack, rose, mike give `[a-z]{4}`; the OA-wide login
node of pack O, with sales usernames of 3–8 letters, gives `[a-z]{3,8}`). An operator who
knows a business rule ("≤ 10") can pin it through B23 feedback (`expected` policy on the
constraint), which widens the learned range and never narrows it.

Grammar coverage c_g (mass matched) and unseen-shape mass U_s = (N1_shapes + 0.5)/(N + 1)
(N on the confidence channel) are published with it. A value that fails the grammar
scores p = U_s; a value whose length is outside the length range scores the numeric p
of §6.10 on the length; a value containing a character class absent from the grammar's
charset (quote, `=`, space, `<`, `;`, `--` in a `[a-z]` grammar) is flagged
`injection_shape` in the violation. Rendered: "表单必含 username=，取值 `[a-z]{3,8}`
（覆盖 100 %，n = 183）". Bodies that no parser structures (binary, protobuf, encrypted
application payloads) are profiled by length, charset classes and the Drain template of
their printable prefix only.

### 6.12 Bindings: approximate functional dependencies (P08)

Candidates (hourly, from P05's stratified probe sample): pairs (X, Y) with X ∈ who levels
at the system's who-granularity (ip, /24, grp) plus `client.stack` and `sess.key` when
present, and Y ∈ categorical/text targets (never `shape`-policy attributes such as
passwords), such that H(Y) ≥ 0.5 bit, ≥ 2 values of X have ≥ 3 events and
g3(X → Y) ≤ 0.2 — or, for *set bindings*, the per-x value sets are small (median
distinct y per heavy x ≤ 4) while H(Y) ≥ 1 bit. The reverse direction (Y → X) is screened
the same way. At most Q_pairs = 16 pairs per system, written to `model.pwant.pairs` with
the nodes where Y is a target (or the root). P04 then keeps a `PairSketch` at those nodes:
SpaceSaving(k_bx = 64) over x, each with a SpaceSaving(k = 4) over y, mass at H_l and
evidence on the confidence channel.

Fit per node (n_x, k_x in evidence units on the confidence channel):

```
for heavy x:  y*_x = argmax_y c(x, y);  k_x = c(x, y*_x)
prior_x       leave-one-out empirical Bayes over the other heavy x' of the node:
              mean m = (Σ_x' k_x' + ½) / (Σ_x' n_x' + 1), strength s = min(20, s_mom, Σ_x' n_x'),
              s_mom = the method-of-moments strength of the purities k_x'/n_x' (∞ when they are all equal);
              Beta(a0, b0) = Beta(m s, (1 − m) s); Jeffreys Beta(½, ½) with < 2 other heavy x'
LB_x          = 5 % quantile of Beta(a0 + k_x, b0 + n_x − k_x)
binding(x)    ⇔ n_x ≥ 5 and LB_x ≥ 0.8
FD holds      ⇔ bindings cover ≥ 80 % of the mass of heavy x and g3 = 1 − Σ_x k_x / N ≤ 0.05
one-to-one    ⇔ the reverse FD Y → X also holds (each username used from one IP)
set binding   ⇔ no binding(x), but x's value set is closed: U_x = (N1_x + E_x + 0.5)/(n_x + 1) ≤ 0.05
              with ≤ 4 values covering ≥ 95 % (a shared terminal: "192.168.5.7 → {jack, rose}";
              reversed, a service account: "svc_backup ← {10.1.1.5, 10.1.1.6}")
```

Leaving x out of its own prior avoids using x's data twice. Borrowing strength across the
IPs of a node is what lets a small department's bindings be confirmed in days: with two
other members at 5 pure logins each, an IP with 5 pure logins has LB_x ≈ 0.88 (≥ 0.8, a
binding), and LB_x ≥ 0.9 from its 6th; a per-IP Jeffreys bound needs ≈ 9 pure logins to
reach 0.8 (computed with scipy for this review; P08 unit tests pin the numbers).

`shared:<ip>` items (§5.4.1) never get per-IP bindings; where `sess.key` is present, the
binding is learned on (sess.key → Y) instead, which is how usernames behind one NAT
address stay separable.

Violation scoring for event (x, y): if the FD and binding(x) hold and y ≠ y*_x:
`p_bind = (b0 + n_x − k_x) / (a0 + b0 + n_x)` (posterior predictive of a non-bound value;
≈ 0.03 at 5 logins, falling with every consistent login), flagged `cross_binding` when y
is the bound value of another IP (a credential axis signal: 192.168.1.21 submitting
`username=rose`). For a set binding, a value outside the set scores U_x. For the reverse
direction (y's own source set is closed and x is not in it) the flag is `foreign_source`,
refined by a **concurrency test**: when a bound source of y was active at the same node
within ±T_conc = 1 h, the event is `concurrent_use` (the credential is being used from
two places at once: credential axis); when the source is `unknown_ip` (§6.16.2) and no bound
source of y has been active in the system since the start of the local day, it is
`readdress_candidate` (a DHCP / VPN / re-cabled machine that kept its user overnight), which
scores p = U_x but whose discrete severity is capped at LOW and which P11 uses as a hint to
move the new IP into the old IP's group (§6.15); in between (a bound source active earlier
the same day, not within T_conc) only `foreign_source` is set.
Rebinding (legitimate rename): a new y' with c(x, y') ≥ 5 over ≥ 2 normal days from
trusted, non-held events and no occurrence of y*_x in the last 5 events of x → y*_x ← y'
(`binding_changed` INFO, cver + 1, confidence channel of the pair reset). Output per node:
`{X, Y, fd: {g3, n, holds, one_to_one}, table: {x: (y* | set, n_x, LB_x | U_x, first, last)}}`.

### 6.13 Time windows (P09)

Per txn node and day type (workday / non-workday), every 6 h, for the nodes whose
evidence grew by ≥ 10 % or ≥ 20 units since their last fit, or whose drift state changed
(dirty nodes; §6.20 — the rest keep their windows):

1. Density: the node's `hist96` (smoothed with a 3-slot kernel), and the minute
   reservoir where P09 requested it (≥ 50 % of mass in ≤ 4 slots).
2. Unwrap the circle at the minute of lowest smoothed density.
3. Bayesian Blocks (Scargle, Norris, Jackson & Chiang 2013), event mode with weights,
   block fitness `N_k (ln N_k − ln T_k)`, prior `ncp = 4 − ln(73.53 · p0 · n^−0.478)`,
   p0 = 0.05; O(n²) on ≤ 256 reservoir points or 96 slots.
4. Windows = maximal runs of adjacent blocks with rate ≥ κ · r_bg (κ = 3,
   r_bg = mass/1440 per minute); drop windows with < 5 % of mass; edges at minute
   resolution from the reservoir, else slot edges.
5. Confidence: coverage (mass inside), n distinct dates, and day stability (share of
   dates whose events all fall inside the windows).

The root windows of each system's txn tree become the learned level ℓ2 of the time
hierarchy (`w:0900-0921`), so P04 can split on them. P03's `p_when` is the highest-density
region p-value under the node's density for the event's day type,
`f(s) = (h(s) + α_t/96) / (N + α_t)` with α_t = 0.1, where h(s) = the slot's mass share ×
N and N = the node's evidence for that day type on the confidence channel (PPC-9: mass
would make a thinned or aggregated node look far more certain than it is; an empty slot
has a small, not a zero, density):
`p_when = Σ_{s : f(s) < f(s_obs)} f(s) + ½ Σ_{s : f(s) = f(s_obs)} f(s)`, with minute
refinement inside windows and back-off to the all-day-type density, then to the parent,
when the day type has n_c < 20; holidays score against the non-workday density, and a
make-up workday against the workday density. An event in an empty slot therefore scores about
0.05/N: a login at 03:05 against 60 evidence units of 09:00–09:21 logins gives ≈ 8e-4,
and the evidence strengthens with every observed day. Rendered: "工作日 09:00–09:21（覆盖 97 %，21 个工作日）".

### 6.14 Workflows (P10)

Actions are patterns. The action of an event is `act(e) = (r, v)`: r = its route
template (ℓ1; for non-HTTP channels the SNI eTLD+1, the templated qname or `net.dst`),
v = the deepest node on its path whose own split is on a content attribute (an action
variant, §6.5.2), else 0; actions get integer ids from a per-system dictionary
(SpaceSaving k = 4096). Ids are never reused: an evicted action's id is retired together
with its DFG edges, so a later action never inherits another's statistics. `act_node` = the shallowest node on the path whose context
constrains the route, used for rendering. Actions therefore stay stable when the tree
later splits them by who or time.

Session state (every tick, all events of the tick, needed for scoring): per session key
(ip, sess.key) (LRU, cap allocated by P15, default 65 536 per system) the previous action,
its time and a 64-bit hashed set of actions seen; P03 writes `prev_act` and the delay into
`pat.assign`. Counting (from the learned rows of tick t − D with trust, §6.9.3, using the
recorded `prev_act`, so an attacker's session is not learned before it can be judged). Per system
and scope g ∈ {`*`} ∪ {groups with ≥ 2 IPs active in the system} (at most 32 scopes):
edge SpaceSaving (k = 4096 per system) keyed (g, a, b) with decayed counts and a
12-bin log2 histogram of the delay (1 s … 1 h, then "long"); action, start and end
counts; eventually-precedes counts (g, b, a) for the 256 most frequent b (SpaceSaving
k = 4096).

Mining (every 6 h):

```
dep(a ⇒ b) = (|a>b| − |b>a|) / (|a>b| + |b>a| + 1)          (heuristics miner, Weijters & van der Aalst 2003)
keep edges with dep ≥ 0.8, |a>b| ≥ 10 evidence units, and ≥ 5 % of a's outgoing mass
(|·| in evidence units on the confidence channel)
workflow   = maximal simple paths (≤ 8 actions) from start actions (start share ≥ 0.1) along kept
             edges, support = min edge count, delay band per edge = [q10, q90] of its histogram,
             time anchors = P09 windows of each action (e.g. report submission at 17:00–17:10)
requires(b, a) ⇔ c(b) ≥ 15 and the 5 % quantile of Beta(½ + c(b with a earlier), ½ + c(b without a)) ≥ 0.85
```

Scoring (P03): `p_trans` = highest-density p of b among the successors of a under the
hierarchical estimate (scope g → `*` → action marginal, α = 2); `p_req = (c(b without
a) + 0.5)/(c(b) + 1)` for each required predecessor a absent from the session;
`p_seq = min(1, 2 · min(p_trans, p_req))`. P10 runs only where P12 finds sessions
identifiable (§6.18); elsewhere p_seq is NaN.

### 6.15 Who-discovery (P11)

1. **Signatures.** Per IP (org-wide, over systems in `ip` mode) a SpaceSaving(k = 24) of
   items `(tree key, act)` (a family counts as one system, §6.20) with H_m-decayed counts,
   updated from the learned rows of t − D with trust; `shared:<ip>` items get no signature
   (several users); at most
   S_max = 50 000 IPs (LRU by last activity; IPs with < 10 evidence units are dropped
   after 7 d). Systems in `prefix` mode contribute items keyed by /24 instead (their
   "users" are prefixes); systems in `none` mode contribute nothing.
2. **Similarity.** Daily, weighted MinHash by Improved Consistent Weighted Sampling
   (Ioffe 2010), k = 64, over each signature; LSH with b = 16 bands × r = 4 rows gives
   candidate pairs; each IP keeps its 10 best candidates with Ĵ ≥ 0.2; the graph is the
   mutual-kNN graph weighted by Ĵ.
3. **Communities.** Louvain (Blondel et al. 2008), resolution 1, deterministic node
   order (IP as integer), on the mutual-kNN graph; the final level is the group level,
   the first level is kept as sub-groups for display. Singletons are ungrouped (`grp:∅`).
   Between daily runs the assignment is incremental: an ungrouped IP that reaches 10
   evidence units joins the group holding the weighted majority of its LSH neighbours
   (one label-propagation step, Raghavan, Albert & Kumara 2007) if that majority is ≥ 0.6;
   the daily run re-optimises. A `readdress_candidate` (§6.12: a never-seen IP that
   submits the bound value of an IP that has gone silent) joins the old IP's group at once
   with a provisional flag, cleared by the next daily run if its signature disagrees.
   Each group with its representative patterns (item 5) is a bicluster of the IP × pattern
   matrix.
4. **Stable ids.** Hungarian matching on 1 − Jaccard(members) against the previous run;
   inherit at J ≥ 0.3, else a new id (`group_formed` / `group_changed`); an IP moves only
   after 2 consecutive runs assign it elsewhere.
5. **Labels and names.** Representative patterns: top-5 actions by
   `lift · sqrt(support)` with lift = P(a | g)/P(a) ≥ 2. Config
   `who_group_names: [{name: '综合部', ips: [...]} | {name, cidrs: [...]}]` assigns a name
   to the group with the largest Jaccard(members, configured set) ≥ 0.5 (Hungarian when
   several); otherwise the auto name `G<id>` plus the short labels of its top-2 patterns.
   The configured list is a data interface, not a hand-kept table: an IPAM export, DHCP
   scope descriptions (`dhcp_scopes[].name`), an asset inventory or a directory export may
   write it (W-P9 provides a CSV/JSON importer). Traffic can tell *who belongs together*;
   only such a source can tell that the group is called 综合部.
6. **Prefix covers.** Greedy bottom-up on the IP trie: the smallest CIDR set covering
   ≥ 90 % of members with purity (members / active IPs inside) ≥ 0.8 — the "某几个 IP 段"
   rendering and level `reg` of the IP hierarchy when no config region matches.
7. **Export.** `ip2g` for members (lib/phier level `grp`), group records, and optional
   class keys `class:grp:<gid>` at (`__org__`) for B18/B30 (§9). Group views (§6.17.3) are
   materialised for the G_max = 256 groups with the most mass and for named groups; others
   are rendered on read.

Cost: signatures O(1) per learned event; daily clustering O(n_IP · k) for MinHash,
O(n_IP · b) for LSH, O(E log n) for Louvain with E ≤ 10 n_IP.

### 6.16 Conformity: scoring every event against the most specific confident pattern (P03)

P03 runs before P04 in the same tick, on every event of the tick (not only the learning
sample), against the tree and the fitted constraints as they were at the end of the
previous tick (prequential; P03 never writes a model).

#### 6.16.1 Covering pattern with back-off

```
path  = route(tree, e)
conf  = deepest N on path with state ∈ {confirmed, stable, evolving, stale}
if e.ip in conf.exc: X = conf.exc[e.ip] (for its targets) else X = conf
for each violation type t: use the deepest node on path[: index(conf)+1] that has a
fitted model for t with sufficient support (n_c ≥ 20 on the confidence channel; when: ≥ 20
for the day type);
if none, p_t = NaN (unscored, never 1)
```

#### 6.16.2 Typed p-values

| Type | p-value | Source |
|---|---|---|
| who | The node's who level ℓ* = the finest IP level at which its who summary is *closed*: unseen mass U_ℓ ≤ 0.05 (confidence channel, §5.5.3), the *heavy set* (the smallest set of items, by mass, that covers ≥ 95 % of the node's mass) has ≤ 8 items, and the evidence spans ≥ 5 normal days (a one-day burst is not a closed population). Items outside the heavy set are not members, however often they appear (§6.9.2). No closed level → NaN (an open population has no who constraint). If gen(net.src, ℓ*, ip) is a heavy item → 1; else `p_who = U_ℓ*` (Good–Turing probability that the next event comes from an unseen member). Flags: `outsider_group` when the IP *has* a group (grp ≠ ∅) and that group has zero mass at the node; `unknown_ip` when the IP has no group and no history in this system (absent from the root who summary at /32 and from P11's LRU); `system_new` when the IP's group has zero mass in the whole system; plus `readdress_candidate` / `concurrent_use` from the binding check (§6.12). | §5.5.3 |
| when | HDR p under the node's day-type density (§6.13). | P09 |
| content | Per target attribute: numeric → §6.10; categorical → HDR p of the value under the node's evidence-scaled hierarchical-Dirichlet predictive (§6.5.3); set → new key U_key or missing required key (§6.11); text → grammar / closed set / length (§6.11); binding → p_bind, set-binding U_x (§6.12); invariant → `(n_viol + 0.5)/(n + 1)` when the invariant is broken; per-IP intensity → numeric p of the IP's running count in the current hour at this node against the node's `rate.ip_h` bounds (§6.16.4). `p_content = min(1, m · min_a p_a)` over the m attributes checked. | P06–P08 |
| seq | §6.14. | P10 |
| novel | The event's action (§6.14) is not in the system's action dictionary: `p_novel` = the dictionary's unseen mass U (§5.5.3 convention); or it landed in the `other` branch of a confirmed route split whose `other` child is not confirmed: U of the route level at that node. | P04, P10 |

Every p_t is computed against the current constraints and the reference snapshot:
`p_t = min(1, 2 · min(p_cur, p_ref))` (p_cur alone before the first snapshot). While an
attribute of the node is `evolving` with a coordinated (≥ 2-IP) change,
`p_t = max(p_old, p_new)` for that attribute and any finding on it is capped at LOW
(§6.9.2). Event p: `p_ev = min(1, 5 · min_t p_t)`; `vtype` = the types with p_t ≤ 1e-3.
All counts n, N in these formulas are evidence on the confidence channel (PPC-9, §6.9.4).

#### 6.16.3 Outputs

- `pat.assign` (row-aligned with the tick's batches): leaf, conf node, act, prev_act,
  act_node, exception flag, the five p_t, p_ev, vtype, damp (§6.9.3).
- Per (s, ip) with events this tick and each type t:
  `p_tick,t = 1 − (1 − min_e p_t(e))^{n_t}` (Šidák over the n_t scored events), written as
  `behavior.score[conf_t] = −log10 p_tick,t` and `behavior.pm[conf_t] = p_tick,t`, plus
  `behavior.axes`. IPs without events this tick get nothing (NaN; absence is data).
- Discrete `pattern_violation` events for events whose node is confirmed or stable, by
  the table below; dedupe key `pv|<nid>|<type>|<ip>|<local date>`; extra =
  `{pattern_id, statement_zh/en, type, flags, observed, expected, p, p_day, U, sensitivity,
  n_c}`. **Per-day multiplicity.** A busy IP is scored many times a day, so a per-event
  threshold would give it many chances to cross by chance. The content, when and seq
  thresholds of the table therefore apply to
  `p_day = 1 − (1 − p_min)^{n_day}`, where p_min is the smallest p_t of the IP at that
  (node, type) today and n_day the number of its scored events there today (a running
  counter in the same bounded per-(ip, node) structure as §6.16.4). The expected number of
  chance emissions per clean IP-day is then at most Σ over the (node, type) cells it
  visits of the threshold — for a typical 5–10 cells and the LOW threshold 1e-3, ≤ 0.01,
  inside the PG6 budget of 0.05 — provided the model p-values are valid, which PG6's KS test
  checks. who and novel use U, which is already a probability of the first unseen event;
  the dedupe key makes them once per IP, node and day.
  At most V_max = 20 per system per tick (highest severity first; the rest are counted in
  health). Node sensitivity `σ(N) ∈ [1, 3]` = 1 + [write method] + 0.5·[≤ 3 distinct IPs at
  ℓ*] + [route matches `sensitive_patterns`], capped at 3.

| Type | Condition | Severity (default, calibrated in PG6) | Axes |
|---|---|---|---|
| who | U_ℓ* ≤ 0.01, σ ≥ 2, `outsider_group` | HIGH candidate (B25 corroboration rules apply) | privilege; lateral if `system_new` |
| who | U_ℓ* ≤ 0.02, σ ≥ 2, `outsider_group` or `unknown_ip` | MEDIUM; LOW when `readdress_candidate` | privilege |
| who | U_ℓ* ≤ 0.05 | LOW; INFO when `readdress_candidate` | privilege |
| content / binding | `cross_binding` with LB_x ≥ 0.9 (§6.12) on a node whose route is an auth route (POST answered 3xx/2xx then followed by session activity, or `sensitive_patterns`) | MEDIUM; HIGH candidate at ≥ 2 events | credential |
| content / binding | `concurrent_use` (a bound value used from a foreign source while a bound source is active, §6.12) | MEDIUM; HIGH candidate together with a who violation on the same event | credential |
| content / binding | y unseen for a bound x; value outside a set binding | LOW | credential |
| content | p_day ≤ 1e-4 on a write action, or `injection_shape` | MEDIUM | content; exfil when the attribute is an upload size on the upper tail |
| content | p_day ≤ 1e-3 | LOW | content (volume for `m.*` volume metrics and `rate.ip_h`) |
| when | p_day ≤ 1e-3 and outside all windows of the node | LOW; MEDIUM on a write action at night daypart | temporal |
| seq | a required predecessor (requires(b, a) holds, §6.14) is missing from the session | LOW; MEDIUM on a write action | sequence |
| novel | new action at system tier matching `sensitive_patterns` or a write | MEDIUM; otherwise INFO | categorical; privilege when sensitive |

The table is a default calibrated on seeds 0–1 (PG6) and reported on seeds 2–4; it mirrors
B08's tier-based discrete severities.

#### 6.16.4 Per-IP intensity at IP-agnostic nodes

Where who is open (large populations), per-IP abuse shows as intensity. P03 keeps, for
the current local hour, per system one SpaceSaving over (ip, conf node) pairs with
k_int slots (allocated by P15, default 16 384; exact while fewer pairs exist), cleared at
the hour. Scoring uses the *guaranteed* count `count − error` (so sketch error can never
raise an alarm; every pair with more than N_hour / k_int events is tracked, e.g. more than
6 events at 100 000 events per hour) against the node's `rate.ip_h` target (P06 bounds on
the distribution over IPs of events per IP-hour). At the hour close P03 writes the
finished guaranteed counts, plus the node's untracked mass and HLL-estimated number of
untracked IPs, into the batch `pat.rate`; P04 learns `rate.ip_h` from them at t − D like
any other learned input. `rate.ip_h` is an always-on numeric target of every node on the
route level (it is not counted in m_t). This is how "one IP makes 400 login attempts an
hour on a portal where any IP may log in" becomes a content (volume, credential)
violation, without a per-IP structure that grows with the population.

#### 6.16.5 Complexity

Per event: routing O(D), type checks O(m_t + #bindings at the node + 1) dictionary or
array lookups, plus one t-digest CDF (O(log δ)) per numeric attribute. Per-IP state is the
bounded hourly intensity sketch only. Estimate 15–40 µs per event in Python; PG4
measures. Under overload the last step of the degradation ladder (§6.19) samples scoring
on low-risk strata; nothing else in P03 is sampled.

### 6.17 Facets and views (P13, P14)

#### 6.17.1 Facet registry

A facet declaration (written by any engine into `model.facets` under its own key, so
facets are added at runtime without code changes in P13):

```
{id: 'content.bindings', parent: 'content', name_zh: '绑定关系', name_en: 'Bindings',
 subjects: ['system', 'group', 'ip'], producer: 'behavior.binding',
 sources: ['model.pbind'], applicable: 'payload_visible', render: 'binding_v1', order: 43}
```

Default facet tree (the analogue of a person's appearance / biology / social attributes):

| Facet | Sub-facets (producers) |
|---|---|
| 1 功能（外观）functional | actions and routes (P04), action variants (content splits, P04), channel mix (P02) |
| 2 时间节律 temporal | activity windows (P09), weekly rhythm (B07, earned IPs; P09 per day type otherwise), automation periodicity (B12, D0) |
| 3 空间/网络 spatial | who sets / prefixes / regions (P04 who summaries, P11 covers), destinations (B08 dimensions), cross-system footprint (P11 items, B21) |
| 4 内容 content | size bands and bounds (P06), payload grammar and key sets (P07), bindings (P08), invariants (P05/P04), per-IP intensity (P06 `rate.ip_h`) |
| 5 序列/流程 sequential | workflows and required predecessors (P10), personal action grammar (B10, earned IPs) |
| 6 关系/社会 relational | behavioural group and members (P11), co-access neighbours (P11 kNN), links / aliases / shared IP (B17), role class (B02) |
| 7 技术 technical | client stacks (R3, B09), TLS posture (P02 over `tls.*`) |
| 8 量/预算 volume | `win`-tree bands (P06 on `m.*`), budgets (B13, earned IPs), workload bands (B30) |
| 9 身份/连续性 identity | separability and confusables (B15), attribution (B16), binding stability (P08) |
| 10 风险/健康 risk | risk (B26), incidents (B27), violation history (P03) |

P13 composes, per subject (system, group `class:grp:<g>`, B02 class, and IP — IPs only
for earned IPs on its 2-h cadence, every other IP on API read, the same rule as B30 in
bounded mode, §10.3), the facet tree
with the non-empty facets whose `applicable` predicate holds for the subject's system
(P12 characteristics), each facet payload `{items: [statement], confidence, as_of,
version}`. A portrait version is stored with `store.put_profile_version`; the diff uses
B30's thresholds (categorical JSD > 0.1, quantile move > 25 %, window move > 1 h) plus
pattern lineage ops. B30 embeds P13's facet tree for IPs and classes (§9).

#### 6.17.2 Statement objects and rendering rules

```
{id, pattern_id, view: 'system'|'group', subject, text_zh, text_en,
 support: n_c, confidence, first_seen, last_seen, version, cver, state,
 facets: [facet ids], evidence: {who, when, content, bindings, workflow}}
```

- **Who**: ≤ 8 IPs covering ≥ 95 % with U ≤ 0.05 → the IP list; else a group covering
  ≥ 90 % → the group name (members listed if ≤ 5); else ≤ 4 prefixes or regions covering
  ≥ 90 % → the prefix list; else "任意 IP（约 N 个，分散）".
- **When**: P09 windows per day type, minute resolution when available.
- **Numbers**: §6.10 rounding; units KB/MB; percentages without decimals unless < 1 %.
- **Confidence** = min over the statement's constraints of: 1 − U (who), coverage ×
  day stability (when), 1 − |empirical coverage − nominal| / nominal (bands), c_g ·
  (1 − U_s) (grammar), LB_x (bindings), min over edges of the dependency and of the Jeffreys 5 % quantile of the edge's share of a's successors (workflow); × 0.5 for stale
  patterns; candidate patterns are not rendered by default.

Example (system view, the requirement's OA case; numbers illustrate the format only):

> 【OA 系统 · 192.168.100.100:8080】工作日 09:00–09:21，综合部（192.168.1.21、192.168.1.23、
> 10.168.7.121）访问登录页（POST /login）：提交数据量 90 % 在 1–2 KB，全部在 0.5–3 KB
> （n = 183，下次越界概率 ≤ 1.1 %）；表单必含 username=，取值 `[a-z]{4}`（3 个取值的共同形状）；绑定：
> 192.168.1.21 → username=jack、192.168.1.23 → username=rose、10.168.7.121 → username=mike
> （g3 = 0，各 ≥ 60 次）。随后 192.168.1.21 进入审批页（GET /approval/list → POST
> /approval/{num}/approve，间隔 1–6 分钟）；17:00–17:10 192.168.1.23、10.168.7.121 向报告生成页
> 提交数据（POST /report/generate，20–60 KB）。置信 0.97 · 首次 2026-09-01 · 最近 2026-09-28 · v7.3。
>
> [OA · 192.168.100.100:8080] On workdays 09:00–09:21, 综合部 (192.168.1.21, 192.168.1.23,
> 10.168.7.121) opens the login page (POST /login): 90 % of submissions are 1–2 KB, all
> within 0.5–3 KB (n = 183, P(next outside) ≤ 1.1 %); the form always carries username=
> matching `[a-z]{4}`; bound values 192.168.1.21 → jack, … Confidence 0.97, first seen …

Group view (same store, other projection): "综合部（3 个 IP）使用 OA 与邮件：OA 工作日
09:00–09:21 登录 …；在财务系统中从未执行写操作（21 天、0 次）"; the last clause is a
*negative statement* rendered from who-closed write nodes of other systems where the group
has zero mass — the fact that makes "综合部去财务系统审批" a who-violation, and a statement
the operator can read before it ever happens.

#### 6.17.3 Views

- **System view** (`model.pviews@(s,__system__)`): system → act_nodes (sorted by mass) →
  per act_node the who summary, windows, content constraints, bindings, variants
  (content-split children), exceptions → workflows through the act_node.
- **Group view** (`model.pviews@(__org__, class:grp:<g>)`): group → members / prefixes
  → systems where the group has ≥ 5 % of its mass → actions with lift ≥ 1 or group
  share ≥ 20 % → when / content / bindings restricted to the group's members →
  workflows of scope g → negative statements.
- **IP view**: the group view of the IP's group, plus the IP's exceptions, its bindings and
  the B-engine facets for earned IPs.
- **Member of a system family** (§6.20): the system view of the family tree restricted to
  the nodes whose context admits the member's `net.dst` (every node without a `net.dst`
  constraint, plus the member's own branches and exceptions). Where members' users or
  actions differ, the tree has split on `net.dst` and the member's view shows its own
  statements; where they do not, the member's view is the family's, which is the truth.

Views are refreshed every 2 h per key (entity_due phase) and on API read when older than
15 min.

### 6.18 Scenario adaptation: system characteriser and strategy selector (P12)

#### 6.18.1 Characteristics (daily per system)

| Characteristic | Definition | Source |
|---|---|---|
| population | distinct src IPs in 7 d (HLL); mean distinct IPs per active hour | P04 root who; P01 `win` counts |
| churn | new IPs per day / population; share of IPs with lifetime < 1 d; share of IPs in `dhcp_scopes`; B17 `entity_resolution` rate; `readdress_candidate` rate | P11 LRU, config, events, P03 |
| ip_info | max over IP levels of CR(net.src@ℓ) and the argmax level | P05 |
| snat | `snat_suspect` (§5.1.4); share of mass from `shared:<ip>` items | P00, P02, B17 |
| route_card, growth | distinct route templates (7 d); Heaps exponent β of distinct-vs-events growth | P02 |
| payload_vis | share of txn events with `http.*`, with `body.*` or `q.*`, and opaque TLS share | P02 coverage |
| sess_ident | max(coverage of `sess.key`, share of IPs with one client stack in ≥ 90 % of events × share without B17 `shared_ip` / NAT flags × [bimodality coefficient of log gaps > 0.555]) | P00, P01, R3, B17 |
| volume | events/day; learned share | P00 |
| automation | share of IPs with `derived.periodicity_score` ≥ 0.8 | D0 |
| calendar | share of mass on non-workdays; strength of monthly structure (P05's U_s of `ctx.mend` / `ctx.dom`) | P01, P05 |
| family | family id and member similarity (§6.20) | P12 |
| criticality | config `ip_classes` / system criticality | config |

#### 6.18.2 Strategy dimensions, preconditions, utility

| Dimension | Arms | Preconditions | Utility measurement |
|---|---|---|---|
| who granularity | ip, grp, prefix, reg, none | ip: ip_info at /32 ≥ 0.05, churn < 0.3/day and not snat; grp: P11 groups cover ≥ 50 % of events; none: no level's address code saves ≥ 1 bit/event | full information: the held-out behaviour gain of every level (below), with the per-level address code as tie-break |
| P06 bounds | on (always) | numeric targets exist | full information (P06 reports its prequential gain) |
| P07 grammar, P08 bindings | on / off | payload_vis (body or query) ≥ 0.05; P08 also who ≠ none or `sess.key` present | full information when on; while off, a 1-day probe every 14 days |
| P10 workflow | on / off | sess_ident ≥ 0.6 and route_card ≥ 5 | budgeted UCB (Tran-Thanh et al. 2012) over the gain/cost ratio; exploration 1 day in 14 |
| `win` tree | on / off | – | full information |
| node tier | XS 32, S 256, M 1024, L 4096 | – | marginal utility of the lowest-utility decile of nodes (§6.6) |
| B-engine earned cap (§10.2) | 8, 32, 128 | – | B03/B04 earned-vs-class prequential gain |
| B-engine applicability | per engine on / class-only / off | B10 per-IP PPM: sess_ident ≥ 0.3; B11/B12: automation > 0 or the §10.3 prefilter; B09/B15: client-stack visibility (share of events with JA3 or UA) ≥ 0.2; B13 exfil budgets: upload-capable routes exist; content-derived B-detectors off when payload_vis < 0.05 | preconditions only, plus a veto from B23 labels: a detector family with ≥ 5 `fp` labels and label precision < 0.2 over 30 d on the system drops to class-only |

Who-level code length (the full-information measurement for the first row): every who
summary accumulates, per level ℓ, the prequential code length of each learned event's IP
under a two-part code, `L_ℓ(ip) = −log2 p_ℓ(g) + log2 |g|`, where g = gen(ℓ, ip) is the
item, p_ℓ the level's evidence-scaled hierarchical-Dirichlet predictive, and |g| the size
of the item's address space (1 for /32, 256 for /24, 65 536 for /16, the address count of a
group's or region's prefix cover, 2^32 for `*`; IPv6 analogously). An item never seen at
the level is coded as the escape `−log2 U_ℓ` plus the item's own address bits (32 for a
/32, 24 for a /24, 16 for a /16, log2(#groups + 1) for grp/reg). Every level is thus a
complete code for the IP, and the per-level sums are comparable. Under DHCP churn or a
large random population the /32 model keeps paying escapes plus 32 bits for new IPs and a
prefix, region or `none` model wins; for a stable department /32 wins (grp costs about the
same when the group's cover is its members' addresses). The per-level sums (5 floats per
who summary) are summed over leaves by P12.

**Who utility (revised, §16.2 A1).** The address code above measures how compactly the
population is stated at a level, not what conditioning on that level buys. The who arm
decides at which granularity behaviour is conditioned on who (P04's @who coding target,
P11's signature level, P08's screened levels), while the who summaries themselves are kept
at every level whatever the arm (§5.5.3). Its utility is therefore the **held-out
behaviour gain** of the level: P12 codes the behaviour b = (action, workday, local hour) of
every learned event prequentially, by the marginal p_m(b) = (n_b + 1/2^16)/(M + 1) and given
the event's who item g at level ℓ, p(b | g) = (n_{g,b} + p_m(b))/(n_g + 1), each block coded
before it is learned; G_beh(ℓ) = (L_marginal − L_ℓ)/n bits per event, by the chain rule the
out-of-sample information I(who_ℓ; behaviour). The (item, behaviour) statistics are
Space-Saving with the same slots at every level and at most 384 sources (bottom-k by an
address hash) per tick, so a level whose statistics do not fit pays in evictions — the gains
compare arms at equal memory, i.e. gain per cost. Then

    U(arm) = G_beh(arm) + 0.05 · clip(32 − bits(arm), 0, 32)/32 − 0.001      (none: U = 0)

the second term (≤ the 0.05 bits/event switching margin) letting the address code decide
only between levels whose behaviour gains are within noise (a system used by one
department: no level predicts behaviour, the level that states the population most compactly
is its who); `none` is ruled out while some level's address code saves ≥ 1 bit/event (the
population has structure: who uses the system is itself the pattern). Measured offline on
pack O (seed 0, bits/event): oa G_beh grp 0.92, /24 0.90, ip 0.13 (研发's DHCP users
re-address daily, a per-IP model relearns them); finance ip 0.20 > grp 0.11 (approver vs
bookkeepers inside one department); portal ip −2.0 (returning visitors: their addresses code
well, their behaviour per IP does not), /16 0.25. Each Hedge round is one completed day's
utility (each component — address code, behaviour gain — falls back to its 7-day figure
while the day has < 30 of its evidence units): feeding overlapping
7-day sums counted every day seven times and delayed the leader by a week (§16.9 A11); the
switch margin is twice the detrended noise of the daily utility difference (A9).

Utility of an arm, in the single currency of PPC-5: `U = gain − λ_c · cost` bits per event,
where gain = prequential gain in bits per event (every model is scored before it learns,
so the gain is held-out by construction) against the arm's baseline (the node categorical
distribution at the coarsest level for content engines; the action marginal for P10; the
`*` level for who), cost = measured µs per event (engine health and P15), λ_c = 0.001
bits/µs as in P05. Selection per system: arms whose preconditions fail are never chosen;
the remaining full-information dimensions use Hedge (exponential weights, η = 0.5 per day)
on U clipped to [−8, 8] bits/event (Hedge's regret bound needs bounded losses; a ratio
gain/cost is unbounded as cost → 0), and the chosen arms are packed greedily by gain/cost
into the system's CPU and memory budget from P15 (a knapsack by utility per unit cost). A
switch needs the new arm's U ≥ the current one's + 0.05 bits/event for 3 consecutive days
(`strategy_changed` INFO). P12 writes `model.sysprof@(s,__system__) = {characteristics,
arms{dim: {arm: {gain, cost, U, weight}}}, chosen{dim: arm}, history}`; P00–P11 and P15
read `chosen` at their next run.

Typical outcomes the rules produce (and PG8 checks): a departmental OA → who = ip or grp,
P07/P08/P10 on; a large portal with irregular IPs → who = prefix or none, P08 off (no
binding pays), P07 on, per-IP intensity carries per-IP abuse; an opaque-TLS mail system →
P07/P08 off, who = grp, P10 off unless sessions are identifiable from sizes/timing;
an API gateway of machine clients → who = ip, P10 on (machine call sequences), P07 on
JSON bodies; a DHCP pool → who = grp or prefix, bindings keyed by `client.stack` or
`sess.key`; a system behind an SNAT proxy without `trusted_proxies` → who = none and a
configuration hint in the view.

### 6.19 Resource governor (P15)

Budgets (config `progressive.budget`), in units that do not depend on the tick length:
`pcore_cpu_share` (fraction of one core used by P00–P15, averaged over 1 h; default 0.25),
`lib3_cpu_share` (all of lib-3 including B01–B30; default None = report only, until PG9
has measured bounded mode), `mem_mb_total` (default 2048), optional per-system caps.
The first draft used the gate-14 latency figure (80 ms per tick at 35 IPs) as the budget;
today's lib-3 already needs ≈ 909 ms per tick at 35 entities (integration.md §9.2) and the
P-core alone needs ≈ 1–2 s per 900-s tick on pack O (§7.3), so the ladder would have been
permanently engaged. Tick latency remains a separate gate, not a budget.

Each tick P15 reads engine durations (engine health), P-core model sizes
(`lib/pnode.nbytes`, `psketch.nbytes`) and `store.memory_report()` (every 15 min), then:

1. allocates per tree (§6.20) the node tier, L_max, e_rate, W_max, A_win, R_p, the LRU caps
   (sessions, evidence runs, intensity k_int, held H_max), the P11 signature share of S_max,
   and the B-engine earned cap E_max, by water-filling on P12 utility × criticality × recent
   learned-event rate within the global budget: an idle tree gets tier XS and minimal caps,
   a tree idle for 30 d is checkpointed and evicted (§6.20);
2. applies the degradation ladder when a budget is exceeded for 3 consecutive ticks, one
   step per 3 ticks, undone in reverse after 24 h under 70 % of budget:
   (1) halve e_rate (learning only; scoring unaffected) → (2) stop exploration arms →
   (3) halve L_max → (4) lower the node tier (prune by utility) → (5) lower E_max (B-engines
   return IPs to class mode; only when `resource_mode = bounded`) → (6) B29/B30/P13/P14
   refresh on read only → (7) last resort, scoring priority: P00 skips body parsing and P03
   scores a uniform 1-in-k sample (HT-weighted in the tick scores) of events on
   IP-agnostic, read-only, non-sensitive route nodes, while write actions, sensitive routes,
   who-closed nodes, new IPs and flagged IPs are always scored in full;
3. publishes `model.budget@(__org__,__org__)` = caps, the active set A_t(s) and the earned
   set E_t(s) (§10.2), and `ops.budget` (measured ms and MB per engine and tree).

Active set `A_t(s)` = IPs with observations in (t − Δt, t] ∪ IPs with an open incident ∪
quarantined IPs ∪ IPs with non-zero accumulator state ∪ IPs with held rows; the last three
come from sets that B28, B13 and B27 publish (§10.1), never from a scan of all entities.

### 6.20 Servers are not enumerated either: system families, activity-proportional state

The requirement forbids traversing all users *and servers*. A design whose every system
carries a fixed ≈ 20–30 MB and a fixed set of hourly fits (the first draft) grows linearly
with the number of servers, however cleverly it treats IPs. Three mechanisms remove that.

**1. System families.** Many servers run the same application: a load-balanced OA
cluster, fifty branch servers of one ERP, a fleet of API nodes. P12 computes daily, per
system, a weighted MinHash signature (k = 64) over its route prefixes (ℓ2) and hosts, SNI
eTLD+1, destination port, top client stacks and registered attribute names. LSH (16 × 4)
proposes pairs; systems with weighted Jaccard ≥ 0.6 on two consecutive days and a
compatible channel mix (payload_vis within 0.2) form a family; config `system_families`
can force or forbid membership. A family shares **one** pattern tree, attribute registry
and attribute selection under the tree key `fam:<id>` (store key (`fam:<id>`,
`__system__`)); `model.sysfam` maps each system to its tree key, and P03/P04 route each
system's batch into its tree. The server becomes an ordinary attribute, `net.dst` with the
hierarchy ℓ0 `peer:port` → ℓ1 member system id → ℓ2 site / region (config) → `*`, which
P05 may make a split candidate like any other. Where members differ (a branch server whose
users are only its branch), the tree splits on `net.dst` because that pays; where they do
not, one set of patterns describes them all. This is the server-side mirror of "一类用户 →
某一个用户": "一类服务器 → 某一台服务器", decided by the same evidence rule.

A member leaves its family (its subtree is copied into its own tree, lineage kept) when
the family tree's split on `net.dst` gives it a branch holding ≥ 20 % of the family's nodes
for 7 days, i.e. when sharing no longer saves anything; `family_changed` INFO.

**2. Activity-proportional state.** Every per-tree cap — node tier (XS 32 nodes with no
learning leaves until the tree receives ≥ 200 evidence units a day; S, M, L), learning
leaves, probe size R_p (256 at XS), registry cap (128 at XS), LRU caps, intensity slots,
held events, W_max — is allocated by P15 from the global budget in proportion to utility ×
criticality × recent learned-event rate (§6.19). A tree with no event for 30 days is
written to the store as a checkpoint (`put_checkpoint`) and dropped from memory; its next
event restores it. Memory is therefore O(budget) and, below the budget, O(Σ over active
trees of their tier), never O(#servers × constant).

**3. Dirty-node periodic work.** P05 (hourly), P06–P08 (hourly), P09 (6-hourly) and P10
mining (6-hourly) skip trees that received no learned events since their last run and,
inside a tree, refit only *dirty* nodes: nodes whose evidence grew by ≥ 10 % or ≥ 20 units
since their last fit, or whose drift state changed. Periodic CPU is then proportional to
the evidence that arrived, not to #systems × N_max.

**Cold start of a new server.** A new system whose first-day signature matches a family
joins it on the second daily match and is scored at once against the family's nodes
without a `net.dst` constraint, so a new branch server of a known application is profiled
from its second day. A system that matches nothing starts its own XS tree; until it has
confirmed nodes, its events are scored for `novel` at system tier only (and by the
B-library), and its view says "新系统，画像学习中（已观察 N 天）".

### 6.21 Real-world situations and how the core adapts to them

Each row names the signal by which the situation is recognised, the automatic response,
what is still not possible, and the pack-O-real item that tests it (§11.4, PG11).

| Situation | Signal | Automatic response | Residual limitation | Test |
|---|---|---|---|---|
| Reverse proxy / load balancer with source NAT (every user appears as the proxy) | `trusted_proxies` + `x-forwarded-for`; else `snat_suspect` (§5.1.4) | who resolved from XFF; otherwise who = none, the proxy is not profiled as a user, the view carries a configuration hint | without XFF no per-user who facts exist | R1 |
| NAT branch office, VDI / terminal server (many users, one IP) | B17 `shared_ip`; many client stacks, session keys or bound values per IP | IP maps to `shared:<ip>`; no exceptions or per-IP bindings for it; sessions and bindings keyed by `sess.key` when present | without `sess.key` the users behind it are profiled as one class | R2 |
| DHCP, VPN pools, IPv6 privacy addresses | churn, `dhcp_scopes`, short IP lifetimes, /64 | who at grp / prefix / /64; `readdress_candidate` (a new IP carrying a silent IP's bound username) is LOW, joins the old group provisionally | a pool shared by several departments supports prefix-level who only | R3 |
| Encrypted traffic without decryption (TLS, TLS 1.3 with ECH) | payload_vis ≈ 0 | content engines off; the tree works on sizes, timing, SNI, JA3, `net.dst`; P10 only with identifiable sessions | no username, grammar or binding facts; the view says so | pack O mail, code |
| Shared terminal (one PC, several people) | per-IP purity low but a small closed value set | set binding "IP → {jack, rose}", violations outside the set | none beyond the set semantics | R4 |
| Service accounts, machine clients | automation index, one value used from several IPs | reverse set binding "svc_backup ← {…}"; B11/B12 applicable; narrow rate bands | a stolen service credential used from a member host is not a who violation | R5 |
| Bursty batch jobs (backup, bulk export) | bursts of one source on one node | a burst is one piece of evidence (§6.5.4), its mass still shapes volume bands; GPD tails for sizes | first run of a new job is novel | pack O automation |
| Month-end, quarter-end, annual jobs | `ctx.mend`, `ctx.dom`, dormant patterns | split on calendar context when it pays; dormant memory revives recurring patterns (§6.8.1) | patterns recurring less than once in 400 d are novel each time (INFO unless sensitive) | R7 |
| Holidays, make-up workdays, long national holidays | calendar `dayclass`, normal-day flag | stale timers and drift acceptance count normal days only; holiday events scored against the non-workday density | a holiday on-call schedule is learned only after it recurs | R8, pack O days 5–6 |
| New system (cold start) | no tree / family match | inherit the family tree; else XS tree, novelty at system tier and B-library only | an unmatched new system has no who/when/content facts for its first days | R9 |
| New IP, new employee | IP absent from summaries | covered at once by group / prefix / system patterns; `unknown_ip` at who-closed sensitive nodes is MEDIUM; P11 groups it after 10 evidence units | a new employee's first sensitive action raises one MEDIUM | pack O DEV, portal |
| Schema change (a field disappears or appears) | coverage collapse with normal volume; new name | `attribute_gone` → splits on it collapse, scoring on it stops; `attribute_new` → registered, typed, probed | none | R10, PG7 |
| Scanners, crawlers, fuzzers, random-path floods | novel at p ≤ 1e-4, high novel rate per IP | outlier damping (§6.9.3), action dictionary cap, B08 novelty and P03 novel findings | – | R11 |
| Sensor sampling, packet loss | `extra.sample_rate`, gaps | mass scaled, evidence unchanged; statements say "观测到的请求" | rare events may be missed entirely | R12 |
| Load-balanced clusters, branch servers of one application | family signature | one family tree, `net.dst` split only where members differ (§6.20) | members with small real differences are described jointly until the difference pays | R13 |
| HTTP/2, gRPC, WebSocket, binary bodies | parse failures, content types | per-request events when the adapter decodes them; unparseable bodies profiled by length, charset and template | no key-level facts for opaque bodies | unit tests |
| Late or out-of-order logs | event ts < tick start | scored in the arrival tick; learned if still within the D + 1 retention; windows use event time | rows later than D are scored but not learned | unit tests |
| Legitimate reorganisation (new schedule, renamed accounts, moved routes) | coordinated drift, rebinding, route novelty with retirement | acceptance path (§6.9.2), rebinding (§6.12), revision / stale / retire (§6.6, §6.8) | a single-IP change takes 5 normal days to be accepted | pack O D1–D5 |

### 6.22 Worked trace of the requirement's example (design reasoning; estimates)

This trace follows the OA system of pack O (§11.3) to show how the mechanisms combine; the
day counts are estimates from the defaults and are what PG1/PG2 snapshots measure.

- **Days 1–2.** P02 registers the OA attributes as they appear (≈ 60: net.*, http.*,
  body.kv.username / password (shape) / captcha (shape) / csrf (shape), client.stack, ctx.*);
  P05's first run uses the bootstrap seeds. The root of the OA txn tree splits on
  `http.route` within hours: routes differ in method, size, keys and time, so a
  route-level target saves many bits per event and rule (V) passes after a few dozen events.
- **Days 2–6.** The `POST /login` node is a learning leaf. Its candidates include
  `net.src@/24`, `net.src@/16`, `ctx.tod_min@slot`, a size bin and `client.stack`. The
  username target is highly predictable per /24 (GA's 192.168.1.0/24 → {jack, rose},
  10.168.7.0/24 → {mike}, 192.168.2.0/24 → FIN's three names, 192.168.3.0/24 → 20 sales
  names) and so is the login minute; (V) passes for `/24` after a few workdays, and value
  grouping keeps the GA /24s apart because their username distributions differ.
- **Day ≈ 7.** P11's daily run groups the IPs by their (system, action) signatures: GA
  (OA approvals and reports, mail), FIN (finance), SALES (CRM), DEV (code, pool). Level
  `grp` becomes available; EFDT revision at the login node finds that the `grp` split
  predicts usernames and minutes better than `/24` (GA's two /24s unite), and replaces the
  /24 split once the new children are confirmed.
- **Days 8–14.** The node "POST /login ∧ grp = GA" is confirmed (n_c ≥ 20 at 3 units per
  workday, ≥ 3 dates). Its who summary is closed at /32 with the three IPs; P09's minute
  reservoir gives 09:00–09:21; P06 gives the 1–2 KB band (the "all within 0.5–3 KB"
  statement appears at n_rng ≥ 30, about ten workdays, with its bound ≈ 6 %, tightening
  every day); P07 finds `username` required with the grammar `[a-z]{4}`; P08 confirms the
  three bindings after about five logins each with the leave-one-out prior.
- **Other GA actions.** `POST /approval/{id}/approve` is its own route node whose who
  closes on {192.168.1.21} alone; P10 mines login → approval list → approve with its delay
  band; `POST /report/generate` closes on {.23, .121} with the 17:00–17:10 window and
  20–60 KB, and after about 15 sessions `GET /report/form` becomes its required
  predecessor.
- **Views.** The OA system view renders the statements of §6.17.2; the GA group view lists
  OA (login, documents, approvals, reports) and mail, and — because the finance approval
  node is who-closed on 192.168.2.10 and GA has zero mass there — the negative statement
  "在财务系统中从未执行写操作", which is what makes A1 a who violation.
- **The same machinery on the portal.** Its login node never closes: hundreds of IPs with
  U far above 0.05, so no who constraint exists and who is rendered as regions; P05 finds
  CR(net.src) ≈ 0 at /32 and removes IP as a split candidate; P12 picks who = reg or none;
  P08 finds no binding worth keeping; the per-IP intensity target carries A7.

---

## 7. Complexity, memory and budgets

### 7.1 Defaults and why

| Constant | Default | Reason |
|---|---|---|
| H_s, H_m, H_l | 1 d, 7 d, 30 d | one daily cycle; one weekly cycle (weekday/weekend); a month (month-end, stable identity). Shape follows H_m; confidence uses H_l with reset on accepted change (§6.9.4). |
| Confidence plateau | rate × H_l / ln 2 ≈ rate × 43 d evidence units | the price of forgetting changes no detector flags. Examples (pack O, burst-damped): GA login ≈ 2.1 units/day → ≈ 93 units, U_min ≈ 0.005; finance approval ≈ 2.6 units/day → ≈ 114, U_min ≈ 0.004; a single-IP weekly pattern ≈ 0.14 units/day → ≈ 6 units, never who-closed (U ≈ 0.07) and never confirmed on its own — it is described by its parent node. Raising H_l to 90 d triples every plateau (config). |
| D_max | 8 | who, where (2 route levels), when (2), content variant, exception, one spare; deeper trees did not appear in the design examples. |
| m_t targets per node | 8 | covers method/status/size/keys/value/stack/referrer-class/one more; beyond 8 the split statistics grow linearly while the marginal target contributes < 0.02 bits in the design examples (P05 re-ranks per node). |
| C candidates per learning leaf | 6 | who ×2 levels, when ×2, route or content ×2. |
| k_v, k_b | 8 (+other), 9 | value groups of practical interest are small (departments, variants); the `other` branch keeps generality. |
| τ0, δ, τ_tie | 10 bits, 1e-4, 0.05 bits/unit | τ0: an anytime-valid e-value threshold, false split ≤ 2^−10 · H(C_ever) per leaf under continuous monitoring (§6.5.5); δ as VFDT practice, made time-uniform; τ_tie is the VFDT tie-break scale. |
| n_child_min, n_conf, n_g | 5, 20, 32 evidence units | n_conf on the confidence channel: a department of 3 users with one daily login reaches 20 units in ~7 workdays. |
| R_learn | 14 d or 2 000 units | a leaf that has not split restarts its statistics rather than accumulate across regimes. |
| L_max, R_max, alternates | 64, 16, 4 per tree (at tier M; scaled by tier) | split statistics are the largest per-node block (≈ 21 KB); these caps bound them at ≈ 1.8 MB per tree. |
| N_max tiers | XS 32, S 256, M 1024, L 4096 nodes | per-tree memory ≈ 0.3 / 2.3 / 9.2 / 37 MB for the tree (§7.2); XS for idle and new systems. |
| A_max, A_ev, A_win, A_probe | 512 (128 at XS), 32, 96, 64 | registry bounded; per-event registry work bounded; window events bounded; hourly P05 work bounded. |
| e_rate, W_max, R_tick | 10 learned events/s per tree, 512 window events per tree and grain, 10⁶ rows per tick | ≈ 860 000 learned events a day per tree is ample for the multinomial models (rare strata are always learned in full); caps are lowered by P15 by budget share; scoring is sampled only at ladder step 7. |
| k_x, φ_x | 16, 0.02 | exception candidates are the heavy IPs of a node only. |
| Q_pairs, k_bx, T_conc | 16, 64, 1 h | binding candidates per system; heavy keys per pair; concurrency window for credential reuse. |
| S_max, G_max | 50 000 IPs (org), 256 groups | P11 signatures; systems in prefix/none mode do not count; materialised group views. |
| session / evidence / intensity / held caps | min(65 536, 4 × sources in 7 d) / same / k_int 16 384 / H_max 65 536 per tree | bounded per-source transient state, sized to the population actually seen, allocated by P15. |
| V_len | 64 characters | longer values are kept as shape and length only. |
| Family thresholds | weighted Jaccard ≥ 0.6 on 2 days; detach at ≥ 20 % of the family's nodes for 7 d | sharing must be the rule, not the exception, inside a family. |
| pcore_cpu_share, mem_mb_total | 0.25 core, 2048 MB | budgets independent of tick length (§6.19). |
| severities | table §6.16.3 | calibrated against the PG6 FAR budget on p_day; mirrors B08's tier-based discrete severities. |

### 7.2 Memory

Per node (packed float32 arrays, key strings interned per tree):

```
M_node ≈ h + s_who + s_when + s_rate + Σ_{t ∈ targets} s_t
       h ≈ 0.4 KB (header, context, drift) ; s_who ≈ 1.3 KB (5 levels × SpaceSaving(8) with mass + evidence, + HLL p=6)
       s_when ≈ 0.8 KB (2 × 96 float32; +2.5 KB when a minute reservoir is requested)
       s_rate ≈ 1.2 KB ; s_cat ≈ 0.5 KB ; s_num ≈ 1.3 KB ; s_text ≈ 0.7 KB ; s_set ≈ 0.6 KB
M_node ≈ 3.7 KB + 8 × 0.8 KB ≈ 10 KB (estimate; Python overhead to be measured, PG4)
M_split ≈ C (k_v+1)(m_t+2) k_b × 4 B + C (m_t+2) accumulators + slot sketches + covariances ≈ 21 KB
```

Per tree (a system, or a family of systems; both event kinds share N_max, 75 % txn / 25 % win):

```
M_tree ≈ N_max · M_node                         tree            (XS 0.3, S 2.6, M 10, L 41 MB)
       + (L_max + R_max + 4) · M_split          split stats     (1.8 MB at M; 0 at XS)
       + n_pairnodes · Q · k_bx · 64 B          bindings        (≤ 2 MB at 32 nodes)
       + A_max · 3 KB + 16 KB                   registry        (1.5 MB; 0.4 MB at XS)
       + 2 · R_p · ā · 16 B                     P05 probes      (≈ 2.6 MB at ā = 20 attrs/event; 0.16 MB at XS)
       + N_max · 0.5 KB                         fitted constraints (0.5 MB at M)
       + LRUs (sessions, evidence runs) + intensity + held     sized by P15 to the sources seen
                                                (≤ 65 536 · 0.17 KB + 16 384 · 40 B + 65 536 · 0.2 KB ≈ 25 MB worst)
       + M_batch = (Δt + D) · min(rate, e_rate) · ≈ 0.2 KB   retained compacted batches (≤ 9 MB at 900-s ticks)
M_tick = rows in the current tick · ≈ 0.4 KB    transient, before compaction; rows ≤ R_tick
```

Estimates: tier M ≈ 20–35 MB per tree in steady state (worst case ≈ 55 MB with full
LRUs, held events and batches at the e_rate cap); tier XS ≈ 1 MB. Independent of the
number of IPs (every per-source structure is an LRU or a heavy-hitter sketch with a cap
sized to the sources seen) and of the number of metrics (A_max, A_win, m_t). Across
servers: Σ over *active trees* of their tier, where trees ≤ families + unfamilied systems
and idle trees are checkpointed out (§6.20); e.g. 1 000 servers in 40 families plus 200
lightly used singletons ≈ 40 × 30 MB + 200 × 1 MB ≈ 1.4 GB (estimate), inside the 2 GB
default. Org-wide P11: ≤ S_max · 0.8 KB = 40 MB. The transient tick batch M_tick is
≈ 120 MB at 5 000 events/s with 60-s ticks; at 900-s aggregated ticks it is bounded by
R_tick (P00 subsamples `ev_sample` rows per record, keeping ≥ 1 row per record, HT mass).
For comparison, the current library measures 9.4–11.7 MB **per entity**
(integration.md §9.2): 1 000 IPs would need ≈ 10 GB.

### 7.3 CPU

```
c_event = c_route·D + c_score·(m_t + 3)                                   every event (P03)
        + π_e · [ c_upd·(2 + ρ̄(D − 2))·(m_t + 2)                           learned events (P04)
                + 𝟙(learning leaf) · c_split·C·(m_t + 2)
                + c_reg·A_ev + c_sig + c_dfg ]                             (P02, P11, P10)
```

Estimates (pure Python with numpy packed arrays; PG4 replaces them with measurements):
scoring 15–40 µs per event, learning 40–120 µs per learned event, P00 extraction 10–30 µs
per event plus body parsing (bounded by K_BODY = 64 keys and BODY_CAP = 4 KB). Periodic
work per tree, only when the tree received learned events since its last run and only on
dirty nodes (§6.20): P05 hourly O((n_kept + A_probe) · levels · m_t · R_p) numpy work
(≈ 50–300 ms est. at tier M), P06–P08 hourly O(dirty nodes) fits (≈ 5–60 ms est.), P09
6-hourly O(dirty nodes · 256²) worst case (≈ 0.2–3 s est., numpy), P11 daily O(n_IP · k)
(≈ 1–5 s at 50 000 IPs est.), P12 daily O(#systems · k) for signatures and families.
On pack O at 900-s ticks the P-core needs ≈ 1–2 s per tick (≈ 10–30 events/s × 900 s ×
(30 µs + π · 80 µs), est.), i.e. well under 1 % of one core on average.

Cost is proportional to **events** (and to evidence that arrived, for periodic work),
never to IPs × metrics or to the number of servers: the scoring term covers every event,
the learning term is capped by e_rate per tree, and periodic work is bounded by dirty
nodes, A_max, R_p and S_max.

### 7.4 Worked examples (estimates, to be replaced by PG4 measurements)

| Deployment | Events | IPs | Metrics | Estimated P-core CPU | Estimated P-core memory |
|---|---|---|---|---|---|
| Org pack O (§11): 6 systems, tiers S–M | ≈ 30 events/s peak | ≈ 600 (portal 500) | ≈ 120 attributes/system | ≈ 1–4 % of one core at peak | ≈ 60–150 MB |
| Org pack O-scale: portal with 20 000 IPs | ≈ 200 events/s peak | 20 000 | ≈ 120 | ≈ 3–5 % of one core | portal ≈ 35 MB (prefix mode), + P11 ≈ 16 MB |
| Org pack O-servers: 300 systems in 12 families + 30 singletons (§11.6) | ≈ 300 events/s | 5 000 | ≈ 120 | ≈ 5–10 % of one core | ≈ 12 × 30 + 30 × 1–10 ≈ 0.4–0.7 GB |
| Large site: 200 systems (20 L, 60 M, 120 S) | 5 000 events/s | 100 000 | 500/system | ≈ 0.5 core (scoring 5 000 × 30 µs + learning ≤ 2 000 × 80 µs) | ≈ 1.2 + 1.9 + 1.2 ≈ 4.3 GB before P15 lowers tiers to the 2 GB default budget; shard by tree across processes beyond one core |

---

## 8. Engine specifications

Each card: file · layer / registry slot / cadence · purpose · reads · writes · algorithm ·
budget (estimate) · unit tests. All engines honour `config['progressive']['enabled']`
(default False until M5 of §13.3) and strict mode.

### P00 — EventBuilderEngine (`raw.event`)
- **File** `backend/app/engines/raw/event_builder.py`; lib `engines/behavior/lib/pevent.py`,
  `lib/pparse.py`.
- **Layer / slot / cadence** raw / after R3 client_stack / every tick.
- **Purpose** Observations → per-system `EventBatch` with the open attribute map; body,
  query, header parsing; `ev_sample` expansion; value policy; learning sample (§6.2.1–6.2.2).
- **Reads** observations; `model.template@(s,__system__)` (R2, read-only via m_template);
  `model.attr` (parse hints); `model.sysprof.chosen` (content engines off → skip body
  parsing); `model.budget` (e_rate).
- **Writes** `evt.batch` (batch series); health: events, learned, parse errors, approx share.
- **Budget** 10–30 µs per event + parsing (est.).
- **Tests** (a) form body `username=jack&password=x&captcha=1234` → `body.keys`
  = {captcha, password, username}, `body.kv.username` = jack, password shape-only;
  (b) JSON body with nested arrays flattens to `items[].name`; (c) an aggregated record with
  count 40 and 8 `ev_sample` rows gives 8 events of weight 5 with their own sizes;
  without `ev_sample` one event of weight 40 flagged approx; (d) a new `extra['meta']` key
  appears as an attribute with no code change; (e) threshold sampling keeps every event of a
  stratum with c_k ≤ τ and HT-weighted totals equal the true totals within 1 % over 100 seeds;
  (f) pseudo entities and active probes are skipped; (g) a request from a `trusted_proxies`
  address with `x-forwarded-for: 10.1.2.3, 192.168.0.9` (the latter trusted) gets
  `net.src` = 10.1.2.3 and `net.peer_src` = the proxy; (h) a 40-character random value under
  an unknown key and a 200-character comment are kept as shape (+ length), `username=jack`
  in clear; (i) `extra.sample_rate` = 10 multiplies mass by 10 and leaves `pi` and the
  learning sample unchanged; (j) a tick with more than R_tick rows keeps ≥ 1 row per record
  and HT totals within 1 %.

### P01 — EventContextEngine (`derived.event_context`)
- **File** `backend/app/engines/derived/event_context.py`.
- **Layer / slot / cadence** derived / after D2 session / every tick.
- **Purpose** local time and day type, sessions (id, position, previous route, think time),
  window events from fresh raw/derived metrics (§6.2.3).
- **Reads** `evt.batch`; `model.seq` gap (lib/m_seq); config tz/calendar; `store.snapshot`
  for active IPs at H decision ticks; `model.attrsel` (kept `win` targets).
- **Writes** `evt.ctx` (aligned), `evt.win`.
- **Budget** ≤ 5 µs per event + O(A_win) per window event, ≤ W_max window events per grain (est.).
- **Tests** gaps across tick boundaries keep one session; a 60-s vs 900-s run of the same
  events gives identical session ids; a metric name first written on day 3 appears in `evt.win`
  on day 3; DST day types match lib/timebins; `ctx.dayclass` separates holiday, weekend and
  make-up workday; two users behind one IP with different `sess.key` get different sessions;
  with 10 000 active IPs in a grain, `evt.win` holds W_max events, every flagged or earned IP
  among them, and HT-weighted means within 5 % of the full population's.

### P02 — AttributeRegistryEngine (`behavior.attr_registry`)
- **File** `backend/app/engines/behavior/attr_registry.py`; lib `lib/pregistry.py`, `lib/phier.py`.
- **Slot / cadence** behaviour, first P engine after B21 / every tick (sampled updates).
- **Purpose** schema inference, per-attribute statistics, hierarchy models (§5.3, §5.4, §6.3).
- **Reads** `evt.batch`, `evt.win`, `evt.ctx` (learning sample); `model.attrsel`.
- **Writes** `model.attr`; events `attribute_new`, `attribute_gone`.
- **Budget** ≤ A_ev updates per learned event; daily hierarchy refresh ≤ 50 ms per system (est.).
- **Tests** type inference on synthetic columns (ip, numeric-log, ordinal status, set, kv text →
  `parse_as`); bin refresh only above the JSD threshold; cap A_max and overflow admission;
  a header that stops arriving at normal volume is declared `gone` within one normal day,
  one that only dips on a holiday is not; statistics ignore current-tick rows (t − D only).

### P03 — ConformityEngine (`behavior.conformity`)
- **File** `backend/app/engines/behavior/conformity.py`; lib `lib/pscore.py`.
- **Slot / cadence** after P02, before P04 / every tick.
- **Purpose** route every event, typed p-values with back-off and reference, per-IP tick
  scores, discrete `pattern_violation` events (§6.16).
- **Reads** `evt.batch`, `evt.win`, `evt.ctx`; `model.ptree` (as of t−1), `model.pbounds`,
  `model.pgrammar`, `model.pbind`, `model.pwin`, `model.pflow`, `model.who_groups`, `model.sysprof`.
- **Writes** `pat.assign`; `behavior.score/pm[conf_who, conf_when, conf_content, conf_seq,
  conf_novel]`, `behavior.axes` at (s, ip); events `pattern_violation`.
- **Budget** 15–40 µs per event (est.).
- **Tests** (a) finance approval node closed on {192.168.2.10} with n_c = 200 (confidence
  channel): an event from 192.168.1.23 (group GA) gets p_who = U ≤ 0.005 and a HIGH-candidate
  violation (σ = 2.5, write, `outsider_group`); from a never-seen IP, `unknown_ip` and MEDIUM;
  from a never-seen IP that submits the bound username of an IP silent for 2 days,
  `readdress_candidate` and LOW; the same while the bound IP is active within 1 h,
  `concurrent_use` and MEDIUM; (b) open node (portal login) → p_who
  NaN; (c) cross binding (192.168.1.21, rose) → credential axis; (d) HDR p_when for 03:05
  against a 09:00–09:21 window learned from ≥ 60 evidence units ≤ 1e-3; (e) a report
  submission without GET /report/form in the session, after 15 consistent sessions →
  required-predecessor violation with p_req ≤ 0.05; (f) prequential: an event is scored
  with the model before P04 learns it (the same event learned first would change the
  score); (g) Šidák per tick; (h) an IP with 500 clean events a day at one node emits no more
  LOW content findings than one with 5 (p_day), over 1 000 simulated clean days ≤ 0.2 %;
  (i) intensity: with 200 000 IP-node pairs in an hour and k_int = 16 384, an IP with 400
  events is tracked and scored on its guaranteed count, and no IP with ≤ N/k_int events is
  ever flagged.

### P04 — PatternTreeEngine (`behavior.pattern_tree`)
- **File** `backend/app/engines/behavior/pattern_tree.py`; lib `lib/pnode.py`, `lib/ptree.py`,
  `lib/pmdl.py`, `lib/psketch.py`.
- **Slot / cadence** after P03 / every tick (learns the batch of t − D).
- **Purpose** counting, evidence, splits, value grouping, prune/merge/revise, exceptions,
  pair statistics, lifecycle, drift, reference snapshots, trust gating, held events
  (§6.5–6.9).
- **Reads** `evt.*` batches of t − D, `pat.assign`, `model.attrsel`, `model.pwant`,
  `model.pbounds/pgrammar/pbind/pwin` (for confirmation and the reference snapshot),
  `behavior.trust`/`behavior.quarantine` via the B28 published set, `model.control`,
  `model.budget`, `model.who_groups`, `model.pwin` root windows (time level ℓ2).
- **Writes** `model.ptree` (daily checkpoint `ptree`); events `pattern_confirmed`,
  `pattern_retired`, `pattern_replaced`, `pattern_drift`, `pattern_absent`, `pattern_revived`.
- **Budget** 40–120 µs per learned event; split check O(C²) per 32 units (est.).
- **Tests** (a) e-value evidence: 3 IPs × 1 login/day with distinct time windows and a 4th
  population → the who split appears within 10 workdays and not before (V) holds; (b) a
  random attribute with no relation to targets, monitored every 32 units over 10⁵ events:
  false split rate ≤ 0.5 % over 2 000 simulated leaves (bound 2^−10 · H(C_ever)); (b′) the same
  with the targets duplicated three times (perfectly dependent targets): the false split
  rate does not increase (averaged e-values), whereas the summed G_c alone would cross τ
  (the review's counter-example); (b″) an aggregated row of mass 40 and a row with HT weight
  1 000 each contribute one evidence unit at most; (c) empirical-Bernstein tie-break chooses deterministically between
  equal candidates; (d) value grouping yields one group of 3 IPs + other; (e) prune after the
  children converge; (f) EFDT revision replaces a /24 split by `grp` once groups exist;
  (g) exception for an IP whose sizes differ, none for IPs that only differ in bound username;
  (h) quarantined IP's events are held, released on `model.control` release, discarded on
  reject; (i) memory stays under the tier cap at 10⁴ random IPs; (j) confidence channel: a
  node with 3 units/workday reaches U ≤ 0.01 after ≈ 50 units, and after an accepted
  window change its n_c restarts from the H_m state; (k) outlier damping: one IP sending
  p ≤ 1e-4 values for 5 days moves a confirmed band by less than a tenth of what an
  undamped learner moves it; (l) a split attribute declared `gone` collapses its split at the
  next daily check; (m) a month-end pattern retired after 30 d is revived as dormant on its
  next occurrence without a novel finding.

### P05 — AttributeSelectionEngine (`behavior.attr_select`)
- **File** `backend/app/engines/behavior/attr_select.py`; lib `lib/pselect.py`.
- **Slot / cadence** after P02 / period 1 h (entity_due per system).
- **Purpose** roles, kept sets, per-node overrides, redundancy (§6.4).
- **Reads** `evt.*` (probe reservoir), `model.attr`, `model.ptree` (node entropies).
- **Writes** `model.attrsel`; events `attribute_role`.
- **Tests** with 300 synthetic attributes (60 informative, 200 noise, 40 constant): informative
  kept ≥ 90 %, noise dropped ≥ 95 %, constants → invariants 100 %, a redundant copy of an
  informative attribute becomes `redundant`; `net.src` removed from split candidates on a
  random-IP population; with one stratum 1 000× heavier than another, the rare stratum
  keeps ≥ 32 probe rows (stratified probe); an hourly run evaluates n_kept + 64 attributes and
  every attribute at least once in 24 runs.

### P06 — ContentBoundsEngine (`behavior.content_bounds`)
- **File** `backend/app/engines/behavior/content_bounds.py`; lib `lib/pbounds.py` (uses lib/evt).
- **Cadence** 1 h. **Purpose** §6.10. **Reads** `model.ptree`. **Writes** `model.pbounds`,
  `model.pwant.p06` (none by default).
- **Tests** mixture 90 % U(1, 2) KB + 10 % in [0.5, 1) ∪ (2, 3] KB, n = 200 → band rendered
  "1–2 KB" with coverage in [0.88, 0.95], range "0.5–3 KB" with n_rng = 200 and bound
  2/201; the bound uses the observations of the ring's days, not n_eff; GPD p for 12 KB ≤ 1e-4;
  approx share 0.5 → no hard bound; after an accepted change the range restarts with the new
  segment.

### P07 — PayloadGrammarEngine (`behavior.payload_grammar`)
- **File** `backend/app/engines/behavior/payload_grammar.py`; lib `lib/pgrammar.py`.
- **Cadence** 1 h. **Purpose** §6.11. **Reads** `model.ptree`, `model.attr`. **Writes** `model.pgrammar`.
- **Tests** {jack, rose, mike} → `[a-z]{4}` and closed set; adding `mike.w` → `[a-z]{4}(\.[a-z])?`;
  `admin' OR '1'='1` flagged `injection_shape`; required key `username` from 100 % presence;
  lengths 3–8 give `[a-z]{3,8}`, never a pre-set `{1,10}`; an operator-pinned bound 10 widens
  the rendered range to `{3,10}`.

### P08 — BindingEngine (`behavior.binding`)
- **File** `backend/app/engines/behavior/binding.py`; lib `lib/pfd.py`.
- **Cadence** 1 h (screening and fit). **Purpose** §6.12. **Reads** `model.ptree`, P05 probe via
  `model.attrsel` (probe summary), `model.sysprof`. **Writes** `model.pbind`, `model.pwant.pairs`;
  events `binding_changed`.
- **Tests** three IPs with fixed usernames → FD holds, g3 = 0, one-to-one, and with 5 pure
  logins each LB_x ≈ 0.88 under the leave-one-out prior (6 logins: ≥ 0.9); portal with random
  usernames → no FD (g3 > 0.5); rename after 5 trusted events over 2 normal days → rebinding;
  the same from a quarantined IP → no rebinding; a shared terminal alternating jack/rose →
  set binding {jack, rose} and `lucy` outside it flagged; `svc_backup` from 3 servers →
  reverse set binding; `shared:<ip>` gets no per-IP binding.

### P09 — TimeWindowEngine (`behavior.time_window`)
- **File** `backend/app/engines/behavior/time_window.py`; lib `lib/pwindows.py`.
- **Cadence** 6 h. **Purpose** §6.13. **Reads** `model.ptree`. **Writes** `model.pwin` (node and
  system-root windows), `model.pwant.p09` (minute reservoirs).
- **Tests** arrivals U(09:00, 09:21) on 20 workdays → window 09:00–09:21 ± 1 min, coverage ≥ 0.95;
  a window crossing midnight is found after unwrapping; weekends learned separately.

### P10 — WorkflowEngine (`behavior.workflow`)
- **File** `backend/app/engines/behavior/workflow.py`; lib `lib/pdfg.py`.
- **Cadence** every tick (counting), 6 h (mining). **Purpose** §6.14. **Reads** `pat.assign`,
  `evt.ctx`, `model.who_groups`, `model.pwin`, `model.sysprof`. **Writes** `model.pflow`.
- **Tests** login → approval list → approve (1–6 min) mined with dep ≥ 0.8; after 15 sessions
  POST /report/generate requires GET /report/form (Jeffreys 5 % quantile ≥ 0.85); a session with
  the submission only → required-predecessor violation with p_req ≤ 0.05; NAT-style interleaved
  sessions reduce dependency (documented limitation) unless `sess.key` separates them; DFG
  counts use only learned trusted rows of t − D (a quarantined IP's sessions add nothing);
  an evicted action id is never reused.

### P11 — WhoGroupsEngine (`behavior.who_groups`)
- **File** `backend/app/engines/behavior/who_groups.py`; lib `lib/pminhash.py`, `lib/plouvain.py`.
- **Cadence** every tick (signature updates), 24 h (clustering; entity_due phase).
- **Purpose** §6.15. **Reads** `pat.assign`, `evt.batch` learning sample, `model.sysprof`
  (mode per system), config `who_group_names`, `ip_classes`, `dhcp_scopes`. **Writes**
  `model.who_groups`; events `group_formed`, `group_changed`.
- **Tests** 4 departments (3, 3, 20, 60 IPs) with distinct pattern sets → ARI ≥ 0.9; the configured
  name 综合部 attaches to the 3-IP group; ids stable across 10 runs with unchanged membership;
  prefix covers for a /24 department are the /24; a `readdress_candidate` joins the old IP's
  group provisionally and is confirmed or released by the next daily run; an imported
  DHCP-scope name attaches to the matching group.

### P12 — SystemProfileEngine (`behavior.system_profile`)
- **File** `backend/app/engines/behavior/system_profile.py`; lib `lib/pstrategy.py`.
- **Cadence** 24 h per system (entity_due). **Purpose** §6.18. **Reads** `model.attr`,
  `model.attrsel`, `model.ptree` (per-level who code lengths), fitter gains in `model.pbounds`/
  `pgrammar`/`pbind`/`pflow`, engine health, B17 events, config. **Writes** `model.sysprof`,
  `model.sysfam` (§6.20); events `strategy_changed`, `family_changed`.
- **Tests** synthetic systems: departmental OA → ip/grp + P07/P08/P10 on; random-IP portal →
  prefix/none, P08 off; opaque TLS → P07/P08 off; SNAT proxy without `trusted_proxies` → who
  none; hysteresis prevents flapping under noisy utilities; 3 OA replicas behind a load
  balancer form one family within 2 days, a fourth system with different routes does not; a
  member whose users differ gets a `net.dst` split, and detaches only when its branch holds
  ≥ 20 % of the family's nodes for 7 d; the who-level code of a DHCP pool prefers prefix/grp.

### P13 — FacetsEngine (`behavior.facets`)
- **File** `backend/app/engines/behavior/facets.py`; lib `lib/pfacets.py`.
- **Cadence** 2 h per subject / on read. **Purpose** §6.17.1. **Reads** `model.facets` and every
  source a declaration names. **Writes** facet trees in `profile.extra.facets` of (s, ip) for
  earned IPs (others on read), (s, class:*), (`__org__`, class:grp:*) for the G_max groups,
  (s, `__system__`); profile versions.
- **Tests** a new facet declared at runtime by a test engine appears in the next composition;
  inapplicable facets (content on an opaque system) are omitted.

### P14 — ViewsEngine (`behavior.views`)
- **File** `backend/app/engines/behavior/views.py`; lib `lib/prender.py` (zh/en templates).
- **Cadence** 2 h per key / on read. **Purpose** §6.17.2–6.17.3. **Reads** all P models,
  `model.who_groups`. **Writes** `model.pviews`, profile versions.
- **Tests** golden rendering of the OA example (zh and en) from a fixed model fixture; negative
  statement "从未在财务系统执行写操作"; numbers rounded per §6.10.

### P15 — ResourceGovernorEngine (`behavior.resource_governor`)
- **File** `backend/app/engines/behavior/resource_governor.py`.
- **Slot / cadence** first behaviour engine (before B01) / every tick.
- **Purpose** §6.19 and the active / earned sets of §10.
- **Reads** engine health, `store.memory_report()` (every 15 min), P-core `nbytes`, `model.sysprof`,
  sets published by B13/B27/B28 (§10.1), earned-gain records (§10.2). **Writes** `model.budget`,
  `ops.budget`.
- **Tests** over-budget for 3 ticks triggers exactly one ladder step; recovery after 24 h; the
  active set of an idle org with 10⁴ known IPs and 10 active is 10 plus open-state IPs; 300
  systems of which 280 are idle get tier XS and the idle ones are checkpointed after 30 d;
  ladder step 7 never samples write, sensitive, who-closed, new-IP or flagged-IP events.

---

## 9. Integration with the existing library

### 9.1 Registry order (`pipeline/build.py::build_registry(progressive=...)`)

```
raw        l2l3, l4flow, http, tls, dns, probe, R2 action_token, R3 client_stack, P00 event_builder
derived    D0 aggregation, periodicity, trend, D1 ratio, entropy, graph, D2 session, P01 event_context
behaviour  P15 resource_governor,
           B01 … B18, B19 … B21 (as today),
           P02 attr_registry, P05 attr_select (1 h), P03 conformity, P04 pattern_tree,
           P06 content_bounds, P07 payload_grammar, P08 binding, P09 time_window,
           P10 workflow, P11 who_groups, P12 system_profile,
           B23 feedback, B24 calibration, B25 fusion, B26 risk, B27 incident, B28 governor,
           B29 explain, P13 facets, P14 views, B30 portrait
signature  rule_match, correlation
```

Same-tick producer-before-consumer holds; the allowed one-tick lags are P03 reading the
tree of t−1 (by design, prequential), P04 reading trust/quarantine of t−1 (the lib-3 learner
rule), P15 reading engine costs of t−1, and every reader of the daily or hourly models
(`model.who_groups`, `model.sysprof`, `model.sysfam`, `model.attrsel`) seeing the latest
finished run. `progressive=False` (default until M5) registers
no P engine; the conf_* detector columns then stay NaN, which leaves the tick-mode golden
test untouched (it compares the first 31 detector columns only).

### 9.2 Detector registry, families, axes (`lib/detectors.py`, `lib/stages.py`)

- Append (append-only contract) `conf_who`, `conf_when`, `conf_content`, `conf_seq`,
  `conf_novel`: owner P03, kind `inst`, stream T, family `conformity` (appended to FAMILIES),
  `FAMILY_DEFAULT_AXES['conformity'] = ['content']`; P03 refines axes per event type as in the
  table of §6.16.3.
- New axis `content` mapped to stage `behavior` in lib/stages; the other axes P03 uses
  (privilege, lateral, credential, temporal, sequence, categorical, exfil, volume) exist.
- `pattern_violation` is a discrete kind for B25 corroboration ("a discrete finding ≥ HIGH"),
  B26 stage weights, B27 opening, B29 explanation, B23 policies and the eval harness — the same
  treatment `first_access_system` received in round 4.

### 9.3 Per-engine integration (functional; scalability changes are in §10)

| Engine | Integration |
|---|---|
| B02 peer_group | Optional (`progressive.groups_as_classes`): P11 groups become classes `class:grp:<g>` next to role/static/pool classes; ablation decides whether B02's role descriptor adds a pattern-set Jaccard term. |
| B08 novelty | Keeps SNI / dport / qtype / peer novelty tiers. Route novelty is shared with P03 `novel`: both use dedupe key `novel\|<s>\|<ip>\|<route template>` so B27 opens one incident. |
| B10 sequence | Keeps per-IP PPM for earned IPs (§10.2); P10 supplies group-level sequence evidence for every IP. |
| B13 budget | Per-IP intensity of IP-agnostic systems comes from P03 `rate.ip_h`; B13 keeps exfil / breadth budgets for earned and exfil-capable IPs. |
| B17 entity_link | Candidate IPs for a new IP come from P11's LSH neighbours (O(k), not O(#IPs)). |
| B18 class_monitor | With `groups_as_classes`, P11 groups get class aggregates, class_novel and class_rhythm. |
| B21 cross_system | Home class = the IP's P11 group when present. |
| B23 feedback | Labels on `pattern_violation`: `fp` with scope pattern → P04 adds the IP (or its group, scope `group`) to the node's who allow-set (a synthetic n_conf of who mass flagged `allowed_by_label`, removable), or accepts the new binding value; `tp` → node sensitivity +1 and the value is kept in a node deny-list; `expected_change` → P04 accepts the evolving attribute immediately. Policies live in `model.feedback` as today with scope `pattern:<pid>`. |
| B24 calibration | Rings for conf_* per (key, detector, stratum), T-stream strata; P03's Good–Turing / HDR p is the model p (`pm`) for small-sample blending. Keys follow §10.2 (class rings for unearned IPs). |
| B25 fusion | Family `conformity` in wHMP; volume-only cap applies when a content violation's axes ⊆ {volume}. |
| B26 risk | Stage weights for `pattern_violation` by type and severity (who HIGH-candidate 15, MEDIUM 8, LOW 3; binding MEDIUM 10; content MEDIUM 8; novel MEDIUM 8), like the first_seen tiers. |
| B27 incident | Opens on `pattern_violation` ≥ MEDIUM like other discrete findings; one system-level incident when ≥ 3 IPs violate the same who-closed node within 24 h (campaign on one pattern). |
| B28 governor | `model.control` release / reject apply to P04's held events; publishes the quarantined and low-trust set (§10.1) that P04 and P15 read. |
| B29 explain | For conformity-driven incidents the top reasons are the violated statements with observed vs expected in natural units; the counterfactual replaces the violating attribute by the node's modal value (or a bound value, or an in-window time) and recomputes p_t with the pure scoring function `lib/pscore` against the archived models, then the decision chain as today; hit@3 ranks `pattern:<pid>:<attr>` features. |
| B30 portrait | Embeds P13's facet tree (IPs, classes, groups); an unearned IP's portrait is its group view plus its exceptions and bindings. |
| lib-4 (optional) | With `progressive.lib4_inputs`, P03 writes per (s, ip) and tick the derived metrics `pat.violations.<type>` (counts) and `pat.conformity` (min p_ev), so signatures can compose them (e.g. who-violation then bulk download). |

### 9.4 API and UI (owner W-P7)

**Implemented (2026-10-01)** — `backend/app/api/routes_v3.py` (HTTP layer), `progressive_views.py`
(read projections), `progressive_runtime.py` (`APPMON_PROGRESSIVE=decision|only|full` builds the Runtime
on an organisation pack, default O, `APPMON_PROGRESSIVE_DAYS` of 900-s warm-up, then live 60-s ticks),
`frontend/js/progressive.js` ("画像模式" tab, zh/en), tests `tests/api/test_routes_v3.py`. Endpoints as
specified below, plus: `GET /api/v3/status`, `GET /api/v3/systems`, `GET /api/v3/systems/{s}/precision`
(per local day: confirmed patterns, mean depth, splits, drift, violations from the store; the evaluation
runs' recall / precision against the generator truth as a separate, labelled curve), `GET
/api/v3/systems/{s}/lattice` (bounded BFS), `GET /api/v3/violations` (typed, bilingual reasons and flags),
`GET /api/v3/{systems/{s},groups/{g},systems/{s}/entities/{ip}}/facets` (P13), and the IP view at
`/api/v3/systems/{s}/entities/{ip}/view`. Rendering on read reuses P14's `system_view` / `group_view` /
`ip_view` and P13's composers; the lattice reads P14's route index for each node's action. Deviation:
the group name POST persists `{name, ips (, cidrs)}` into both the runtime config and the pipeline's
config copy; it takes effect at P11's next daily run (no immediate rename).

Original specification:

New read-only endpoints in `backend/app/api/routes_v3.py` (same serialisation helpers):
`GET /api/v3/systems/{s}/view` (system view), `GET /api/v3/groups`,
`GET /api/v3/groups/{g}/view`, `GET /api/v3/systems/{s}/entities/{ip}/view`,
`GET /api/v3/patterns/{pid}` (node, constraints, statements, lineage, versions),
`GET /api/v3/systems/{s}/attributes` (registry and roles), `GET /api/v3/systems/{s}/strategy`
(P12), `GET /api/v3/budget` (P15). Writes: `POST /api/v3/groups/{g}/name` (operator name,
persisted into config `who_group_names`). Frontend: a new "画像模式" view with the two views,
a pattern detail panel (constraints, lineage timeline, violations), and an attribute registry
table.

---

## 10. Library-wide scalability: making B01–B30 resource-bounded

Measured today (integration.md §9.2, pack A seed 0, 43 entities): lib-3 ≈ 662 ms per live tick
(≈ 15 ms per entity-tick), p95 ≈ 1.1 s, memory 9.4–11.7 MB per entity. Every per-entity engine
loops over its entities each tick and keeps per-entity models, rings and checkpoints, so cost
and memory are O(#IPs). The change is `config['lib3']['resource_mode'] ∈ {'full', 'bounded'}`
(default `full` until M7 of §13.3; `bounded` is required for the O-scale packs).

### 10.1 Active-set processing

Per tick every engine iterates only over `A_t(s)` from `model.budget` (§6.19). The sets the
active set needs are published by their owners instead of being discovered by scans:

- B28 → `model.quarantined@(s,__system__)` = {ip: (since, regime)} and the IPs with trust < 1;
  default trust for an IP without a row is 1 (trust_prov 1).
- B13 → `model.acc_nonzero@(s,__system__)` = IPs with non-zero budget accumulators; B25 likewise
  for non-zero evidence CUSUMs; B26 → `model.risk_nonzero` (risk is decayed lazily at read time
  from `(value, ts)`).
- B27 → open incidents (existing `store.incidents(status=open)` index).
- The orchestrator writes a per-system tick clock `ops.tick@(s,__system__)` (one float per tick).

### 10.2 Per-entity models only where earned

B04 maintains a *shadow* record for **candidate** IPs only: per feature the H_m moments of the
IP's standardized residual under its class predictive (52 × 3 float32, ≈ 0.6 KB per IP).
Candidates are the heavy hitters of a per-system SpaceSaving over active IP-ticks (k_sh
slots allocated by P15, default 2 048, i.e. ≈ 1.2 MB per system) plus forced IPs; an IP that
is active on few ticks cannot earn a per-IP model anyway (n_rows ≥ 48 below), so the cap
loses nothing that could be earned. (The first draft kept an LRU of 100 000 IPs per system,
≈ 60 MB per system — a per-IP structure growing with the population.) The shadow moments are
shrunk towards the class (N(0, 1) residuals) with a pseudo-count of 8 rows per feature, so the
first rows of a new candidate do not produce a large negative gain from an unstable variance.
The earned gain is the prequential log-likelihood of the IP's committed rows under the shadow
model minus under the class predictive:

```
g_ip = Σ_rows Σ_f [ log N(z_f; μ_ip,f, σ²_ip,f) − log N(z_f; 0, 1) ]      (H_m-decayed; z_f class residuals)
earned(ip) ⇔ n_rows ≥ 48 and g_ip / n_rows ≥ τ_earn = 2 bits per row
           or forced: criticality 'high' (config), open incident, P04-distinctive (exception or
              binding), automation index ≥ 0.6 (machine clients are few and distinct),
              B15-identifiable in systems where identification is required
E_t(s)  = top-E_max(s) earned IPs by (g_ip / n_rows) × criticality weight      (E_max from P15/P12)
```

Hysteresis: demotion after 3 daily checks below τ_earn/2; a demoted IP's models are checkpointed
(`store.put_checkpoint`) so re-promotion restores them, and freed after 30 d. Promotion starts
per-entity learning from the promotion time (warm-started from the class model).

### 10.3 Change list

| Component | Today | Bounded mode | Cost after |
|---|---|---|---|
| store | per-entity rings 8 d for every entity | batch series (§5.6); `ops.tick`; per-entity long rings (8 d) only for earned IPs, 1 d for unearned; retention rule keyed by the earned set | memory O(earned · 8 d + active · 1 d) |
| R2 action_token | zero-fills `act.events` for every entity seen in 30 d, every tick | zero-fill only for earned IPs; D0/D2 read idle ticks from `ops.tick` (missing point = 0) through `derived/fresh.CLOCK` | O(active + earned) |
| D0 aggregation / periodicity / trend | every entity with a clock | entities with activity within the window span (6 h / 12 h) | O(recently active) |
| D2 session | per entity with stream points | unchanged (already activity-driven) | O(active) |
| B01 feature_vector | rows for all clocked entities incl. idle | rows for A_t; idle decision rows only for earned IPs (their baselines need zeros); unearned idle rows are implicit (`feature.active` = 0) | O(active) |
| B02 peer_group | refit on all eligible entities | refit on earned IPs + a stratified sample of ≤ 512 unearned descriptors per system; unearned assigned lazily to the nearest medoid when active | O(E_max + 512) per refit |
| B03 baseline | per-entity two-anchor baselines | earned only; unearned rows commit into the class aggregate (B18 `model.classagg`) | O(earned) models |
| B04 likelihood | entity → class → system back-off | unearned scored at the class tier; maintains the shadow record (§10.2) | O(active) |
| B05 common_mode | active | unchanged | O(active) |
| B06 multivariate | per-entity robust covariance | earned only; unearned scored with the class covariance (fitted on pooled class rows) | O(earned) fits |
| B07 rhythm | per-entity slot models | earned only; unearned timing via P09 windows and B18 class_rhythm | O(earned) |
| B08 novelty | entity, class, system tiers for all | entity tier for earned; class/system tiers for all active | O(active) |
| B09 client_identity | per-entity stack model | LRU ≤ 50 000 IPs per system (small states) | bounded |
| B10 sequence | per-entity PPM (largest cost today) | earned only; P10 group workflows for everyone else; class tier unchanged | O(earned) |
| B11 timing, B12 beacon | every active entity | prefilter: earned, or periodicity_score ≥ 0.5, or automation index ≥ 0.4 | O(candidates) |
| B13 budget | per-entity budgets | earned + exfil-capable IPs of the tick (upload-dominant or external destination), LRU ≤ 50 000; class budgets otherwise; publishes `acc_nonzero` | O(active) |
| B14 changepoint | per entity | earned only; B18 covers classes | O(earned) |
| B15 identity_model | all identifiable personas | earned ∩ identifiable, ≤ 32 per system | O(32) |
| B16 attribution | candidates = entities | candidates = P11 group prototypes + earned IPs (O(#groups + E_max)) | bounded |
| B17 entity_link | new IP vs all | new IP vs P11 LSH neighbours | O(k) |
| B18 class_monitor | per class | unchanged (+ P11 groups if enabled) | O(#classes) |
| B21 cross_system | per active IP | unchanged; home class from P11 | O(active) |
| B24 calibration | rings per entity | per-entity rings for earned; class rings (`classkeys`) for unearned | O(earned + classes) rings |
| B25 fusion | per entity meta rings and CUSUMs | earned: as today; unearned: class meta rings, CUSUM state in an LRU flushed after 7 idle days; publishes non-zero set | O(active) |
| B26 risk | per entity each tick | lazily decayed `(value, ts)`; iterate active ∪ non-zero | O(active) |
| B27 incident | findings and open incidents | unchanged | O(findings) |
| B28 governor | per entity | active ∪ quarantined ∪ open; publishes quarantined / low-trust sets | O(active) |
| B29 explain | per incident, in tick | unchanged; asynchronous under ladder step 6 | per incident |
| B30 portrait | every IP and class every 2 h | earned IPs, classes and groups every 2 h; unearned IPs on read from P13/P14 | O(earned + classes) per 2 h |

After the change, per-tick cost ≈ `|A_t| · c_light + |E_t| · c_full + Σ_events c_event`, with
c_light (B01 row, class-tier B04, class-ring B24, B25, B26, B28) and c_full (today's ≈ 15 ms
per entity-tick) to be measured in PG4; memory ≈ `|E| · 10 MB + |A_1d| · m_light + P-core`.

### 10.4 What bounded mode costs in accuracy (to be measured, PG9)

Unearned IPs lose per-IP rhythm, PPM, covariance and rings; their deviations are judged
against their class/group and the pattern tree. The design bet — stated so PG9 can refute it —
is that an IP whose own model does not beat its class by 2 bits per row gains nothing from that
model (by definition of g_ip, its rows are as well predicted by the class), and that the IPs that
matter for identity scenarios (twins, T9, T19) are earned or forced. PG9 compares packs A–E in
`full` and `bounded` modes.

Measured (§16.9 A5, seed 0): the bet holds for the feature models but NOT for the decision
chain's calibration rings: with B24 / B25 rings pooled per class for unearned IPs, pack E's T5'
(an unearned IP) fell from CRITICAL to LOW and pack A gained a MEDIUM false alarm — members'
score nulls differ even where their feature models do not. The chain therefore keeps per-IP
rings for active IPs and releases the per-IP chain state of an IP idle for 7 days (P15), i.e.
O(|E_t| + |A_7d|) state; class pooling is available as `lib3.pool_unearned` (off).

---

## 11. Generator extension: an organisation that reproduces the requirement

Owner W-P8. The generator's truth for the progressive core is the *persona program* itself:
the same declarative spec produces the traffic and `gen.pattern_truth`, so recovery can be
scored constraint by constraint.

### 11.1 Code structure and compatibility

- New module `backend/app/pipeline/orggen.py`: the `OrgSpec` DSL (§11.2), program personas,
  DHCP leases, per-event body rendering, truth export. `TrafficGenerator` delegates to it when
  `pack.org` is set; the existing population can run in the same generator (used by PG9).
- `extra['ev_sample']` in `_aggregate` behind the pack flag `ev_sample` (default False, so packs
  A–E, smoke, mini and the golden stay bit-identical); when on, the sample is drawn from its own
  RNG stream `rng_for(seed, pack, 'evs', key)` so the other streams are unchanged.
- Bodies are rendered only for org systems with `visibility = 'clear'`; opaque systems emit TLS
  records (SNI, JA3, sizes) without `http_*` and without `extra.l7`.

### 11.2 `OrgSpec` DSL (dataclasses in orggen.py)

```python
ValueSpec(kind: 'bound' | 'choice' | 'regex' | 'hex' | 'digits' | 'decimal' | 'text' | 'const',
          params: dict)                               # 'bound': per-IP value map (e.g. username)
SizeSpec(components: [(weight, lo_bytes, hi_bytes)], clip: (lo, hi))   # body size; a pad field fits it
BodySpec(fmt: 'form' | 'json' | 'multipart' | 'none', fields: {key: ValueSpec}, size: SizeSpec)
Step(method, route_fmt, body: BodySpec | None, p: float = 1.0, repeat: (lo, hi) = (1, 1),
     think_s: (lo, hi), status: {code: p}, resp: SizeSpec)
WindowSpec(daytypes: {'workday'|'nonworkday'|'all'}, start: 'HH:MM', end: 'HH:MM',
           arrival: 'uniform' | 'normal', dow: set | None)
Activity(name, system, who: 'each' | [ips] | 'one_of' | 'pool', when: WindowSpec,
         steps: [Step], per_day: (lo, hi), new_session: bool)
Department(name, ips: [str] | None, pool: (cidr, n_personas, lease_h) | None,
           usernames: {ip: name} | 'random', stacks: {stack: p})
SystemSpec(id, addr: 'ip:port', visibility: 'clear' | 'tls_opaque', host)
Drift(day, kind: 'window' | 'rename' | 'route' | 'growth' | 'churn', params)
Anomaly(id, day, time, kind, entity, params, truth)   # truth: expected_types, axes, severity
AttrSchedule(day, system, where: 'headers' | 'meta', name, ValueSpec, by: 'const'|'dept'|'noise')
Perturbation(id, day, kind: 'snat_proxy' | 'nat_branch' | 'readdress' | 'shared_terminal' | 'service_account'
             | 'month_end' | 'holiday' | 'replica' | 'attr_gone' | 'scanner' | 'sampling' | 'branches', params)
OrgSpec(systems, departments, activities, drifts, anomalies, attr_schedule, portal_n, portal_regions,
        perturbations, families)
```

### 11.3 The organisation of pack O

Systems (addresses exactly as in the requirement where it names them):

| System | Address | Visibility | Used by |
|---|---|---|---|
| oa | 192.168.100.100:8080 | clear HTTP with bodies | all departments |
| finance | 192.168.100.110:8443 | clear (reverse-proxy log with bodies) | 财务部; automation backup |
| crm | 192.168.100.120:8080 | clear | 销售部 |
| code | 192.168.100.130:443 | opaque TLS | 研发 |
| mail | 192.168.100.140:443 | opaque TLS | all departments |
| portal | 192.168.100.150:80 | clear | public population |

Departments:

| Department | IPs | usernames (bound) |
|---|---|---|
| 综合部 GA | 192.168.1.21, 192.168.1.23, 10.168.7.121 | jack, rose, mike |
| 财务部 FIN | 192.168.2.10 (the only approver), 192.168.2.11, 192.168.2.12 | lucy, tom, kate |
| 销售部 SALES | 20 IPs 192.168.3.20–192.168.3.39 | 20 distinct `[a-z]{3,8}` names |
| 研发 DEV | pool 10.50.0.0/22, 60 personas, lease 24 h (a persona gets a new IP from the pool each day) | per persona |
| public | portal_n IPs (default 500; scale 5 000 / 20 000) in 10.60.0.0/16 (50 %), 10.61.0.0/16 (30 %), 172.16.0.0/16 (20 %); each IP active on Poisson(3) random days | random per login from 50 000 names (not bound) |
| automation | 192.168.9.9 monitor (GET /health every 60 s ± 1 s on oa, finance, portal); 192.168.9.5 backup (finance POST /backup/export at 01:00, 50–200 MB) | – |

Activities (Asia/Shanghai; workdays unless stated; calendar with one holiday on day 5 and a
make-up Saturday on day 6):

| Dept → system | Activity | Window | Steps and content |
|---|---|---|---|
| GA → oa | login | 09:00–09:21, each member once | POST /login, form {username: bound, password: hex 8–16 (shape policy), captcha: digits 4, csrf: hex 32, pad}, body 90 % U(1, 2) KB, 5 % U(0.5, 1) KB, 5 % U(2, 3) KB, clip [0.5, 3] KB; 302; then GET /home |
| GA → oa | approvals | 09:25–11:30, only 192.168.1.21, 3–8 per day | GET /approval/list → GET /approval/{id} → POST /approval/{id}/approve form {id, opinion: choice(同意, 退回, 同意，请尽快办理), sign: hex 32}, 0.8–1.5 KB; think 30–300 s |
| GA → oa | documents | 09:30–16:30, all | GET /docs, /docs/{id}; POST /docs/{id}/comment form {text 20–200 chars} |
| GA → oa | report | 17:00–17:10, 192.168.1.23 and 10.168.7.121 | new session: GET /report/form → POST /report/generate JSON {dept: 'GA', period, items[10–40]}, 20–60 KB |
| GA → mail | mail | 09:25–09:40 and 13:30–14:00 | TLS records to mail SNI, sizes by message |
| FIN → finance | login | 08:50–09:10, all three | POST /fin/login form {username: bound, password, otp: digits 6}, 0.6–1.2 KB |
| FIN → finance | approval | 10:00–11:30 and 15:00–16:00, only 192.168.2.10 | GET /fin/approval/list → POST /fin/approval/{id}/approve form {voucher: digits 8, amount: decimal 100–50 000, opinion, sign: hex 32}, 0.9–1.6 KB |
| FIN → finance | bookkeeping | 09:15–17:30, 192.168.2.11 and .12 | GET /fin/ledger, /fin/ledger/{id}; POST /fin/voucher/create JSON 2–8 KB |
| FIN → oa | login and documents | login 09:05–09:30, documents all day | as GA, no approvals, no report |
| SALES → oa | login and documents | login 08:30–09:30 (irregular), documents | as GA; weekly report on Friday 16:00–17:00 (POST /report/weekly) |
| SALES → crm | CRM | 09:00–18:00 | GET /crm/customer/{id}; POST /crm/visit form {customer, note text 20–400} |
| DEV → code | git over TLS | 10:00–22:00 | opaque TLS, bursty sizes |
| DEV → oa | sporadic login | random 09:30–11:00 on 40 % of days | login only |
| public → portal | visit | 07:00–23:00, all days, diurnal | GET / → POST /login {username random, password} 0.3–0.8 KB → GET /news/{id} ×1–10 → POST /comment {text 50–500} with p = 0.2 |

Runtime attribute schedule (PG7): day 8, oa headers gain `x-client-ver` (constant 5.2.1 for
GA/FIN; 5.1.9 or 5.2.1 for SALES — informative by department); day 10, `meta.waf.score`
(0–2 normally, ≥ 12 on A3); scale variants add 60 or 300 synthetic `meta.f###` attributes on
day 3 (20 % department- or route-dependent, 67 % noise, 13 % constant).

### 11.4 Drift (legitimate) and anomalies (malicious)

| Id | Day / time | Entity / scope | Change | Expected |
|---|---|---|---|---|
| D1 | day 12 → | GA → oa login | window moves to 08:30–08:51 (all three, coordinated) | accepted as evolving → new window; ≤ LOW meanwhile |
| D2 | day 13 → | 10.168.7.121 | username mike → mike.w | rebinding after 5 events over ≥ 2 d; `binding_changed` INFO |
| D3 | day 14 → | oa approvals | route /approval/{id}/approve → /flow/{id}/approve (list likewise) | new patterns confirmed; old stale |
| D4 | days 12–18 | portal | population × 1.5 in the same regions | no who violations (open population) |
| D5 | day 15 → | DEV pool | lease 12 h | no incidents |
| A1 | day 16 10:30 | 192.168.1.23 → finance | GET /fin/approval/list, POST /fin/approval/{id}/approve | who (outsider_group, system_new); ≥ MEDIUM, target HIGH |
| A2 | days 17–21 09:10 | 192.168.1.21 → oa | login with username=rose, daily (also the non-adoption check) | binding (cross_binding), credential; ≥ MEDIUM; binding for .21 still jack on day 21 |
| A3 | day 18 11:00 | 192.168.3.25 → oa | POST /login with username `admin' OR '1'='1` and a 12 KB body | content (grammar injection_shape, size); ≥ MEDIUM |
| A4 | day 18 03:05 | 10.168.7.121 → oa | login at 03:05 | when; ≥ MEDIUM (write at night) |
| A5 | day 19 17:02 | 192.168.1.23 → oa | POST /report/generate without GET /report/form in the session | seq (required predecessor); ≥ LOW |
| A6 | day 19 14:00 | 192.168.3.30 → oa | GET /admin/export (sensitive_patterns) | novel; ≥ MEDIUM |
| A7 | day 20 20:00–21:00 | 10.60.7.7 → portal | 400 POST /login, random usernames, 90 % 401 | content (`rate.ip_h`), credential; ≥ MEDIUM |
| A8 | day 20 (a workday) 10:15 | 192.168.2.99 → finance | approval as username=lucy from a never-seen FIN-subnet IP while lucy's IP 192.168.2.10 is approving in the same hour | who (`unknown_ip`) + reverse binding (`concurrent_use`); ≥ MEDIUM, target HIGH candidate |
| A9 | days 11–21, 10:30 daily | 192.168.3.33 (SALES) → finance | one GET /fin/approval/list a day, read-only, trying to become normal (low-and-slow poisoning) | who (`outsider_group`) ≥ LOW on day 11 **and still on day 21**; 192.168.3.33 never enters the node's heavy set; no habituation of the incident (B26–B28 rules) |
| A10 | day 21 09:00–12:00 | 50 IPs in 203.0.113.0/24 → portal | POST /comment spam (text 2 KB, links) | who at region level (pack O declares the three public ranges as `ip_classes`, so the portal's who closes at `reg`; in the red-team variant without them it closes at /16), content (length, grammar); ≥ LOW |

Pack O relaxes generator.md's "one scenario per entity per pack": an entity may carry two
anomalies when their windows are ≥ 24 h apart and their expected types differ (A1/A5 on
192.168.1.23, A4 on 10.168.7.121 after D2) — the truth rows stay unambiguous. A2's cross
binding (192.168.1.21 as rose at 09:10) also meets the concurrency test, since rose's IP
192.168.1.23 logs in at 08:30–08:51 after D1; both flags are expected.

**Pack O-real: real-world perturbations (PG11).** Each item is a switch on pack O (all on
in O-real; one at a time in the ablation runs) with its own truth rows:

| Id | Perturbation | Expected |
|---|---|---|
| R1 | from day 8, oa is reached through a reverse proxy 192.168.100.99 with SNAT; the proxy adds `x-forwarded-for`; variant R1′ without `trusted_proxies` configured | R1: PG1 on oa unchanged within 0.05 (who from XFF); R1′: `snat_suspect` within 1 day, oa who = none, no who violation against the proxy IP, a configuration hint in the view |
| R2 | a SALES branch of 10 personas behind one NAT address 192.168.30.1; session cookies present | `shared:192.168.30.1`; no per-IP binding or exception for it; per-session bindings (`sess.key` → username) for ≥ 8 of 10 |
| R3 | FIN member 192.168.2.11 is re-addressed to 192.168.2.51 by DHCP on day 9 | `readdress_candidate` (≤ LOW), provisional join to FIN, no incident ≥ MEDIUM; the old IP goes stale |
| R4 | a shared GA terminal 192.168.1.40 used alternately by two personas (jack2, rose2) | set binding {jack2, rose2}; a third name flagged LOW |
| R5 | service account `svc_sync` used by 3 servers 10.9.0.1–3 on finance every 10 min | reverse set binding, automation facet, no who violations for the three |
| R7 | a finance month-end close (POST /fin/close) on the last 2 workdays of the month (pack spans a month end) | a `ctx.mend` split or a dormant revival; no novel ≥ MEDIUM on the second occurrence |
| R8 | a 7-day holiday (days 22–28, pack extended to 35 d) with 5 % on-call activity | no `pattern_absent` ≥ LOW, no drift acceptance during the holiday, daily patterns intact on day 29 |
| R9 | a new oa replica 192.168.100.101:8080 appears on day 10 behind the same users | joins the oa family by day 12 and is scored against its nodes; no novel storm |
| R10 | the adapter stops sending `x-client-ver` on day 15 | `attribute_gone` within one normal day; splits on it collapse; no drift or violation storm |
| R11 | a scanner 10.60.99.9 requests 5 000 random paths on the portal on day 13 | novel findings for the scanner; tree node count on the portal grows < 5 %; no pattern learned from it by day 21 |
| R12 | 5 % of observations dropped at random and flows sampled 1:4 on the portal (`sample_rate`) | PG1 on the portal within 0.05 of the unperturbed run; statements marked "观测到的" |
| R13 | 12 branch servers of one CRM (crm-01 … crm-12), each used by its own 5-IP branch | one family; a `net.dst` split per branch where branch users differ; memory ≤ 2 × one CRM tree |

### 11.5 Truth published by the generator (read only by eval)

- `gen.pattern_truth`: one row per truth pattern `{tid, system, kind, route, method,
  who: {level: 'ip'|'grp'|'prefix'|'reg'|'any', value}, daytypes, windows: [(start, end)],
  content: {attr: {band90, range} | {grammar, required_keys, closed_values}},
  bindings: {attr: {ip: value}}, workflow: [(from_tid, to_tid, delay_band)], group,
  valid_from, valid_to}`; drifts close rows and open new ones.
  **Each step of an activity has its own arrival law and windows (round 3).** A row is one
  step k of an activity; its arrivals are the session start (drawn from the window spec's law:
  uniform, or N(mid, width/4) clipped) plus the think times of the records before it (optional
  steps skipped with 1 − p, repeats U{repeat}, every record but the session's first preceded by
  U(think_s) of its step) — `orggen.step_arrival_law`, simulated once per window spec (20 000
  sessions, fixed RNG). The row publishes `gen.arrival_q` = per day type a list of
  `[weight, quantile function at 129 equally spaced probabilities]` (weight = the spec's share
  of sessions × the step's mean arrivals per session) and `windows` = per spec the central 99 %
  of that law (step 0 of a non-repeated step: the activity's window itself). The scorer draws
  held-out minutes from `arrival_q` (pmetrics.holdout_events). Round 2 published the activity's
  window widened by the sums of the think-time bounds and drew minutes uniformly in it, so
  later records of a session (mail records 2–8, KS 0.23 against the generated arrivals) were
  checked in a tail the generator hardly reaches.
  **Held-out events follow the traffic mix (round 3).** The scorer draws the held-out events of
  a statement's context from the truth rows, day types and member sources in proportion to the
  events the program emitted up to the scored day (`PTruth.traffic`, from the opportunities
  bookkeeping); round 2 gave every row of a route an equal share (pack O's mail: four
  departments at 25 % each, while 销售部 and 研发 send ~90 % of the records).
  **Content truth is stated on the observed quantity, and a hard range on what the data could
  show (evaluator round 3).** A TLS row's `net.bytes_up` is the step's payload plus
  `orggen.TLS_UP_FRAMING` (300 B); the truth's band and range now include it (round 2 stated
  the bare payload, so every learned mail / git band was exactly truth + 300 B and failed PG1).
  The generator records the smallest / largest emitted value per (row, size attribute, date)
  (`ptruth.extremes`); `PTruth.observable(row, day)` cuts a row's `range` (the generator's
  SUPPORT) to `[max(lo, data min), min(hi, data max)]` over the row's lineage up to the scored
  day, keeping the support as `range_support`. PG1 content, PG10 and the example checklist
  accept a learned hard range that matches either (±25 % per endpoint): a tail the generator
  never drew cannot be learned (pack O seed 0 emits 45 综合部 logins and none of the 5 % below
  1 KB, so "100 % in 0.5–3 KB" was unrecoverable by any engine), while a learned range that
  misses an emitted extreme still fails (seed 1: a 643 B login on day 2 that the 综合部 node
  never saw). Precision and calibration still check the stated range against held-out events
  drawn from the support.
- `gen.group_truth`: departments → members over time (pool personas → their IPs by lease).
- `gen.strategy_truth`: per system the acceptable arms (e.g. portal who ∈ {prefix, reg, none},
  P08 off; mail P07/P08 off; oa who ∈ {ip, grp}, P07/P08/P10 on).
- `gen.truth` rows for A1–A10, D1–D5 and R1–R13 with `expected_types`, `expected_axes`,
  `required_severity` / `max_allowed_severity`, and for R-items the expected adaptation
  (who mode, family, set binding, attribute_gone …).
- `gen.system_truth`: families and members for O-real and O-servers.

### 11.6 Packs

| Pack | Timeline | Registry | Purpose |
|---|---|---|---|
| O | 21 d at 900 s, aggregated with `ev_sample`, portal_n = 500 | `progressive_decision` (§16.2 M25; was full + progressive) | PG1–PG3, PG5–PG8, PG10 |
| O-real | pack O with R1–R13 (35 d for R8), plus one-at-a-time ablations | `progressive_decision` | PG11 |
| O-red | red-team org: different department sizes, windows, body sizes, names, no `ip_classes`, plus 20 attributes independent of every truth constraint (false-split probe); never used for tuning | `progressive_decision` | PG1–PG3, PG6 reported separately; PG2 false splits |
| O-servers | 300 systems: 12 families × 20 replicas or branches + 30 singletons + 30 idle, 7 d at 900 s | `progressive_only` | PG4 server curves |
| O60 | days 1–7 at 900 s, day 8 at 60 s event mode | `progressive_decision` | aggregated vs event-mode agreement (PG1 at day 8 within 0.05) |
| O-scale-{500, 5k, 20k} × attrs {0, 60, 300} | 7 d at 900 s | `progressive_only` (P engines + B24–B28 bounded) | PG4 IP and metric curves |
| A–E (+P) | as today | full and bounded, progressive on | PG9 |

Runs longer than one evaluation slot (O-real, 35 d; O-scale-20k and O-servers-300 at 7 d)
are resumable (round 3, `eval/resumable.py`): runner.run_pack's tick loop with the whole loop
state pickled (cloudpickle; the engines' identity sentinels such as `pevent.ABSENT` by
reference, locks re-created, weak references with their referents) at the end of a local day
every `segment_s`; a process stops after a checkpoint (`stop_after_s`) and the next one resumes
it (`scripts/progressive_report.py --checkpoint DIR --segment-s S --stop-after T`,
`pscale.run_point_resumable`). A resumed run equals an uninterrupted one except the wall-clock
fields (P12 arm costs, P15 budget weights, engines' ms) — which differ between two
uninterrupted runs too (`tests/eval/test_resumable.py`).

---

## 12. Evaluation gates (progressive core)

Rules as in eval.md: strict mode; medians over 5 seeds with bootstrap 95 % CIs; the scoring
code never reads truth except to score; thresholds and severity tables are tuned on seeds 0–1
and reported on seeds 2–4; the org red-team variant O-red (different department sizes,
windows, body sizes and names, no `ip_classes`) is never used for tuning and is reported
separately; every gate that passes on O but fails on O-red is reported as not passed. Metrics live in
`backend/app/eval/pmetrics.py`; the runner snapshots `model.ptree`, the fitted models,
`model.who_groups`, `model.sysprof` and `model.pviews` at the end of days {1, 2, 3, 5, 7, 10,
14, 21}.

**PG1 — Pattern recovery.** A truth pattern T (valid at time t with ≥ 20 opportunities so far,
spread over ≥ 3 distinct dates — confirmation needs 3 dates, §6.8.1)
is recovered when a confirmed or stable learned node N, with its fitted constraints, satisfies:
same system; N's act_node route template and method equal T's; who-compatible (truth ip →
rendered IP set Jaccard ≥ 0.8; grp → rendered group members Jaccard ≥ 0.8; prefix/reg →
rendered prefixes cover ≥ 90 % of truth mass with ≤ 20 % outside; any → rendered open or ≥ 3
regions); when-compatible (window IoU ≥ 0.7 per day type); each truth content constraint
matched (band endpoints within ± 20 %, range within ± 25 %; learned grammar accepts ≥ 99 % of
1 000 fresh truth samples and rejects ≥ 99 % of 1 000 contrast samples; required keys equal;
closed sets equal); bindings equal for ≥ 95 % of truth pairs; each truth workflow edge present
with dep ≥ 0.8 and an overlapping delay band. Component recalls are reported separately (who,
when, content, bindings, workflow). Precision = share of rendered confirmed statements whose
every constraint holds on held-out generated events of their context at least at its stated
confidence (refinements of truth count as correct). **Targets at day 14:** recall ≥ 0.90
overall and ≥ 0.85 per component; precision ≥ 0.90; GA and FIN bindings 6/6.

**PG2 — Convergence and calibrated confidence.** Recall and precision at the snapshot days;
recall non-decreasing within 0.05 outside [day 12, day 15] (drift); time to 80 % recall ≤ 7 d
for daily patterns and ≤ 21 d (three occurrences) for weekly ones; mean depth of confirmed
nodes non-decreasing before day 12; **precision grows with time**: the median stated
confidence of statements matching unchanged truth patterns is non-decreasing across the
snapshot days (the confidence channel, §6.9.4) and the median unseen-IP mass U of who-closed
truth nodes is non-increasing; reliability of stated confidence: expected calibration error
≤ 0.05 over statements (hold rate of the constraint on held-out events vs stated
confidence); false splits (learned splits on attributes independent of every truth
constraint, identified on O-red where the generator adds 20 such attributes) reported per
system-month, expected ≤ 1.

**PG3 — Specificity reached.** GA login who = the 3 IPs (or the group whose members are those
3) with 3/3 bindings; finance approval who = {192.168.2.10}; portal login who ∈ {prefix, reg,
any}, no portal bindings, portal exceptions ≤ 1 % of portal IPs; DEV patterns who ∈ {grp,
10.50.0.0/22}, never single pool IPs; P11 ARI against static departments ≥ 0.9 and the DEV
pool one group or one prefix.

**PG4 — Resources sublinear in IPs and metrics (measured, replaces §7 estimates).** On
O-scale: log-log slope of P-core steady-state memory against N_portal ∈ {500, 5 000, 20 000}
≤ 0.15 without P11 and ≤ 0.35 with P11 in ip mode; CPU per event slope ≤ 0.1; against
attributes {≈ 40, 100, 340}: memory slope ≤ 0.2, CPU per event slope ≤ 0.2. **Servers**, on
O-servers: memory against the number of systems at a fixed number of families (20, 100, 300
systems in 12 families) log-log slope ≤ 0.3; an idle system costs ≤ 1 MB after 1 day and
0 MB in memory after 30 idle days (checkpointed); periodic CPU (P05–P10) of trees without
new evidence is 0; memory of the 12-branch CRM family ≤ 2 × a single CRM tree (R13).
Absolute: P-core ≤ 40 MB per tree at tier M; scoring p95 ≤ 100 µs per event, learning p95 ≤ 250 µs per learned
event (one core, Python). Bounded B-library: per-tick lib-3 cost slope against idle known IPs
≤ 0.1.

**PG5 — Drift adaptation latency.** D1 new window confirmed (IoU ≥ 0.7 with the new truth)
within 3 workdays, ≤ 1 GA incident ≥ LOW meanwhile, old window gone within 5 workdays; D2
rebinding within 5 events and 2 days; D3 new route patterns confirmed within 5 workdays, old
ones stale within 3 workdays; D4 and D5 no incident ≥ LOW. Non-adoption: on day 21 the binding
of 192.168.1.21 is still jack; portal `rate.ip_h` p99 within 10 % of its pre-A7 value.

**PG6 — Anomaly detection of the requirement's examples.** For each of A1–A10: a
`pattern_violation` of the expected type on the scenario entity within 1 tick of its first
anomalous event, and an incident ≥ the required severity within max(4 ticks, 1 h). Recall ≥ 0.95
over 5 seeds × 10 anomalies. FAR on clean org entities: incidents ≥ LOW ≤ 0.1 and ≥ MEDIUM
≤ 0.02 per entity-day; discrete `pattern_violation` ≥ LOW ≤ 0.05 per entity-day; KS D of
randomised `behavior.p[conf_*]` on clean ticks ≤ 0.05; B29's top reason is the violated
constraint for ≥ 90 % of TP incidents. A9 additionally requires a who violation on day 21
(no habituation) and 192.168.3.33 outside the finance approval-list node's heavy set on day 21.

**PG7 — Open schema.** New attributes registered in the first tick they appear; type correct
≥ 95 % (synthetic); role within 24 h; informative synthetic attributes kept ≥ 90 %, noise
dropped ≥ 95 %, constants → invariants 100 %; no engine or lib module mentions the synthetic
names (grep in the test). Truth types follow §5.3 for non-payload attributes; a string
value in a payload namespace (hdr.*, body.kv.*, q.kv.*) is `text` whatever its
cardinality (§5.4.5: the shape hierarchy, with P07's closed set carrying the categorical
constraint; §16.2 A2).

**PG8 — Scenario adaptation.** P12's chosen arms ∈ `gen.strategy_truth` for ≥ 95 % of (system,
seed) at day 14; ≤ 2 switches per system after day 7; the chosen who arm's utility (§6.18.2,
revised) within 5 % + 0.01 bits/event of the best arm's, measured offline on the same events
(`gen.act_log`: source, action, hour per system and date; exact groups, unbounded state;
`pmetrics.who_arm_utilities`). The who arms of `gen.strategy_truth` are the arms that
offline utility puts within that tolerance on the pack's own address plan (a test checks
the two agree), not a designer's guess: two levels that induce the same partition of a
system's sources (every department in its own /24s) are equally right.

**PG9 — Non-regression and bounded mode.** Packs A–E with the progressive core on (full mode):
gates 1–15 not worse than without it beyond the bootstrap CI. Bounded mode: gate 1 recall drop
≤ 0.03, gate 3 FAR not worse, gate 8 twins / T9 / T19 preserved (earned or forced), live lib-3
p95 at 35 entities ≤ 50 % of full mode.

**PG10 — Views.** Golden rendering tests for the OA example (zh and en) from a fixture; on pack
O at day 11 (before D1) the OA system view contains a statement matching the GA login truth
(3 IPs, window 09:00–09:21 ± 2 min, band 1–2 KB, a username grammar accepting jack, rose and
mike, 3 bindings; the range 0.5–3 KB rendered either as "observed range" or, when
n_rng ≥ 30, with its bound), and at day 21 the statement for the truth then valid (window
08:30–08:51 ± 2 min after D1, range 0.5–3 KB with its bound, a grammar accepting jack, rose,
mike.w and contained in `[a-z.]{1,10}`, bindings jack, rose, mike.w with 192.168.1.21 still
jack despite A2); the finance approval statement with the single IP; and the GA group view's
negative statement for finance writes.

**PG11 — Real-world robustness (pack O-real, §11.4).** Every R-item meets its expected
adaptation in the table of §11.4 on ≥ 4 of 5 seeds; with all R-items on, PG1 overall recall at
day 21 drops by ≤ 0.05 against pack O on the systems an R-item does not make unprofilable
(R1′'s oa who, R2's NAT users without session keys are excluded and reported), and PG6's FAR
budgets hold; each R-item's one-at-a-time ablation is reported.

---

## 13. Work breakdown and file ownership

### 13.1 Workstreams

| WS | Scope | Files owned (new unless marked) |
|---|---|---|
| W-P0 Foundation | sketches, MDL maths, event batch, hierarchies, node/tree layout, pure scoring, accessors, store batch series | `engines/behavior/lib/psketch.py`, `pmdl.py`, `pevent.py`, `phier.py`, `pnode.py`, `ptree.py`, `pscore.py`, `pevalue.py` (e-process accumulators of §6.5.5, with the simulation test of P04 (b)–(b″)), `m_ptree.py`; `core/store.py` (modified: batch series, `compact_batch`, retention rules, `ops.tick` helper); `lib/template.py` (modified, additive: read-only `Templater.apply_path`) |
| W-P1 Events | P00, P01, body/query parsing, who resolution, session key, capture contract | `engines/raw/event_builder.py`, `engines/behavior/lib/pparse.py`, `engines/derived/event_context.py`; `pipeline/orchestrator.py` (modified: `ops.tick`); `docs/lib3/contract.md` (section "capture extension") |
| W-P2 Registry and selection | P02, P05 | `engines/behavior/attr_registry.py`, `lib/pregistry.py`, `engines/behavior/attr_select.py`, `lib/pselect.py` |
| W-P3 Core (critical path) | P04, P03 | `engines/behavior/pattern_tree.py`, `engines/behavior/conformity.py` |
| W-P4 Content and time | P06–P09 | `content_bounds.py` + `lib/pbounds.py`, `payload_grammar.py` + `lib/pgrammar.py`, `binding.py` + `lib/pfd.py`, `time_window.py` + `lib/pwindows.py` |
| W-P5 Workflows and groups | P10, P11 | `workflow.py` + `lib/pdfg.py`, `who_groups.py` + `lib/pminhash.py` + `lib/plouvain.py` |
| W-P6 Adaptation, budgets, bounded mode | P12 (incl. system families, §6.20), P15, §10 coordination | `system_profile.py` + `lib/pstrategy.py` + `lib/pfamily.py`, `resource_governor.py`; §10 rows are applied by each B-engine's owner file (modified: `feature_vector.py`, `peer_group.py`, `baseline.py`, `likelihood.py`, `multivariate.py`, `rhythm.py`, `novelty.py`, `client_identity.py`, `sequence.py`, `timing.py`, `beacon.py`, `budget.py`, `changepoint.py`, `identity_model.py`, `attribution.py`, `entity_link.py`, `calibration.py`, `fusion.py`, `risk.py`, `governor.py`, `portrait.py`, `raw/action_token.py`, `derived/aggregation.py`, `derived/periodicity.py`, `derived/trend.py`, `derived/fresh.py`) |
| W-P7 Facets, views, API, UI | P13, P14, B30 embedding, routes, frontend | `facets.py` + `lib/pfacets.py`, `views.py` + `lib/prender.py`, `api/routes_v3.py`, `frontend/js/` pattern view (new file), `portrait.py` (modified: embed facets) |
| W-P8 Generator and evaluation | org generator, packs (O, O60, O-real, O-red, O-scale, O-servers), truth, metrics, report | `pipeline/orggen.py`; `pipeline/generator.py` (modified: `ev_sample` flag, org delegation); `eval/packs.py`, `eval/truth.py`, `eval/report.py`, `eval/runner.py` (modified: snapshots); `eval/pmetrics.py`; `scripts/evaluate.py` (modified: `--org`, `--scale`, `--real`, `--servers`) |
| W-P9 Integration | detectors, stages, registry, B23–B29 hooks, config defaults, docs | `lib/detectors.py`, `lib/stages.py`, `pipeline/build.py`, `core/engine.py` (`DEFAULT_CONFIG['progressive']`, `['lib3']['resource_mode']`), `feedback.py`, `calibration.py`, `fusion.py`, `risk.py`, `incident.py`, `governor.py`, `explain.py` (hooks of §9.3); `scripts/import_who_names.py` (IPAM / DHCP-scope / asset CSV-JSON → `who_group_names`); docs: `docs/组织业务系统画像平台设计.md` §3.6 (Chinese, from §0), `docs/lib3/engines.md` (P cards), `docs/lib3/summary.md`, `docs/lib3/eval.md` (PG gates) |

Conflicts: `calibration.py`, `fusion.py`, `risk.py`, `governor.py` and `portrait.py` are touched by
two workstreams; W-P9 owns the functional hooks, W-P6 the bounded-mode rows, applied in that
order (M5 before M7).

### 13.2 Configuration keys (`core/engine.py::DEFAULT_CONFIG`)

```
'progressive': {
  'enabled': False,
  'value_policy': {...§5.1.3..., 'secret_globs': [...], 'v_len': 64},
  'trusted_proxies': [], 'client_ip_headers': ['x-forwarded-for', 'forwarded', 'x-real-ip'],
  'session_cookies': ['JSESSIONID', 'PHPSESSID', 'ASP.NET_SessionId', 'sid', 'session'],
  'type_hints': {'code': ['status', 'code', 'port', 'qtype', 'rcode', 'method']},
  'system_families': [],                   # [{name, systems | cidrs, force: bool}]
  'groups_as_classes': False, 'lib4_inputs': False,
  'budget': {'pcore_cpu_share': 0.25, 'lib3_cpu_share': None, 'mem_mb_total': 2048, 'per_system': {}},
  'defaults': {...§7.1 constants...}},
'who_group_names': [],                     # [{name, ips | cidrs}]; operator- or import-fed
'lib3': {'resource_mode': 'full'}          # 'bounded' per §10
```

### 13.3 Milestones

| M | Content | Workstreams | Exit criterion |
|---|---|---|---|
| M0 | Contracts frozen: lib signatures, store batch API, store names (§5.6), config keys, detector append | W-P0, W-P9 | contract tests; golden untouched |
| M1 | Events end to end: P00, P01, P02; generator `ev_sample`; pack O skeleton with truth | W-P1, W-P2, W-P8 | PG7 registration and typing parts |
| M2 | P04 + P03 with system-level roles (P05 stub) | W-P3 | P04 unit tests (a)–(m) incl. the e-value simulations; first PG1 who/route components |
| M3 | P05 full, P06–P09 | W-P2, W-P4 | PG1 content and when components |
| M4 | P10, P11, P12 (incl. families), P13, P14 | W-P5, W-P6, W-P7 | PG1 full, PG3, PG8, PG10 |
| M5 | Integration with B23–B30; `progressive.enabled` on for pack O; PG1–PG8 first report | W-P9 | PG6 measured |
| M6 | Packs A–E with the core on; severity table tuned (seeds 0–1); O-real and O-red | W-P9, W-P8 | PG9 full mode, PG11, O-red report |
| M7 | Bounded mode (§10), P15 ladder; O-scale and O-servers runs | W-P6 + B owners | PG4 (IPs, metrics, servers), PG9 bounded |
| M8 | Docs sync (design doc §3.6 in Chinese, engines.md, contract.md, summary.md, eval.md); report | W-P9 | gate table generated from the report |

Standing rules: engines couple only through the store; strict mode in tests and eval; the
per-engine unit suite stays under 20 s; implementers do not change git state (the lead
commits); no threshold is tuned to one seed; every accuracy or cost number quoted in the docs
comes from a report.

---

## 14. Decisions and risks

### 14.1 Alternatives considered and not adopted

- **Per-(IP × metric) models with class roll-up** (the current B-library shape): bottom-up, every
  IP needs a model before anything is learned about the group; cost O(#IPs × #metrics). The
  requirement asks for the opposite order.
- **Frequent itemsets / FP-growth over attribute values, or formal concept analysis over the
  whole generalisation lattice**: the number of itemsets / concepts is exponential in the
  attributes, supports need all data, and there is no content model. The pattern tree is a
  greedy, evidence-guided path through the same lattice with a node budget.
- **Offline clustering of all events** (k-means, HDBSCAN on event vectors): needs the full data,
  mixed types, unstable ids, no incremental update.
- **Plain Hoeffding-bound splits**: with a gain range of ≈ 25 bits they need ≈ 10⁴ events per
  decision (§6.5.5); small departments would never be specialised. Kept (empirical-Bernstein,
  time-uniform) as the stability test between candidates.
- **The summed prequential code-length saving (a ratio of two Bayes mixtures) as the
  significance test** (the first draft): not a test martingale under a composite null, double
  counts dependent targets, and charged multiplicity per evaluation instead of per candidate.
  Replaced by averaged per-target universal-inference e-values (§6.5.5); the summed saving is
  kept as the MDL gain and ranking statistic.
- **One fixed exponential decay for both shape and confidence**: confidence would plateau at
  rate × H/ln 2 forever (the finance approval node could never be closed at U ≤ 0.01).
  Replaced by the confidence channel with reset on accepted change (§6.9.4). A pure sliding
  window or ADWIN-only design was rejected too: it forgets stable facts as fast as it forgets
  changed ones.
- **One tree and fixed state per server**: linear in the number of servers, which the
  requirement forbids as much as linearity in users. Replaced by system families,
  activity-proportional tiers and dirty-node periodic work (§6.20).
- **Learning group names from traffic**: impossible — traffic shows who belongs together, not
  what the department is called. Names come from configuration or an inventory import.
- **Deep sequence or embedding models, isolation forests, autoencoders on events**: no calibrated
  tails, no bounded incremental memory, no per-constraint explanation (decisions.md reasons apply).
- **Full regular-language inference (RPNI, L\*)**: needs negative examples or queries; character-
  class anti-unification is enough for value shapes and is bounded.
- **Exact FD discovery (TANE, FUN)**: exponential in attributes; screened pairs with g3 suffice.
- **HMM or GMM for activity windows**: Bayesian Blocks is parameter-light and exactly optimal for
  piecewise-constant rates on bounded samples.
- **Leiden via an external dependency**: Louvain is implemented in-house; Leiden's connectivity
  guarantee is an upgrade path behind `lib/plouvain`.
- **Rollback of the pattern tree**: not cheaply replayable; replaced by delayed learning, trust
  gating, held events, multi-day confirmation and reference snapshots.
- **A/B running of every strategy arm**: most dimensions are full-information through prequential
  code lengths; a bandit is used only where an engine must actually run to be measured.

### 14.2 Risks

- **Greedy order dependence.** A pattern may be learned at a different depth or split order than
  the truth. Mitigated by EFDT revision, alternates, prune/merge; PG1 matches through act_nodes
  and statement consistency rather than tree shape.
- **Dependent evidence.** Events within sessions and automation streams are not independent;
  evidence units damp bursts (ω ≤ 1 per observed row, harmonic within a run) and
  who-closedness needs ≥ 5 normal days, but a source's behaviour on successive days is still
  one person's habit, so population statements about few sources remain optimistic. PG2's ECE
  and the O-red false-split probe measure how much.
- **Weak who evidence at small n.** `p_who ≥ 0.5/(n + 1)` by construction; a closed who-set gets
  strong only with time — the "longer is more precise" requirement, stated honestly. Discrete
  severities rely on sensitivity and corroboration, calibrated in PG6.
- **Circular validation.** Generator and algorithms are designed together. Mitigations: hold-out
  seeds, a red-team org variant never tuned on, and a pilot on real proxy/WAF logs with request
  bodies before any accuracy claim is made outside the eval report.
- **Python cost.** §7 figures are estimates and may be off by 2–3×. P15 degrades learning first;
  compiled kernels behind `lib/pnode` are an upgrade path that needs a dependency decision.
- **FAR interaction.** Five instantaneous detectors join the single-tick and evidence-CUSUM
  budgets; PG9 must show no regression, and B23 feedback weights the family.
- **Groups ↔ patterns loop.** P11 groups feed P04 splits, which feed P11 signatures; hysteresis,
  stable ids and daily cadence damp it; PG3/PG8 check stability.
- **Aggregated inputs without per-event rows.** Bands degrade to approx and hard bounds are
  withheld (§5.1.2); adapters should supply `ev_sample`.
- **Shared IPs (NAT, VDI, proxies).** Handled by `shared:<ip>`, `sess.key` and XFF resolution
  (§5.1.4, §6.21); without session keys or XFF the users behind one address are profiled as
  one class, and the view says so. PG11 (R1, R2) measures it.
- **Value retention.** Non-secret values are kept in clear by default because the requirement
  asks to see them; secrets are shape-only by key glob and by value randomness (§5.1.3). This
  supersedes the decisions.md line for the PPC and needs the lead's confirmation. Legal review
  is out of scope by instruction.
- **Plateaued confidence.** With H_l = 30 d, confidence stops growing after about two months of
  unchanged behaviour, and a single-source weekly pattern is never confirmed on its own (§7.1).
  Raising H_l is a configuration choice with a stated trade-off.
- **Family mistakes.** Two systems with similar routes but different users could be merged into
  one family; the `net.dst` split separates them as soon as that pays and detachment follows,
  but until then statements describe both. PG4/PG11 (R9, R13) check it; `system_families`
  config can forbid a merge.
- **Dormant memory.** Revival of a dormant pattern is keyed on its context only; an attacker who
  replays a month-end action out of season is still checked against the dormant constraints
  (who, content), not waved through.

---

## 15. References

- Agarwal, Cormode, Huang, Phillips, Wei, Yi. Mergeable summaries. PODS 2012.
- Duffield, Lund, Thorup. Priority sampling for estimation of arbitrary subset sums. J. ACM 2007.
- Howard, Ramdas, McAuliffe, Sekhon. Time-uniform, nonparametric, nonasymptotic confidence sequences. Ann. Stat. 2021.
- Ville. Étude critique de la notion de collectif. Gauthier-Villars 1939.
- Vovk, Wang. E-values: calibration, combination and applications. Ann. Stat. 2021.
- Wasserman, Ramdas, Balakrishnan. Universal inference. PNAS 2020.
- Bifet, Gavaldà. Learning from time-changing data with adaptive windowing (ADWIN). SDM 2007.
- Bifet, Gavaldà. Adaptive learning from evolving data streams (Hoeffding Adaptive Tree). IDA 2009.
- Blockeel, De Raedt, Ramon. Top-down induction of clustering trees. ICML 1998.
- Blondel, Guillaume, Lambiotte, Lefebvre. Fast unfolding of communities in large networks. J. Stat. Mech. 2008.
- Cormode, Korn, Muthukrishnan, Srivastava. Finding hierarchical heavy hitters in data streams. VLDB 2003; Mitzenmacher, Steinke, Thaler. Hierarchical heavy hitters with the Space Saving algorithm. ALENEX 2012.
- Cormode, Muthukrishnan. An improved data stream summary: the Count-Min sketch. J. Algorithms 2005.
- Cormode, Shkapenyuk, Srivastava, Xu. Forward decay: a practical time decay model for streaming systems. ICDE 2009.
- Dawid. Present position and potential developments: the prequential approach. JRSS A 1984; Grünwald. The Minimum Description Length Principle. MIT Press 2007.
- Domingos, Hulten. Mining high-speed data streams (VFDT). KDD 2000.
- Duffield, Lund, Thorup. Learn more, sample less: control of volume and variance in network measurement. IEEE Trans. IT 2005.
- Dunning, Ertl. Computing extremely accurate quantiles using t-digests. 2019.
- Efraimidis, Spirakis. Weighted random sampling with a reservoir. IPL 2006.
- He, Zhu, Zheng, Lyu. Drain: an online log parsing approach with fixed depth tree. ICWS 2017.
- Ioffe. Improved consistent sampling, weighted minhash and L1 sketching. ICDM 2010.
- Kivinen, Mannila. Approximate inference of functional dependencies from relations (g3). TCS 1995.
- Krichevsky, Trofimov. The performance of universal encoding. IEEE Trans. IT 1981.
- Manapragada, Webb, Salehi. Extremely Fast Decision Tree. KDD 2018.
- Maurer, Pontil. Empirical Bernstein bounds and sample variance penalization. COLT 2009.
- Metwally, Agrawal, El Abbadi. Efficient computation of frequent and top-k elements in data streams (Space-Saving). ICDT 2005.
- Raghavan, Albert, Kumara. Near linear time algorithm to detect community structures in large-scale networks. Phys. Rev. E 2007.
- Scargle, Norris, Jackson, Chiang. Studies in astronomical time series analysis VI: Bayesian Block representations. ApJ 2013.
- Teh, Jordan, Beal, Blei. Hierarchical Dirichlet processes. JASA 2006.
- Tran-Thanh, Chapman, Rogers, Jennings. Knapsack based optimal policies for budget-limited multi-armed bandits. AAAI 2012.
- Weijters, van der Aalst. Rediscovering workflow models from event-based data using Little Thumb (heuristics miner). ICAE 2003.
- Wilks. Determination of sample sizes for setting tolerance limits. Ann. Math. Stat. 1941.

---

## 16. Integration and measured results (2026-09-30)

Everything in this section is measured; nothing is an estimate. Runs: pack O (§11.3), 21 days at
900-s ticks, aggregated records with `ev_sample`, registry `progressive_decision` (P00–P15 +
B24–B29; the pack's default `full+progressive` did not fit in memory, §16.2 M25), strict mode,
seeds 0–2 (the doc's tuning seeds are 0–1; seed 2 is reported as held out). Scoring is
`backend/app/eval/pmetrics.py` only; the report is `reports/progressive/progressive_report.{json,html}`
(`scripts/progressive_report.py`), per-seed results `reports/progressive/runs/O_<seed>.json`,
scaling points `reports/progressive/scale/`. Five seeds, O-red, O-real and O60 were **not** run
in this round (wall-time budget); gates that require them are reported as not measured, and every
gate result below is therefore provisional in the sense of §12 (medians over 3 seeds, not 5).
**Round 2 (2026-10-01): §16.9 (adaptation, bounded chain, cost) and §16.10 (results on the
final code: 5 seeds, O-red, O60, deviations M26–M46, G1–G7, C1–C2, E1–E11) supersede the
numbers of §16.3–§16.8, which are kept as the round-1 record.**
**Round 3 (2026-10-02): §16.11 (results on the final code of round 3: 5 seeds, O-red, O60,
deviations R3-1 – R3-14, R1 – R5, G8 – G15, E-T1 – E-C2, V1 – V9; before / after on the same
scorer) supersedes §16.10's numbers, which are kept as the round-2 record.**

### 16.1 Registration

`pipeline/build.py::build_registry(progressive=...)` registers P00–P15 exactly in the §9.1
order when `registry_mode` is `full+progressive` (config `progressive.enabled`, or the pack's
`registry_mode`) and the §9.1 subset for `progressive_only`; the default (`full`) registry is
unchanged, so packs A–E, smoke and mini run the same engine set as before (§16.6).

### 16.2 Mechanism changes made during integration (deviations from §6)

Each change below was made because a measurement on pack O showed the specified
mechanism failing for a stated reason; each has a regression test (M1–M18, M22 in
`tests/engines/test_progressive_integration_fixes.py`; M19 in `tests/engines/test_p00_event_builder.py`;
M20, M21, M23 in `tests/engines/test_p03_conformity.py`; M24 in `tests/core/test_store_batches.py`;
M25 in `tests/test_progressive_integration.py`), and the tests of M19–M24 were checked to fail
without the change.

| # | Where | Specified (§) | Changed to | Why (measured) |
|---|---|---|---|---|
| M1 | P04 `pattern_tree._route_partition` | root splits only through rule (V) (§6.5.5) | **route-first multiway partition** of every transaction root: a route that recurred (≥ 3 rows on ≥ 2 local dates) gets its own child of the root at once; the partition is not a learned split (never pruned, merged or revised; its `other` child never learns and is never a pattern); route-family candidates are not re-offered below it | the root's (V) test had to discover each route one binary split at a time; at day 21 the OA tree still held routes pooled in `other` branches, PG1 recall 0.095 on seed 1 (baseline HEAD). A route is the requirement's "页面/路由" – an identifier of the action, not a hypothesis to test |
| M2 | `lib/pevalue.SplitStats` rule (V) | universal-inference e-value `2^(L0−L1)`, L0 = pooled ML code length (§6.5.5) | **blockwise k-sample e-process**: blocks are local days; at a block's start every slot's predictor is frozen (Bayes mixture of the slot's own smoothed counts and the leaf predictive, weight `1/(1+2^−S)` from the bits the slot saved so far); the block's outcomes are coded under their slot's predictor against the evidence-weighted mixture of all slot predictors (the RIPr onto "no dependence"). Validity: for any null θ0 the mixture's expected log-ratio is ≤ 0 twice by Jensen, so each block factor has expectation ≤ 1; Ville's inequality applies unchanged. `log2_e_ui` is kept for diagnosis | the pooled-ML form charges the null model's parametric regret (≈ (k_b−1)/2·log2 n bits per target) before a real dependence counts: the OA login node's /24 split that separates 综合部 / 财务部 / 销售部 had log2 e = 5 after 101 evidence units although usernames alone saved ≈ 45 bits. Null test: sup log2 e over 60 days ≥ 5 in ≤ 2⁻⁵ + 0.02 of 200 runs; power: a 3-IP department is found where the UI e-value is not |
| M3 | `SplitStats.retarget`, P04 `_retarget` | a changed target list restarts the leaf's learning episode (§6.5.5, §6.4) | the episode is **re-targeted**: kept targets keep their bins and statistics, the e-process wealth of dropped targets funds the new ones (a self-financing portfolio of e-processes: wealth is conserved, so the sum stays an e-process) | P05 revises targets daily for young nodes; each restart discarded up to a week of evidence and the login node never accumulated 2^14 |
| M4 | rule (G) | total MDL gain over all targets | **selective gain** Σ_t max(0, G_t) − T_c: a split is paid for by the targets it predicts; targets it does not predict cost their smoothing overhead only once | a split that predicts one target (username) was vetoed by ten unrelated targets' small negative savings |
| M5 | P04 `_do_split` | children start as `candidate` with target summaries seeded; everything else empty (§6.5.5) | children also receive their evidence units (≤ the leaf's), the local dates their values were seen on, and – for a split on `net.src` – the leaf's who summary and `net.src`-keyed binding pairs restricted to their addresses | a department's node (3 logins a workday) re-earned 20 units over 3 dates after the split: split on day ≈ 12, confirmed after day 20 |
| M6 | P04 prune | decayed-saving prune at any age (§6.6) | children younger than 7 days are not judged (`PRUNE_MIN_AGE`) | the two-part saving charges every child parameter in full: the 综合部/财务部/销售部 split was pruned 2.6 days after (V) had proven it |
| M7 | `psketch.BurstEvidence` | one run per burst (gap ≤ τ_burst) (§6.5.4) | a run also restarts after `run_max` = 1 h | a 60-s health check never pauses for τ_burst: its node held n_c = 7 after 21 days (1 440 rows a day) and was never confirmed |
| M8 | P04 drift | ADWIN on the node's log-loss incl. `@when`; structural alarm persists 14 days | `@when` excluded from the ADWIN loss (P09 owns time drift); ADWIN reset when the targets' hierarchy versions change; a structural alarm that never showed higher loss on ≥ T_persist + 2 normal days is cleared as a false alarm | numeric-bin refreshes of a monitor's duration target kept its node `evolving` (not a pattern) for most of pack O |
| M9 | P14 system view | one statement per node, who = union (§6.17) | a node shared by ≥ 2 learned groups (each ≥ 5 % of its mass, and more than the node's mass on other routes) is also stated **per group** ("某类人"), each part with its own arrival window from the node's minute reservoir restricted to the group's members; a part lists only the members seen at the node (no part when none is) | the requirement's system view "OA 服务器的某类人会在哪个时间段访问我什么页面": GET /docs by three departments is correctly ONE node (no target differs), so no split will ever name the groups |
| M10 | P14 group view | negative statement when a who-closed write node has no group mass | only when **no** write node of the system (closed or not) has group mass | "综合部 在 oa 中从未执行写操作" was rendered next to 综合部's own logins (its login node was not yet who-closed) |
| M11 | P14 bindings / invariants | every fitted binding rendered; TLS/DNS invariants by eTLD+1 | bindings rendered only when the bound attribute's registry cardinality ≥ 8; TLS/DNS keyed by `pdfg.host_key` (≤ 5 labels, digit runs templated) | attributes with a handful of values system-wide (body format, content type) were rendered as "bindings" – they are constants of the action, already stated as content; mail/code host names collapsed to one eTLD+1 route |
| M12 | P11 items | pattern items from every split context | time contexts only | address-split contexts (a /24 child) made the group items restate the address and degraded ARI 1.0 → 0.87 / 0.59 |
| M13 | P08 who levels | levels chosen by P12's who arm | level 0 (IP) always screened unless the arm is `none` | OA's arm `prefix` screened only /24 pairs: no IP → username binding could be fitted |
| M14 | P05 | closed small categoricals are shape-only / dropped by cardinality | a local field with registry cardinality ≤ 16 and coverage ≤ 25 % is a target | finance's `opinion` (同意/驳回/退回) was dropped as noise |
| M15 | `eval/pmetrics` | precision over all confirmed statements | statements whose context the generator cannot reproduce (non-judgeable) are reported as `n_unjudged`, not counted; group members may be prefixes | a statement whose pattern is also defined by an attribute the truth program does not label (a request-size bin, a client stack, a session position) cannot be checked on held-out events of its context; the grp-alt who check raised on /24 members |
| M16 | `pipeline/orggen` truth | JSON padding key `remark` always required | required only when no body of the step can exceed the capture cap `BODY_CAP` | P00 keeps the keys of the parsed prefix; the truth asked for `remark` on 20–60 KB reports whose tail is cut |
| M17 | P03 `group_outsider` | `outsider_group` = the IP's group key absent from the node's group level | the group's standing is the group-level evidence its OTHER members brought (≥ MEMBER_EV = 3 units); the IP's own earlier events do not count | the first damped event of a foreign source put its group key at the node, after which its events were learned undamped: A9's sales address was in the finance approval list's heavy set by day 21 (all 3 seeds), against §6.9.2 "persistence alone never makes a foreign source a member" |
| M18 | P03 damping | damp 0.1 only for p_ev ≤ 1e-4 or a who outsider | also for a value credibly bound to another source (`cross_binding` with LB ≥ 0.9) or in concurrent use there | A2's borrowed credential (192.168.1.21 logging in as rose, days 17–21) became .21's own binding ({jack, rose} on day 21, PG5 non-adoption failed on all seeds); a value bound nowhere (D2's rename) is not affected |
| M19 | P00 `event_builder` | per-event fields come from the record or its `ev_sample` row `l7` (§5.1.2) | an `ev_sample` row's `meta` is flattened into that row's event (`meta.*`); §5.1.2 lists `meta` | in aggregated mode every `meta.*` attribute (the WAF score, all synthetic O-scale attributes) was dropped: PG7 never saw `meta.waf.score`, and the first PG4 attribute axis (60 / 300 attributes) measured nothing (memory 22.0 MB at 40, 100 and 340 attributes; after the fix 22.0 / 31.4 / 36.5 MB) |
| M20 | P03 who member | member = in the closed level's heavy set (95 % of mass ∪ 95 % of evidence) at the covering node; standing (≥ `MEMBER_EV` = 3 units) only at back-off ancestors | standing makes a member at the covering node too | the finance approver (≈ 3 % of finance's evidence) was outside the root's heavy set: a who violation on every login (4 HIGH findings, 8 HIGH incidents on seed 0), damped learning, and its own approval nodes stalled at n_c ≈ 12 for 10 days (confirmed on day 19 instead of day 10) |
| M21 | P04 reference snapshot | `ref.who` = the heavy set (top 16) | heavy set ∪ sources with standing (≥ 3 units, ≤ 16 more) | P03's dual anchor (§6.8.3) flagged the same approver against the reference after M20 fixed the current summary |
| M22 | P03 group rule | a member group at a node whose group level is closed makes a new address a colleague; `outsider_group` when the group's evidence from other members < `MEMBER_EV` | the node is a pattern **of the group** only when ≥ 2 other members have standing there (`GROUP_MIN_MEMBERS`); otherwise colleague status does not apply and the IP is a group outsider | P11 put the finance approver into 综合部's group on days 15–17 (seed 0): A1 (192.168.1.23 approving in finance) was scored as the approver's colleague — LOW, no incident. One member's individual habit is not the group's pattern |
| M23 | P03 who p-value | U of the covering node's closed level | when the source is foreign at the covering node, the closed ancestors where it is also foreign are tested too: p = min(U_node, n · min U_ancestor) (Bonferroni over the n closed levels tested) | a young single-user node has U ≈ 1/n_c (finance approval on day 16: 0.029, above the MEDIUM bound 0.02) although the source never used the closed system (finance root U = 0.0013): A1 was LOW |
| M24 | `lib/template.apply_path` (P00's read-only route) | an unsettled segment (unseen, or seen ≤ 2 times) renders `{var}` | a word segment (letters, `-`, `_`, ≤ 32 chars) stays literal unless R2 has merged rare siblings into `{var}` at that position; R2 itself is unchanged | every new action rendered as the same route: `GET /admin/export` (A6) and the first `/flow/list` after D3 were `GET /{var}/{var}`, which P10's dictionary already held — a new action could never be novel |
| M25 | eval registry | pack O runs `full+progressive` | the measured runs use `progressive_decision` = R2, R3, P00, P01, P15, P02–P12, B24–B28, B29, P13, P14 (no R1, D0–D2, B01–B23, lib-4) | `full+progressive` on pack O: the B-library's per-entity state over ≈ 700 sources held a 2.26 GB store by day 4 (pickled; approx 1.27 GB, derived 0.62 GB, vectors 0.48 GB), 290 s per simulated day with two runs in parallel, and two attempts were OOM-killed at 7.1 GB per run; `progressive_decision` runs 21 days in ≈ 35 min at < 2 GB. This is the requirement's own point: enumerating every source is what exhausts resources |


Evaluation-harness corrections (scoring and truth only, no engine behaviour):

- `eval/pmetrics._attr_records` did not read P02's `records` key, so PG7 scored every attribute as
  unregistered (0.0) on every run.
- PG7 truth: `appears` is now the first time an attribute is **observable** (in aggregated mode an
  event's l7 / meta reaches the capture only through `ev_sample`), and `appears_tick` is the end of
  the tick that delivered it — a 60-s monitor's event stamped 0.17 s before a tick boundary is
  delivered in the next window, which made "registered in the first tick" fail by 0.17 s.
- `scripts/progressive_report.py`: `--keep-res`/`--rescore` (re-score pickled runs), `--assemble`,
  `--skip-existing`, `--no-series` (the per-entity p series is only used for PG6's KS of `conf_*` and
  costs ≈ 0.9 GB per run), peak RSS per run.

### 16.3 Pack O: gates (seeds 0–2, registry `progressive_decision`, 21 days at 900 s)

Source: `reports/progressive/progressive_report.json` (per seed: `runs/O_<seed>.json`). Each run:
≈ 261 000 behaviour events (582 anomalous), 27–28 min wall with three runs in parallel, peak RSS
1.37–1.40 GB, no engine exception. "Before" = the same runs at the start of this round (seeds 0–1,
`reports/progressive/baseline/`), i.e. without M19–M25.

| Gate | Target | Seed 0 | Seed 1 | Seed 2 | Before (seed 0 / 1) | Status |
|---|---|---|---|---|---|---|
| PG1 recall @ day 14 | ≥ 0.90 | 0.21 | 0.18 | 0.21 | 0.18 / 0.16 | fail |
| PG1 precision @ 14 | ≥ 0.90 | 0.30 | 0.39 | 0.34 | 0.32 / 0.40 | fail |
| PG1 components who / when / content / bindings / workflow | ≥ 0.85 each | .45/.66/.53/.67/.92 | .37/.66/.50/.67/.92 | .39/.71/.55/.67/.92 | .37/.55/.47/.67/.77 (s0) | workflow passes |
| PG2 ECE | ≤ 0.05 | 0.39 | 0.43 | 0.36 | 0.50 / 0.43 | fail |
| PG3 GA login who = 3 IPs | yes | yes | no | no | yes / no | fail (1/3) |
| PG3 finance approval who = {192.168.2.10} | yes | no | no | no | no / no | fail |
| PG3 P11 ARI vs departments | ≥ 0.9 | 0.86 | 0.86 | 0.87 | 0.97 / – | fail |
| PG5 D2 rebinding (mike → mike.w) | ≤ 5 events, 2 d | pass | pass | pass | pass | pass |
| PG5 D1 / D3 / D4 / non-adoption (.21 still jack) | pass | fail | fail | fail | fail | fail |
| PG6 anomalies detected (violation + incident) | ≥ 0.95 | 7/10 | 6/10 | 6/10 | 4/10 / 4/10 | fail (19/30) |
| PG6 FAR incidents ≥ LOW / entity-day | ≤ 0.10 | 0.027 | 0.032 | 0.031 | 0.030 / 0.032 | pass |
| PG6 FAR incidents ≥ MEDIUM / entity-day | ≤ 0.02 | 0.025 | 0.029 | 0.029 | 0.028 / 0.029 | fail |
| PG6 pattern_violation ≥ LOW / entity-day | ≤ 0.05 | 0.033 | 0.041 | 0.040 | 0.047 / 0.053 | pass |
| PG7 registered in the first tick / role ≤ 24 h / informative kept | 1 / 1 / ≥ 0.9 | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 0 / 1 / 1 | pass |
| PG7 type correct | ≥ 0.95 | 0.67 | 0.67 | 0.67 | 0 | fail (header typed `text`, see below) |
| PG8 arms ∈ strategy truth @ 14 | ≥ 0.95 | 0.33 | 0.67 | 0.50 | 0.33 / – | fail |
| PG10 view checks (day 11, day 21, finance single IP, GA negative) | all | none | none | none | none | fail |
| PG9 (packs A–E × bounded), PG11 (O-real), O-red, O60 | – | not run | | | | not measured |

PG7 type: `hdr.x-client-ver` (two version strings) is typed `text` because P02 types every string
payload attribute (`body`, `q`, `hdr`) as `text` so that P07 fits a grammar and a closed set
(§16.6); the truth says `categorical`. `meta.waf.score` is `ordinal` as in the truth.

### 16.4 The anomalies of the requirement (PG6)

| | Expected | Seed 0 | Seed 1 | Seed 2 | Before | Engine chain / cause |
|---|---|---|---|---|---|---|
| A1 192.168.1.23 approves in finance | who, ≥ MEDIUM | detected (who) | detected | detected | seed 0 no, seed 1 yes | M20–M23: was LOW (scored as the approver's "colleague", node U 0.029) |
| A2 .21 logs in as rose | binding, ≥ MEDIUM | detected | — | — (incident without the typed violation) | seed 0 yes, seed 1 no | needs the .21 → jack binding confirmed before day 17; the GA login node is split late or by client stack (§16.6) |
| A3 injection + 12 KB login | content | detected | detected | detected | detected | P06/P07 |
| A4 login at 03:05 | when | detected | detected | detected | detected | P09 |
| A5 report without its form page | seq | — | — | — | — | P10: the form → generate dependency is learned, but no missing-predecessor finding at 17:02 (not diagnosed in this round) |
| A6 GET /admin/export | novel, ≥ MEDIUM | detected | detected | detected | — | M24 (route was `/{var}/{var}`) |
| A7 400 portal logins in one hour | content (rate), ≥ MEDIUM | violation LOW only | — | — | LOW | population rate digest: crawler-like portal sources put 400/h inside the tail (WHO/VIEWS owner's open issue) |
| A8 unknown FIN-subnet IP as lucy | who + concurrent_use | detected HIGH | detected | detected | LOW | M20/M21 let the approver's own nodes confirm, so P08's lucy → .2.10 binding exists: `concurrent_use` |
| A9 sales IP reads the approval list daily (slow poisoning) | who ≥ LOW on day 11 and day 21, never in the heavy set | violation LOW (no incident) | same | same | same | B27 opens only on ≥ MEDIUM (§9.3); GET is not a write, so sensitivity < 2 keeps it LOW; outside the heavy set on day 21 in 2 of 3 seeds |
| A10 comment spam from 50 public IPs | who (region) + content | detected | detected | detected | detected | |

FAR: 136–164 clean entity-days with an incident ≥ MEDIUM per seed (budget 0.02 → 110). The largest
single source (seed 0) is 39 MEDIUM `novel` findings for SALES' legitimate weekly report (`POST
/report/weekly`, Fridays 16:00–17:00): the 20 sales IPs are flagged on day 11 and again on day 18.
Probable cause (not verified row by row): the MEDIUM incidents make B28 mark the sales IPs
`suspect` (26–34 `regime: suspect` events on 192.168.3.x per seed), P10 does not learn rows of
quarantined sources, so the weekly action never enters P10's action dictionary — a feedback loop
that an analyst label would break (open). The finance approver, which produced 8 HIGH incidents
before M20, produces no HIGH finding and 0–4 MEDIUM incidents per seed.

### 16.5 The requirement's example, as learned (seed 0, day 21; both views)

Rendered statements (zh, as produced by P14; English versions are in the report):

- **System view, OA login** — `【oa】工作日 08:30–09:15（覆盖 100 %，10 个工作日），10.168.7.121、192.168.1.21、192.168.1.23访问 POST /login：…提交数据量 90 % 在 1.2–1.8 KB，观测范围 1.1–1.8 KB（n = 8）；…username= 取值 [a-z]{4}(\.[a-z])?；…绑定：192.168.1.21 → username={jack,rose}`, plus the
  "某类人" parts per learned group: `综合部（10.168.7.121、192.168.1.23）访问 POST /login …` and
  `G22（192.168.1.21）访问 POST /login …`.
- **System view, OA approvals** — `G22（192.168.1.21）访问 POST /approval/{num}/approve：工作日 09:34–11:29 … 提交数据量 90 % 在 0.8–1.4 KB，全部在 0.8–1.5 KB（n = 50，下次越界概率 ≤ 4.0 %）` (state `stale` after D3 renamed the route; the `/flow/...` nodes are candidates).
- **System view, 17:00 report** — `【oa】工作日 17:01–17:14（覆盖 100 %，11 个工作日），综合部（10.168.7.121、192.168.1.23）访问 POST /report/generate：提交数据量 90 % 在 20–55 KB …`, preceded by `GET /report/form` 17:00–17:10.
- **System view, finance approval** — `【finance】工作日 10:00–11:27、15:01–15:56 … 192.168.2.10、192.168.3.33 访问 GET /fin/approval/list` and `来自 192.168.2.0/24（约 3 个 IP）访问 POST /fin/approval/{num}/approve`.
- **User (group) view** — `综合部（2 个 IP）使用 mail、oa` with `综合部 在 crm 中从未执行写操作（15 天、0 次）`, `综合部 在 portal 中从未执行写操作`; `G22（192.168.1.21）… 在 finance 中从未执行写操作（21 天、0 次）（封闭的写操作：POST /fin/approval/{num}/approve、POST /fin/login、POST /fin/voucher/create）`.
  综合部's view correctly has **no** "never writes in finance" statement: A1 (192.168.1.23
  approving in finance on day 16) is exactly such a write.

Checklist against the truth valid on day 21 (`example.checklist` in the report):

| Clause | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| 综合部 3 IPs log in to OA | pass (Jaccard 1.0) | 0.67 | 0.33 |
| workday window 08:30–08:51 (after D1) | IoU 0.47 (08:30–09:15) | 0.00 | 0.46 |
| 90 % of submissions 1–2 KB | pass (1.2–1.8 KB) | 1.41–1.59 KB | 0.5–1.5 KB |
| 100 % in 0.5–3 KB | 1.1–1.8 KB (observed range, n = 8) | 1.4–1.6 KB | 0–12 KB (A3's 12 KB) |
| `username=`, ≤ 10 chars | pass `[a-z]{4}(\.[a-z])?` | none | pass `[a-z]{3,8}(\.[a-z])?` |
| bindings .21→jack, .23→rose, .121→mike.w | 1/3 (`.21 → {jack, rose}`) | 1/3 | 1/3 |
| .21 opens the approval pages | pass | pass | pass |
| 17:00 report by .23 / .121 | pass | pass | pass |
| finance approval only by 192.168.2.10 | 0.5 | 0 | 0 |
| group view: 综合部 never writes in finance | present (G22) | present | present |

What is recovered on all seeds: the approval and report activities with their actors, windows and
sizes, the login grammar on 2 of 3 seeds, the group view's negative statements, and the
anomalies A1, A3, A4, A6, A8, A10. What is not: the GA login node is isolated with its three IPs on
seed 0 only, and only from day 18 (on the others it is split by TCP-window class and /24, §16.6);
consequently the three bindings are not confirmed before A2 starts on day 17 and A2's borrowed
`rose` is adopted for .21 (PG5 non-adoption fails); the login window after D1 includes A2's 09:10
logins; and the finance approval statement includes the damped sources A9 (192.168.3.33) and A8
(192.168.2.99), so it is rendered at /24 or with two IPs instead of the single approver.

### 16.6 Diagnosis of the remaining failures (by engine)

1. **P04 lattice, OA login (PG1 who, PG3, PG10, A2, PG5).** On seed 0 the login node splits first on
   `net.win` (TCP window class, part of the client-stack source group) on day ≈ 10 and then by /24
   (`10.168.7.0/24 + 192.168.1.0/24` vs `192.168.2–3.0/24`); the three GA IPs sit in different
   window branches, and the node holding exactly them appears on day 18. The client-stack keeper is
   still a split candidate, and its gain (it predicts TTL / UA / window) beats the department split.
2. **P11 groups (ARI 0.86).** 综合部 is split by activity (192.168.1.21 approves, .23 / .121 report);
   on some days the finance approver joins 综合部's group (weighted signatures dominated by document
   reading, which both do). M22 removes the consequence for scoring, not the grouping.
3. **Rendering of damped sources (PG3 finance, PG10).** P04 keeps damped rows in the who mass and
   P14 renders the heavy set by mass; a slow-poisoning source (A9) and a one-off (A8) appear in the
   finance approval statement, and the IP level is not closed (each counts as a singleton in U).
4. **Novel feedback loop (FAR ≥ MEDIUM).** §16.4.
5. **A7, A5**: §16.4.
6. **P12 (PG8 0.33–0.67).** oa chooses `prefix` (truth: ip or grp) because P11's groups do not beat
   /24 in code length; portal / code as reported by the ADAPTIVE owner.
7. **Calibration (PG2 ECE 0.36–0.43).** Statements' stated confidence is below their held-out hold
   rate (under-confident); not tuned in this round.

### 16.7 Resources (PG4) and non-regression

Scaling points (`reports/progressive/scale/`, `progressive_only`, 3 days, seed 0; the spec asks for
7 days — shortened for wall time):

| Axis | Points | P-core memory | CPU per event (all P engines) | Slope (target) |
|---|---|---|---|---|
| IPs (portal population) | 500 / 5 000 / 20 000 | 22.0 / 27.2 / 30.0 MB | 2.0 / 3.0 / 2.9 ms | memory 0.085 (≤ 0.15) pass; CPU 0.107 (≤ 0.1) fail |
| attributes (after M19) | 40 / 100 / 340 | 22.0 / 31.4 / 36.5 MB | 2.0 / 2.6 / 2.6 ms | memory 0.23 (≤ 0.2) fail; CPU 0.12 (≤ 0.2) pass |
| systems (O-servers, 12 families) | 20 / 100 | 30.7 / 42.7 MB | – | memory 0.21 (≤ 0.3) pass |
| idle system after 1 day | | 0.41 MB | | ≤ 1 MB pass |
| largest tree | | 11.6 MB | | ≤ 40 MB pass |
| scoring p95 / learning p95 per event | | | 0.19–2.1 ms / 1.6–5.0 ms | ≤ 100 µs / ≤ 250 µs: fail |

The 5 000 / 20 000-IP points were measured before M19–M24 (they do not use `meta.*`; only
`meta.waf.score` differs) under a different machine load, so the CPU slope mixes two code states.
The absolute per-event costs fail by 10–20× (the Python-cost risk of §14.2); a P04 profile at
5 000 IPs shows no single hotspot (Space-Saving updates 15 %, target summaries 38 %, e-process
updates 7 %), and the per-event P04 cost grows 965 → 2 049 µs from 500 to 20 000 IPs (slope 0.20),
mostly from more Space-Saving evictions per event. 300 systems and the 7-day points were not run.

Non-regression (`full` registry, the default): packs smoke, mini (seed 0), A (seed 0) and E (seed 0)
are bit-identical to `reports/round4/runs` (every key except timings). Smoke and mini were
re-checked after all changes of this round; A and E were checked before M19–M24, whose modules
(P00, P03, P04, `Templater.apply_path`, `DecayedVector`) are not executed by the `full` registry.
Full suite after all changes: 2 769 passed, 4 skipped.

### 16.8 Not done in this round

Five seeds, O-red, O-real (PG11), O60, PG9 (packs A–E with the core on and in bounded mode), the
300-system and 7-day scaling points, and pack O with the full B-library (`full+progressive`, §16.2
M25) were not run.


### 16.9 Round 2: scenario adaptation, bounded decision chain, cost (2026-10-01)

Owner: adaptation & cost (P12, P15, P00, P01, eval, orggen, report, build, bounded B paths).
Runs: pack O `progressive_decision`, 21 d; packs A and E seed 0 with the default registry in
`full` and `bounded` resource mode; PG4 points 7 days. Working tree of 2026-10-01 (other
owners' round-2 changes included, so gates other than PG7/PG8/PG9/PG4 moved for their
reasons too). Deviations (ids A*, alongside the M-rows of the other owners):

| # | Where | Specified (§) | Changed to | Why (measured) |
|---|---|---|---|---|
| A1 | P12 who arm utility (`lib/pstrategy.who_utilities`, `system_profile.WhoCode.observe_beh`), PG8 truth (`orggen` strategy, `pmetrics.who_arm_utilities`) | U(who arm) = bits(none) − bits(arm), the per-level prequential two-part code of the source IP alone (§6.18.2); truth who lists written by hand | U(arm) = **held-out behaviour gain** G_beh(arm) (prequential code of b = (action, workday, local hour) given the who item at that level vs the marginal; Space-Saving bounded state, same slots per level, bottom-k 384 sources per tick) + TIE_W·clip(address-code saving)/32 (TIE_W = 0.05 bits/event, the switching margin: decides only between levels whose behaviour gains are within noise) − STATE_COST (1e-3); `none` only while no level's address code saves ≥ 1 bit/event. PG8's third clause and the who truth use the same utility offline (exact groups, unbounded) on the generator's (source, action, hour) log (`gen.act_log`); truth = arms within 5 % + 0.01 bits/event of the best: oa grp/prefix, finance ip, crm grp/prefix, code and mail grp/prefix/reg, portal prefix (identical on seeds 0–2) | the who arm decides at which granularity behaviour is conditioned on who (P04's @who coding target, P11's signature level, P08's screened levels) while the who summaries are kept at every level anyway (§5.5.3), so what an arm buys is predictive information about behaviour, I(who_ℓ; behaviour), not address compression. Measured offline on pack O seed 0 (bits/event): oa G_beh grp 0.92, /24 0.90, ip 0.13 (研发's pool users re-address daily, the per-IP model relearns them); finance ip 0.20 > grp 0.11 (approver vs bookkeepers); portal ip −2.0 (returning visitors: their addresses code well, their behaviour per IP does not), /16 0.25. Under the address code alone OA chose prefix and portal per IP; the hand truth listed ip for oa / crm / mail although per-IP conditioning loses there |
| A2 | PG7 truth (`orggen` attr schedule) | `hdr.x-client-ver` truth type `categorical` (§5.3 rule 5) | `text`: a string in a payload namespace (hdr.*, body.kv.*, q.kv.*) uses the text/shape hierarchy (§5.4.5) and P07 states a few-valued one as its closed set (the categorical constraint) next to its grammar; §5.3 rule 5 governs non-payload strings | typed categorical, payload values lose their grammar (the requirement's username with 3 values); P02 already types payload strings `text` (§16.6), the truth contradicted the hierarchy the spec assigns |
| A3 | P12 fitter measurement (`system_profile.measurements`, `fitted_gain`) | the fitter's gain per day; fallback to its own `bits_per_event` | per-node records only; while the system is younger than 7 days a P08 pair record that has not judged a source (no source with n_bind events) and an empty record set are **unmeasured** (no Hedge update); after 7 days they are a gain of 0 (nothing to bind, e.g. a portal of one-off visitors) | pack O seed 0/2: finance's P08 was switched off on day 7 – before its users had the 5 logins a binding needs – and stayed off (probe every 14 d); PG8 finance P08 wrong on 2 of 3 seeds |
| A4 | `pipeline/build.py` | bounded mode reads P15's sets | P15 is registered in the default registry when `lib3.resource_mode = bounded` | without it no sets were published and `pactive` fell back to every entity: `bounded` on packs A–E was identical to `full` (PG9 had never measured bounded mode) |
| A5 | bounded decision chain: P15 `_release_idle`; B24 `calibration._bounded_rings`, B25 `fusion._bounded_meta`, `m_calib.ring`, `pactive.pool_of/pooled` (opt-in) | §10.3: B24 per-entity rings for earned IPs, class rings for unearned; B25 class meta rings + CUSUM LRU flushed after 7 idle days | per-IP chain state (model.calib incl. B25 meta/CUSUM, model.governor, model.control) exists only for earned and ACTIVE IPs: P15 releases it after 7 idle days (an LRU of the tick's sources, never a scan) unless earned or in the active set (open incident, regime event in 7 d) → O(|E_t| + |A_7d|) instead of O(every IP ever seen). Class-pooled rings for unearned IPs are implemented but OFF by default (`lib3.pool_unearned`) | measured (seed 0): with pooling, pack E's T5' (oa-portal 10.30.2.22, unearned) fell from CRITICAL to LOW (missed) and pack A gained a MEDIUM false alarm on an unearned IP; members' score nulls differ even when their feature models do not beat the class (B04's earned test is about the feature model, not the score's null). The spec's bet is refuted for calibration; activity-proportional per-IP rings keep the power |
| A6 | `eval/pscale.pcore_cpu` | PG4 scoring p95 per event, learning p95 per learned event | measured per SCORED event (P00's batch rows) for P03 and per LEARNED event (P00's learning sample) for P04, from a tap on P00's per-tick stats; units recorded | the harness divided P04's time by all generated events, understating the learning cost by the sampling ratio |
| A7 | `scripts/progressive_report.py` | – | top-level `registry` label = what the runs recorded (`registry`, `resource_mode` per run) | the report said "pack default (full+progressive)" while every run used `progressive_decision` |
| A8 | P00 value policy (`lib/pparse.ValuePolicy`) | – | (name, value) memo for payload strings (≤ 256 chars, ≤ 65 536 entries), non-payload names skipped before the call, the policy kept across ticks; results identical (test) | the policy was 5.2 s of P00's 11.7 s on a 2-day O-scale run (991 k calls, randomness test ~35 µs a value) |
| A9 | `lib/pstrategy.switch_margin` | margin = 2·sqrt(var(U_cur) + var(U_new)) over the last 7 daily utilities | 2 × the noise of the daily DIFFERENCE, detrended: SD(day-to-day change of U_new − U_cur)/√2 | the level variances counted the trend both arms share while a system's models are learnt as noise: pack O seed 0 finance, margin 1.5 bits/event against a steady 0.17 advantage of the per-IP arm (it led from day 11 and was adopted on day 20) |
| A10 | `eval/pmetrics.pg8_adaptation` switch count | every change of `chosen` after day 7 | changes of the arms the truth names (who, P07, P08, P10); a probe day's arm (`sysprof.probe`) is not a switch; P15's tier / e_max are not strategies | finance counted 4 "switches" on seed 0, three of them its P08 probe day and back; crm / code / mail 2 each from tier and content probes |
| A11 | `lib/pstrategy.decide` (who), `system_profile` (`who_day`, `who_pred_day`) | Hedge on the day's utilities | the who arm's Hedge round and switch test use the last COMPLETED day's figures (non-overlapping), each component (address code, behaviour gain) its day figure when that day had ≥ 30 of its evidence units, else its 7-day figure | P12 fed Hedge the 7-day sums every day, counting each day's evidence seven times: the leader followed a change of the best level a week late (pack O seed 2, finance: per IP led the day's gains from day 10, the leader turned on day 13, the switch landed on day 15, after PG8's day-14 check; a department of three addresses has 9–21 address units a day but 114–173 behaviour units, so the day rule must be per component) |

**PG8 (scenario adaptation), day 14.** Before = the round-1 runs (`reports/progressive/runs`,
P12 with the address-code utility); after = P12 with A1, A3, A9, A11, scored with A10 (runs of
the working tree, `progressive_decision`, seeds 0–1 tuning, seed 2 held out).

| | Seed 0 | Seed 1 | Seed 2 (held out) |
|---|---|---|---|
| arms ∈ truth, round-1 truth / round-1 runs | 0.33 | 0.67 | 0.50 |
| arms ∈ truth, settled truth (A1) / round-1 runs | 0.50 | 0.83 | 0.83 |
| arms ∈ truth, settled truth / after | 0.83 | 0.83 | 0.83 |
| who arm correct (6 systems) / after | 6/6 | 5/6 | 6/6 |
| chosen who within 5 % + 0.01 bits of the offline best / after | 6/6 | 5/6 | 6/6 |
| switches after day 7 (≤ 2) / after | 1 | 2 | 2 |

Intermediate runs located the last two causes: with the 7-day sums as Hedge rounds the held-out
seed chose `prefix` for finance on day 14 (per IP led every day's gain from day 5, the switch
landed on day 15: A11), and with an all-or-nothing day rule the finance day never qualified
(9–21 address units a day). With daily rounds seed 1's finance is the late one: per IP leads
most days by 0.1–0.25 bits/event, Hedge's leader turns on day 12, but the low-traffic days 14
and 15 (≈ 115 vs ≈ 165 behaviour units) shrink the difference below the margin and restart the
3-day streak; the switch lands on day 18. The 3-day rule is not seasonality-aware (open).
The remaining miss on every seed is finance's P08 (truth `on`, chosen `off`): finance's login
node screens only `net.src → body.fmt / hdr.content-type` (no username pair), whose code
gain is 0, so the arm's measured utility is −λ·cost — an open issue of P08's pair screening,
not of the selector.

**PG7 type correct:** 1.0 on seeds 0–1 (0.67 before), by A2.

**PG9 (bounded mode).** Packs A and E, seed 0, default registry (B01–B30), `full` vs `bounded`
(P15 now registered in bounded mode, A4). "Bounded, round 1" = the bounded B-library as it was
plus A4; "bounded, final" = + A5 (pooling off).

| Pack A seed 0 | full | bounded, round 1 | bounded + class pooling (A5 opt-in) | bounded, final |
|---|---|---|---|---|
| scenarios detected (gate 1) | 12/12 | 12/12 | 12/12 | 12/12 |
| gate-1 value | 0.667 | 0.667 | 0.667 | 0.667 |
| FAR incidents LOW / MED / HIGH (gate 3, 76 entity-days) | 14 / 6 / 2 | 19 / 6 / 1 | 21 / 7 / 1 | 19 / 6 / 1 |
| lib-3 p95 per tick at 35 entities | 1 230 ms | 980 ms (80 %) | 1 003 ms | 1 072 ms (87 %) |
| earned IPs | (all) | 35 / 43 | 34 / 43 | 34 / 43 |

| Pack E seed 0 | full | bounded, round 1 | bounded + class pooling | bounded, final |
|---|---|---|---|---|
| scenarios detected | 3/3 | 3/3 | 2/3 (T5' LOW) | 3/3 |
| gate-1 value (within deadline) | 0.0 | 0.667 | 0.333 | 0.667 |
| FAR LOW / MED | 2 / 0 | 2 / 0 | 3 / 1 | 2 / 0 |
| lib-3 p95 per tick at 35 entities | 407 ms | 265 ms (65 %) | 264 ms | 257 ms (63 %) |

| Pack O seed 0 (`progressive_decision`) | full | bounded (final) |
|---|---|---|
| anomalies detected | 10/10 | 10/10 |
| FAR ≥ LOW / ≥ MEDIUM per entity-day | 0.0062 / 0.0042 | 0.0062 / 0.0042 |
| PG1 recall / PG8 at day 14 | 0.237 / 0.83 | 0.237 / 0.83 |
| decision chain B24–B29 CPU | 903 s | 335 s (−63 %) |
| all engines / wall | 1 734 s / 1 774 s | 1 189 s / 1 231 s |

Reading: on pack O (≈ 700 sources, most of them transient) the bounded chain gives identical
gates at 37 % of the chain's cost — the per-(system, IP) state of inactive IPs was pure cost.
On A and E (35–43 entities, all active every day) the chain's state is the same in both modes
(final column = round-1 bounded exactly on A); the bounded mode's FAR on A (+5 LOW, n.s. on one
seed: binomial p ≈ 0.25) comes from the earned-only B06 / B07 / B10 / B14 / B15 (extra alarms
are on EARNED IPs, axes change / timing / volume), and its latency is 63–87 % of full mode
(target ≤ 50 %): with every entity active, little is skipped. Class-pooled rings for unearned
IPs (the spec's §10.3 B24 row) were implemented and measured: they cost a detection (E: T5')
and a MEDIUM false alarm (A), so they are off by default (A5).

**PG4, 7-day points** (`reports/progressive/scale7/`, `progressive_only`, seed 0, the working
tree of 13:30–14:00 with other owners' round-2 changes; four to five CPU-bound processes on four
cores, so absolute µs are inflated against round 1's 3-day points). Per-event costs now in the
§12 units (A6): scoring per scored event, learning per learned event (at these sizes the
learning sample keeps every event, so the two counts are equal).

| Point | P-core memory | largest tree | µs per event (all P engines) | P03 per scored event mean / p95 | P04 per learned event mean / p95 |
|---|---|---|---|---|---|
| 500 IPs, 40 attributes | 25.4 MB | 8.5 MB | 2 617 | 208 / 422 µs | 771 / 1 834 µs |
| 5 000 IPs, 40 attributes | 27.6 MB | 9.1 MB | 2 259 | 255 / 485 µs | 980 / 2 675 µs |
| 500 IPs, 100 attributes | 34.1 MB | 10.2 MB | 4 204 | 252 / 630 µs | 941 / 2 732 µs |
| 500 IPs, 340 attributes | 35.3 MB | 10.3 MB | 4 436 | 260 / 697 µs | 938 / 2 652 µs |

| PG4 check | Round 1 (3 d) | Round 2 (7 d) | Target |
|---|---|---|---|
| memory slope vs IPs | 0.085 | 0.036 | ≤ 0.15 pass |
| CPU / event slope vs IPs | 0.107 | −0.06 | ≤ 0.1 pass |
| memory slope vs attributes (40/100/340) | 0.23 | 0.147 | ≤ 0.2 pass |
| CPU / event slope vs attributes | 0.12 | 0.236 | ≤ 0.2 fail |
| largest tree | 11.6 MB | 10.3 MB | ≤ 40 MB pass |
| scoring p95 per scored event | 0.19–2.1 ms (per generated event) | 0.42–0.70 ms | ≤ 100 µs fail |
| learning p95 per learned event | 1.6–5.0 ms (per generated event) | 1.8–2.7 ms | ≤ 250 µs fail |

P00 on the same 2-day observations, sequential, same machine: 78.6–80.9 → 55.6–58.1 µs per
event (−28 %, A8 plus the column-wise learning-sample strata; outputs identical, tests). P12's
behaviour-gain code (A1) adds ≈ 25–40 µs per event (bounded: ≤ 384 sources per tick). The
per-event targets are a P03 / P04 matter: on a profiled 2-day run P04's `_learn_one` is ≈ 1 ms
per learned event (44 % target-summary updates: one `DecayedSpaceSaving.add` per event and
target), P03's `_score_event` ≈ 0.2 ms per scored event; grouping a tick's rows per (leaf,
target, value) and scoring per covering node with numpy are the open proposals (owners of
P03/P04).


**Not done in this round (this owner):** PG9 on packs B–D and seeds 1–4; the earned-only
B-engines' warm start at promotion (the likely source of A's extra LOW alarms); PG4's 20 000-IP
point and the 300-system point.

### 16.10 Round 2: results on the final code (evaluator, 2026-10-01)

Everything here is measured on ONE code state: HEAD 53cd151 (the round-2 checkpoint, which already
holds the P06–P10 hygiene H1–H6 and part of the P04 / P03 / P11 / P14 work) plus the four owners'
round-2 changes (P04/P05/P02: M26–M46; P03/P11/P14: G1–G7; P06: C1–C2; P12/P15/P00/eval: §16.9
A1–A11) plus the evaluator's fixes E1–E11 below. Runs: pack O seeds 0–4, 21 days, `progressive_decision` (now
the pack's default, E10), `--no-series`; O60 and O-red seed 0 (O-red is never used for tuning);
packs A and E seed 0 with the default registry. "Before" = the runs of HEAD 6001027
(`reports/progressive/runs/O_{0,1,2}.json`, seeds 0–2, scored by the round-1 evaluator; the
scoring changes of this round — §16.9 A10, E5, E6 — make some "after" checks easier to pass,
which is stated where it matters). Seeds 0–1 were used for development, seeds 2–4 are held
out. Medians are over the 5 seeds with [min–max]; bootstrap CIs are in the report JSON
(`reports/progressive/progressive_report.json`, per seed `runs/O_<seed>.json`).

Full suite on the final tree: see §16.10.7. Packs A and E seed 0 with the default registry are
identical, key for key apart from timings, to `reports/round4/runs/{A,E}_0_full.json` (the runs
behind `reports/eval_report.json`): the progressive core and this round's changes do not touch
the B-library's results.

#### 16.10.1 Deviations made in round 2

M26–M46 (P04 pattern tree, P05 selection, P02 registry; owner's rows, M44 tried and reverted):


| # | Where | Specified (§) | Changed to | Why (measured) |
|---|---|---|---|---|
| M26 | P04 `_tmask`, P05 `who_proxies` | every target of the leaf (incl. `@who`) pays for a split (§6.5.3) | a split is paid for by the BEHAVIOUR it explains: `@who` and P05's source properties (functions of the source shared by many sources, present on >= 50 % of the probe rows: client stack, TCP window, UA) are never split targets; an action's bound field (username) is not a source property | OA login node, seed 0, day 7: the TCP-window split had log2 e = 21.8, 25.0 bits of it from `@who`; the department /24 split's 25.9 bits were 29.1 from `client.stack` and 1.1 from the usernames |
| M27 | P04 `_leaf_cands`, `_check_valid_first`; `pevalue.selective_margin` | best valid candidate by G, (S) margin on D1 | who first: source properties are not tracked while a who level is a candidate; valid candidates ranked by selective gain and compared per target (clipped at 0) on their common events | offline replay of the captured OA login events (21 d): the 2-value window class led the department /24 by 50-170 bits on D1 for the whole run (prequential regret of the 8-slot who level on the targets it does not predict), although /24 had the larger selective gain (695 vs 598) and e-value (619 vs 505) |
| M28 | P04 `_leaf_learning`, `_new_coder` | a route node starts learning at creation; a young node codes from its parent while n_m < 16 | a route node of the root partition starts learning with LEARN_MIN units of its own; a node with < LEARN_MIN arrivals of its own takes its @when width from its parent (unless the parent is the partition root) | the @when width was judged on a handful of rows (or on the root's 24-h arrivals): 4-hour bins for the whole 14-day episode, @when log2 e ~ 0 on the OA login node; split children got M5 evidence but empty arrival histograms (width 240 again). Replay: /24 log2 e = 42 on day 4 at hourly bins |
| M29 | `pnode.WhoSummary.sus`, P04 `_learn_batch` | damped rows enter the who summary at 0.1 | a row P03 learned damped as a foreign source (damp < 1, p_who < 1) makes its source suspect at the nodes of its path; suspect sources' rows (damped or not) never enter the who levels (mass, evidence, U, heavy sets, rendering); forgotten after 30 d without a row; <= 16 per node | A9 (192.168.3.33): damped once (0.1) on day 11, then learned undamped (1.0 a day): in the finance approval-list heavy set by day 21; the statement widened to 2 IPs / the /24 |
| M30 | `pnode.HoldRecord`, `Node.p_hold`, P04 `_hold_check` | statement confidence = min of the constraints' nominal coverages (P14) | a calibrated node confidence: every learned event of a confident node is checked against what the node STATES as of its previous-day reference snapshot (who heavy set at the closed level with nominal 1 - U, P09 windows, P06 90 % bands, P07 closed sets, each with its stated coverage); p_hold = prod_c P(coverage_c >= nom_c - eps_c | Beta), eps_c = max(0.02, 3 sqrt(nom (1 - nom) / 300)), forgotten at H_l, reset with the confidence channel; exported in `to_plain` | PG2 ECE 0.36-0.43 and the median stated confidence FELL with time (0.48 -> 0.35, seed 0): the stated value is the smallest nominal coverage of the parts (often a workflow edge or 0 for an open who), not a probability that the statement holds |
| M31 | P04 `_pairs`, `_gc_pairs` | pair sketches per node as requested by P08 | <= 8 pair sketches per node (P08's request order); pairs no longer requested are dropped daily | PG4 attribute axis (340 attributes): 6.9 MB of the portal tree's 12.3 MB were pair sketches of every pair ever requested |
| M32 | `pselect.same_source` | `X.keys` ~ `X.kv.*` only | also `X.len` ~ `X.kv.*`, `X.keys` | both children of the OA login node split on the body-size bin, paid by the length of the body's padding field |
| M33 | P04 `_do_split` / `_start_learning` (`_slot_seed`, `_apply_seed`), `_leaf_cands`; `pevalue.SplitStats.update` | children start learning from empty split statistics | a child that is a value GROUP of the split attribute starts its first episode on that candidate from the parent's per-value counts (slot predictors and value dates only; the e-process starts at 1); the who-constant shortcut is judged only after LEARN_MIN arrivals of the node's own; the check cadence counts every unit, also while no candidate is tracked | after a first /24 split (研发 pool vs the rest, day 5) the `other` child's who candidate was judged constant on the parent's copied top-8 sources (all 销售部) and dropped, and since the cadence only counted while a candidate was tracked it never got one back (no second split in 21 d on a 3-group replay; with M33 综合部 is isolated by the second split) |
| M29b | P04 `_held` | released held rows are learned like any row | a released row (B28 `_return` releases a quiet episode automatically) is learned with the suspect flag: the content is learned, the who is not | A1's and A8's held finance approval rows were released and made the approval statement `192.168.2.0/24 (约 3 个 IP)` on day 21 (seed 0) instead of `192.168.2.10` |
| M34 | `pregistry.set_role` / `observe` | every attribute keeps full value summaries | a `dropped` attribute is registry-only (§6.4): top values 32 -> 8, numeric digest / moments and set elements released, rebuilt on re-promotion | PG4 attribute axis: the registry kept a 32-value top, a t-digest and moments for each of the 300 synthetic attributes (2/3 of them noise) |
| M35 | P04 `_structural_alarm`, `_drift_daily` | an ADWIN alarm makes the node `evolving` at once | the alarm is provisional: the node keeps its state until a whole normal day shows its mean loss >= 0.5 bit above the pre-alarm level (then `evolving`; accepted after T_persist such days; cleared as false after T_persist + 2 normal days without) | pack O's 60-s health-monitor nodes were `evolving` (not a stated pattern) on day 14 and day 21: AUTO.monitor x3 never recovered |
| M36 | `pselect.evaluate` (P05 U_s), `attr_select` | split utility measured on the system target list | measured on the behaviour: source properties excluded, the time of day (`ctx.tod_min`) added as a target | pack O mail (opaque TLS): the system target list was the TCP-window class alone; net.src stayed `shape`, no split candidate was ever tracked, the departments' mail windows (09:00-09:30 / 09:15-09:30 / 09:25-09:40 / 10:00-10:30) were never separated |
| M37 | P04 `_try_split` (constant candidates) | a candidate with one occupied value slot after n_g units is constant at the leaf until the R_learn restart | judged only after units of two local days (day0 <= today - 2): the stream is time ordered, so a check's first n_g units are one slice of the day | pack O mail (opaque TLS, departments differ only in their hours): the who levels were 'constant' on the first morning (all units from the first department) and not offered again before R_learn (2 000 units, ~10 d): no mail split in 21 d; after: /16 split day 4, learned-group split day 9-12 (seeds 0, 1) |
| M38 | P04 `_leaf_cands` | the who facet = P05's (system-wide) top-2 IP levels | a ladder: P05's levels, then the finer department-scale levels below each (/16 -> /24, grp / reg -> /24, reg -> grp), two who slots taken by the first valid, non-constant levels | P05 proposed reg and /16 for mail (dev pool vs offices, system-wide); below the first split both were spent and /24 (the departments) was never offered at any node |
| M39 | P04 `_coder_targets` | the coder codes every target P05 / the fitters list, plus @who / @when | the time of day is coded once (@when): targets of the ctx.when source family (ctx.tod_min) are not coded again | with ctx.tod_min among P05's behaviour targets (M36) and P09's requests, a /24 candidate on mail was paid 43 + 40 bits for the same minute (rule (V) evidence doubled) |
| M40 | `pselect.same_source` | X.len ~ X.kv.*, X.keys | transport measures of a payload are the payload's size: net.bytes_up / net.pkts_up ~ body.len, body.kv.*.len (resp side likewise) | the 研发 login node split on the upstream packet-count bin (log2 e 29.5), paid only by body.len and body.kv.viewstate.len |
| M41 | P04 `_learn_batch` (M29 criterion) | a damped row with p_who < 1 makes its source suspect | ... only when the who is why it was damped: p_who <= every other P03 component of the row (p_when, p_content, p_seq, p_novel) and no credential-binding flag | 综合部's .21 / .121 were damped for the new login minute (day 12) while light in an ancestor's heavy set (p_who = 2U): suspect at the 综合部 login node, never entered its who, P03 saw non-members of the next reference and renewed the flag daily: the statement named .23 alone on day 21 |
| M42 | `psketch.HLL.fold`, `EpochHLL.set_precision`; `pregistry.set_role` | a dropped attribute keeps a p = 10 EpochHLL (2 x 1 KB registers) | folded exactly to p = 6 when dropped (sigma ~ 13 %), new epochs at p = 10 after re-promotion | PG4 attribute axis: an empty registry record is ~7 KB, 2.5 KB of it the HLL; 4 systems x 300 synthetic attributes, 2/3 noise (dropped) |
| M43 | P04 `_hold_constraints`, `_hold_check` | the hold record checks who, when, P06 bands, P07 closed sets | every part the statement states: also P06's hard range (1 - cover), P07's grammar (c_g (1 - U_s)) and required keys (0.99), P08's bound pairs x -> y (LB_x, <= 16 per node) | a statement whose user-name grammar or binding failed on new data kept the confidence of its bands |
| M45 | P04 `_hold_check` (`_hold_material`), `_learn_one` | the record of a constraint mixes the checks of every reference it was stated with, for its 30-day half-life; every learned row is a check | a constraint stated materially differently (intervals IoU / sets Jaccard < 0.8, nominal moved > 0.05) restarts its record; rows P03 / B28 learned with weight < 1 (outliers, low trust) are not checks | the narrow windows of a young node's first references failed on most events and stayed in the record: the median stated confidence fell 0.35 -> 0.005 by day 21 (pack O, eval with M30), conf_nondecreasing false |
| M46 | `pnode.HoldRecord.p_hold` | p_hold = product over the constraints of P(theta_c >= nom_c - eps_c) | the checks are grouped into held-out tests (>= 100 checked events or a week); a test passes when every constraint's coverage on the batch is >= nominal - eps (the 3-sigma tolerance of a 300-event check); p_hold = (passes + 1) / (tests + 2) over the last weeks (14-d decay) - the predictive probability that the next held-out test passes; the product is kept as p_constraints() | pack O: with 10-20 constraints per statement the product was ~0 for nearly every statement (median 0.005 on day 21, ECE 0.37-0.41) although 35-45 % of them held on the evaluator's held-out data |


H1–H6 (P06–P10 learning hygiene, content / time owner; committed in the round-2 checkpoint 53cd151,
before C1–C2; regression tests `tests/engines/test_p08_value_history.py`, `test_p09_hygiene.py`,
`test_p06_violation_ledger.py`, `test_p10_missing_predecessor.py`, `test_p07_inherit.py`):

| # | Where | Specified (§) | Changed to | Why (measured) |
|---|---|---|---|---|
| H1 | P08 `pfd.ValueHistory`, `classify`, `fit_pair`; `binding` | per-node binding counts; a rename re-binds after 5 events on ≥ 2 days (§6.12) | a tree-level value history per (pair, source) (LRU-bounded; first / last seen, clean events, clean normal days; only rows P03 did not damp are clean). A source's current value that qualifies as a rename (≥ REBIND_N clean events on ≥ REBIND_DAYS normal days) makes the older value `superseded`; a newcomer next to an established value is `pending` until confirmed the same way, persisting, without a governor episode, and never when it is another source's established value (a credential bound elsewhere is never adopted by persistence); superseded and pending values leave the fit; the binding is refit when only the history changed | round 1: per-node segment baselines did not follow D2 to the split children (.121 → {mike, mike.w}); A2's borrowed `rose` was adopted by persistence (.21 → {jack, rose} on day 21, 0/3 seeds "still jack") |
| H2 | P09 `time_window`, `pwindows` | windows from the node's slot histogram / reservoir, every arrival weight 1 | arrivals weighted by their learning mass relative to the median (a damped outlier, weight < 1/2, never extends a window); a node created by a source split takes the arrivals of ITS sources from the nearest ancestor reservoir whose context differs only by that source restriction | the young 综合部 login node fitted windows in slot mode on a handful of rows; damped rows widened windows |
| H3 | P09 `pwindows.regime_cut` | Page–Hinkley on the arrival minute, acceptance after T_persist (§6.9.2) | P09 finds the local date from which arrivals follow a different time-of-day law (weighted two-sample KS, α = 1e-3, ≥ 3 dates and ≥ 6 points per side; accepted with ≥ 2 sources or ≥ 5 dates) and fits from that date | D1 (09:00–09:21 → 08:30–08:51): the window read 08:30–09:15 (IoU 0.47) from a histogram mixing both regimes |
| H4 | P06 `content_bounds` ledger, `pbounds.clean_range` | the hard range = min / max of the clean days | rows P03 judged violations (FIFO ledger, 1 024 rows per tree) are excluded from the range (refined by C1, C2) | A3's 12 KB login entered the login range on seed 2 ("0–12 KB") |
| H5 | P10 `pdfg` p_req | p_req = (c(b without a) + ½) / (c(b) + 1) | when b OPENS the session, the session-start transition (starts(b) + ½) / (sessions + 1) is also a predictive test; the one with the larger reference class (chosen from past counts, so no multiplicity charge) is used | A5: c(report) ≈ 20 sessions gives p_req ≥ 0.024, which can never pass P03's SEQ_P = 0.02, while the report had opened none of ≈ 150 GA sessions (A5 0/3 in round 1) |
| H6 | P07 `pgrammar.inherit_text`, `payload_grammar._ancestor_grammar` | a node publishes its own grammar only | a young node (e.g. created by a late split) states its nearest ancestor's own grammar when every shape it has seen matches it — looser, never wrong — without a closed set, confidence scaled by its share of the evidence, marked `inherited` | round 1, seed 1: the late 综合部 login node stated no `username=` grammar on day 21 |

G1–G7 (P03 conformity, P11 groups, P14 views), C1–C2 (P06 content bounds) and E1–E11 (the
evaluator: E1–E4, E7–E9, E11 engine fixes, E5–E6 scoring corrections, E10 pack default). Every
row has a regression test that fails without it (E-rows: `tests/engines/test_p14_views.py`,
`tests/lib/test_pwindows.py`, `tests/lib/test_pbounds.py`, `tests/eval/test_pmetrics.py`,
`tests/eval/test_orggen.py`, `tests/engines/test_p10_missing_predecessor.py`, each checked
against the pre-change files).

| # | Where | Specified (§) | Changed to | Why (measured) |
|---|---|---|---|---|
| G1 | P03 colleague rule (`conformity.group_members_at`, `signature_share`) | a member group at a node with ≥ 2 other members of standing (M22), read from the node's IP-level summary | colleagues counted from the node summary OR their own P11 signature holding the action (≥ 2 % share over ≥ 5 active days); min(2, other members); a one-address group judged by its own recurring use; a new group label falls back to the address's own signature (membership and `system_new`) | 10.168.7.121 was `outsider_group` from day 9 at the 30-source login node (its colleague .23 was not among the node's 8 heavy hitters), damped, trust 0, never learned again; 192.168.1.21 was `system_new` after P11 re-formed its group under a new id |
| G2 | P03 numeric p-values | rank / bounded-tail p | a value inside the observed clean range has p ≥ 2/(n+1), one just past an extreme (≤ 1 % of the width or one integer step) p ≥ 1/(n+1); float-noise tolerant comparisons | the bounded GPD tail gave the observed minimum of an integer count p = 1e-9 (exp(log 460) = 460.0000000000001) |
| G3 | P03 hourly counts | rank p | p = min(rank, moment bound from the node's own counts, variance ≥ mean) | the rank alone cannot go below 1/(W+1): A7 (400 portal logins in an hour) was LOW |
| G4 | P03 cross-binding credibility | the covering node's pair table | the best record of the same pair up the tree path | a who split restarted the child's table: A2 was credential-grade two days late |
| G5 | P11 naming | names matched on members | configured names matched on addresses only; pooled /24 sources published as `pools`, not members; a learned group lying ≥ 80 % inside a configured department is named as one of its roles `<dept>·<action>` (rec `dept`) | |
| G6 | P14 department view `class:grp:dept:<name>` | one view per learned group | the roles of one configured department are composed: systems, actions with their members, negative statements only where every role never wrote | 综合部 is learned as approver vs report writers (weighted Jaccard 0.47 vs cohesion 0.74) |
| G7 | P14 parts' confidence | min of the parts' nominal coverages | the node's held-out confidence (pnode `p_hold`) | parts stated 0.09 where their held-out hold was 1.0 |
| C1 | P06 violation ledger (`content_bounds`) | rows P03 judged violations leave the range | only typed p ≤ 1e-3, damped or injection-shaped rows; `above_range` / `below_range` / `grammar` / `length` flags are a young node's growth, not violations | real tail logins (s0 jack 2 616 B; s1 2 740 B) could never widen the range; the ledger (1 024 rows) filled with them and pushed out real violations |
| C2 | P06 `pbounds.clean_range` | a day whose extreme was a violating row leaves the range | only that SIDE of the day leaves; n_rng = min of the two ends' counts; excluded rows subtracted | at HEAD most GA login days had a damped .121 row as maximum: seed 0 lost 7 of 10 days (range 1.2–1.5 KB from 3 observations) |
| E1 | P14 `views.group_parts` members | members SEEN at the node (its IP-level summary) | ∪ members whose own P11 signature holds the action (`conformity.signature_share` ≥ SIG_STANDING, ≤ 64 per group) at a node without an address context | the IP-level summary keeps WHO_K = 8 heavy hitters: at the 25-source GET /docs node the 综合部 part listed 192.168.1.23 alone, 财务部 192.168.2.11 alone, 销售部 6 of 20 (PG1 who failed for every department's documents) |
| E2 | P14 `views.group_parts` departments | one part per learned group | the learned groups P11 names as roles of one configured department (rec `dept`) are ONE part of that department (the operator's "某类人"), with the roles' group ids; groups without a configured department stay one part each | 综合部's two roles gave parts {192.168.1.23, 10.168.7.121} and {192.168.1.21}, neither the department (Jaccard 0.67 / 0.33 against the truth) |
| E3 | P09 `pwindows.fit_daytype` coverage | in-sample share of the reservoir inside the windows | `predictive_coverage`: 1 − (n (1 − cov_in) + 2 n_windows) / (n + 1) in minute mode (edges are snapped to arrivals = order statistics; the rank bound P06 / P07 use) | statements said '覆盖 100 %' while 5–20 % of held-out arrivals fell outside the learned edges; `when` was the constraint failing most often in PG1 precision (24 of 36 failing statements, seed 0, day 14) and P04's hold records test windows at the stated coverage |
| E4 | P06 display (`pbounds.round_band`, `round_range`, `_text`) | coarsest 1-2-5 grid not coarser than the band width, both edges | a positive lower edge never reads 0: rounded on its own grid (coarsest step ≤ the edge, in the unit the band is displayed in) under the same coverage check; far-apart edges in their own units ('512 B–1.95 MB') | heavy-tailed sizes (mail uploads 1.1–230 KB, git pushes up to 2 MB) rendered '0–200 KB': no lower bound stated |
| E5 | eval `pmetrics.who_compatible` (PG1) | grp truth: listed members Jaccard ≥ 0.8 | also a prefix-level who whose prefixes hold every member and no other org source (`same_partition`) — PG8's settled semantics (§16.9 A1) | '来自 192.168.3.0/24 访问 GET /crm/customer/{num}' is exactly 销售部 and failed recall |
| E6 | eval `pmetrics.content_matches` (PG1) | displayed band / range endpoints ± 20 % / 25 % | the fitted band and observed range (`band90_raw`, `range_raw`); the display grid may move an endpoint by up to a grid step while keeping the coverage | portal login fitted 334–795 B, displayed '200–800 B' |
| E7 | P14 `views.group_parts` | parts wherever ≥ 2 groups share a node | no parts where P12 measured that the learned group carries no behavioural information (`who_pred[grp]` ≤ 0 bits/event after ≥ 200 units) | public portal (gain −2.2 bits/event): each part named one returning visitor ('G263（10.60.103.206）访问 POST /login'), PG3 portal login who ∉ {prefix, reg, any} on every seed |
| E8 | P14 `prender.who_block` | ladder ip → grp → /24 → /16 → region | when the system's who arm (P11 / P12 mode) is the region, the configured region is tried before the prefixes | 研发's pool 10.50.0.0/22 read as its four /24s: PG3 DEV who ∉ {grp, 10.50.0.0/22} on every seed |
| E9 | P14 `views.group_parts` departments | members of the department's learned groups | + the department's configured addresses (`who_group_names` ips) that P11 left in no group, when they use the node | the finance approver 192.168.2.10 had no learned group: 财务部's part of GET /docs and GET /home read {.11, .12} |
| E10 | `eval/packs._org_pack` | org packs default to `full+progressive` | `progressive_decision` (O, O60, O-red, O-real*) | the pack's own default could not run (M25: > 7 GB); every measured run passed --registry |
| E11 | P10 `pdfg.seq_scores` (P03's sequence test) | the requires-test of the source's P11 group scope ('*' only when the group was never mined) | backs off to '*' when the group scope has < REQ_MIN_B evidence of the action b (it cannot hold a requirement for b yet) | P11 re-formed 综合部 under a new id on day 17 (G12 → G31, seed 1); P10 mined a G31 scope from two days (edges, no requirement for the report): A5, the report without its form page on day 19, was scored against it and produced no finding on seeds 1 and 2 (`ev/diag/seq`), while '*' and the old G12 scope both required the form page |


#### 16.10.2 Gates (pack O, 5 seeds; status from `compute_pgates`)

| Gate | Status | Failing checks | Not measured |
|---|---|---|---|
| PG1 pattern recovery | fail | recall@14; precision@14; recall_when@14; recall_content@14; recall_bindings@14; GA+FIN bindings 6/6 at day 14; recall@14 (O-red); precision@14 (O-red); recall_who@14 (O-red); recall_when@14 (O-red); recall_content@14 (O-red); recall_bindings@14 (O-red); recall_workflow@14 (O-red); GA+FIN bindings 6/6 at day 14 (O-red) | – |
| PG2 convergence and calibrated confidence | fail | recall non-decreasing (±0.05) outside days 12-15; daily patterns: 80 % recall by day 7; weekly patterns: 80 % recall by day 21; mean depth non-decreasing before day 12; median confidence non-decreasing; median unseen-IP mass non-increasing; ECE | – |
| PG3 specificity reached | fail | GA login who = the 3 IPs; DEV pool one group / one prefix | – |
| PG4 resources sublinear in IPs, metrics and servers | fail | CPU/event slope vs attributes; scoring p95 us/event; learning p95 us/event | memory slope vs systems (12 families); idle system memory after >= 1 day (MB) |
| PG5 drift adaptation latency | fail | D1; D3; D4; D5 | portal rate.ip_h p99 within 10 % of pre-A7 |
| PG6 anomaly detection of the requirement's examples | fail | B29 top reason = violated constraint | KS D of conf_* p on clean ticks |
| PG7 open schema | pass | – | noise dropped; constants -> invariants |
| PG8 scenario adaptation | fail | chosen arms in strategy truth (day 14); who level code length within 5 % of best | – |
| PG9 non-regression | not measured | – | comparison |
| PG10 views | fail | OA statement day 11 (before D1); OA statement day 21 | – |
| PG11 real-world robustness | not measured | – | – |

Per check, median [min–max] over the seeds (after) against the round-1 runs (before):

| Check | After (median [min–max], n=5) | Before (n=3) |
|---|---|---|
| PG1 recall @14 | 0.579 [0.526–0.579] | 0.211 [0.184–0.211] |
| PG1 precision @14 | 0.529 [0.517–0.569] | 0.341 [0.298–0.385] |
| PG1 who @14 | 0.868 [0.842–0.895] | 0.395 [0.368–0.447] |
| PG1 when @14 | 0.842 [0.816–0.868] | 0.658 [0.658–0.711] |
| PG1 content @14 | 0.605 [0.579–0.632] | 0.526 [0.500–0.553] |
| PG1 bindings @14 | 0.667 [0.667–0.667] | 0.667 [0.667–0.667] |
| PG1 workflow @14 | 0.923 [0.923–0.923] | 0.923 [0.923–0.923] |
| PG1 recall @21 | 0.595 [0.524–0.615] | 0.286 [0.256–0.286] |
| PG1 precision @21 | 0.614 [0.491–0.643] | 0.375 [0.297–0.392] |
| PG2 ECE @14 | 0.300 [0.281–0.336] | 0.385 [0.358–0.431] |
| PG3 GA login who = 3 IPs | 4/5 | 1/3 |
| PG3 GA bindings (of 3) | 3.000 [3.000–3.000] | 0.000 [0.000–0.000] |
| PG3 finance approval = {.2.10} | 5/5 | 0/3 |
| PG3 portal login who level | 5/5 | 0/3 |
| PG3 DEV who ok | 5/5 | 0/3 |
| PG3 DEV pool grouped | 0/5 | 0/3 |
| PG3 ARI | 0.974 [0.974–0.974] | 0.857 [0.857–0.869] |
| PG5 D1 | 3/5 | 0/3 |
| PG5 D2 | 5/5 | 3/3 |
| PG5 D3 | 0/5 | 0/3 |
| PG5 D4 | 0/5 | 0/3 |
| PG5 D5 | 0/5 | 1/3 |
| PG5 .21 still jack | 5/5 | 0/3 |
| PG6 FAR inc >= LOW | 0.005 [0.004–0.007] | 0.031 [0.027–0.032] |
| PG6 FAR inc >= MEDIUM | 0.004 [0.003–0.004] | 0.029 [0.025–0.029] |
| PG6 pv >= LOW | 0.016 [0.012–0.021] | 0.040 [0.033–0.041] |
| PG7 type correct | 1.000 [1.000–1.000] | 0.667 [0.667–0.667] |
| PG8 arms in truth | 0.833 [0.667–0.833] | 0.500 [0.333–0.667] |
| PG8 max switches | 1.000 [1.000–2.000] | 3.000 [2.000–3.000] |
| PG10 day 11 | 0/5 | 0/3 |
| PG10 day 21 | 0/5 | 0/3 |
| PG10 finance single IP | 5/5 | 0/3 |
| PG10 GA negative (finance) | 5/5 | 0/3 |

#### 16.10.3 Progress with time (PG1 / PG2)

| PG1 (median over seeds) | day 3 | day 5 | day 7 | day 10 | day 14 | day 18 | day 21 |
|---|---|---|---|---|---|---|---|
| recall after | 0.00 | 0.22 | 0.24 | 0.55 | 0.58 | 0.56 | 0.60 |
| recall before | 0.00 | 0.16 | 0.18 | 0.23 | 0.21 | 0.26 | 0.29 |
| precision after | – | 0.41 | 0.55 | 0.58 | 0.53 | 0.58 | 0.61 |
| precision before | – | 0.40 | 0.41 | 0.29 | 0.34 | 0.28 | 0.38 |

#### 16.10.4 The anomalies of the requirement (PG6; finding types per seed)

| Anomaly | s0 | s1 | s2 | s3 | s4 | before s0/s1/s2 |
|---|---|---|---|---|---|---|
| A1 | yes who | yes who | yes who | yes who | yes who | y/y/y |
| A2 | yes content | yes content | yes content | yes content | **no** | y/n/n |
| A3 | yes content | yes content | yes content | yes content | yes content | y/y/y |
| A4 | yes when | yes when | yes when | **no** | yes when,who | y/y/y |
| A5 | yes seq | yes seq | yes seq | yes seq | yes seq | n/n/n |
| A6 | yes novel | yes novel | yes novel | yes novel | yes content,novel | y/y/y |
| A7 | yes content,seq | yes content | yes content,seq | yes content,seq | yes content,seq | n/n/n |
| A8 | yes who | yes content,who | yes who | yes content,who | yes content,who | y/y/y |
| A9 | yes who | yes who | yes who | yes who | yes who | n/n/n |
| A10 | yes content,seq,who | yes content,seq,who | yes content,seq,who | yes content,seq,who | yes content,seq,who | y/y/y |

detected 48/50 (after); 19/30 (before)

#### 16.10.5 The requirement's example: checklist per seed (day 21)

| Clause (truth valid on day 21) | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| 综合部 3 个 IP 登录 OA（who = 192.168.1.21、192.168.1.23、10.168.7.121） | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 | ✗ who Jaccard 0.67 |
| 工作日登录时间窗 08:30–08:51 | ✓ IoU 0.86 | ✓ IoU 0.91 | ✓ IoU 0.91 | ✗ IoU 0.69 | ✗ IoU 0.42 |
| 提交数据量 90 % 在 1–2 KB | ✗ band [1024.0, 2560.0] | ✓ band [1024.0, 2048.0] | ✓ band [1024.0, 2048.0] | ✗ band [1024.0, 3072.0] | ✓ band [819.2, 1946] |
| 100 % 在 0.5–3 KB | ✗ range [1024.0, 2662.4] | ✗ range [1024.0, 2867] | ✗ range [409.6, 2048.0] | ✓ range [512.0, 3072.0] | ✗ range [819.2, 2048.0] |
| 提交内容含 username=，取值不超过 10 个字符 | ✓ grammar [a-z]{4}(\.[a-z])? | ✓ grammar [a-z]{4}(\.[a-z])? | ✓ grammar [a-z]{4}(\.[a-z])? | ✓ grammar [a-z]{4}(\.[a-z])? | ✓ grammar [a-z]{4}(\.[a-z])? |
| 绑定 10.168.7.121→mike.w、192.168.1.21→jack、192.168.1.23→rose | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✗ 0/3 |
| 192.168.1.21 访问业务审批页面 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 |
| 下午 5 点提交报告（.23、.121） | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 |
| 财务系统审批只有财务部 192.168.2.10 访问 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 |
| 用户视角：综合部在财务系统中从未执行写操作（否定陈述） | ✓ present | ✓ present | ✓ present | ✓ present | ✓ present |


#### 16.10.6 The example as learned (seed 0, day 21, both views; P14's text verbatim)

**System view, OA (statements naming a 综合部 address):**

- `stable`, confidence 0.35: 【oa】工作日 08:32–08:51、09:10–09:11（覆盖 78 %，6 个工作日），10.168.7.121、192.168.1.21、192.168.1.23访问 POST /login：viewstate.len= 90 % 在 1000–3000，观测范围 900–1700（n = 8）；提交数据量 90 % 在 1–2.5 KB，观测范围 1–2.6 KB（n = 23）；下行字节 90 % 在 320–580 B，观测范围 300–600 B（n = 19）；上行字节 90 % 在 1.5–3 KB，观测范围 1.4–3.2 KB（n = 20）；时长 90 % 在 10–70，观测范围 10–80（n = 27）；net.pkts_down 90 % 在 2–2，观测范围 2–2（n = 20）；每 IP 每小时次数 90 % 在 1–1；body.fmt 取值 `[a-z]{4}`，body.fmt 取值集合封闭 {form}；表单键必含 captcha=、csrf=、password=、username=、viewstate=；password= 取值 `[A-Za-z0-9]{7,30}`；password.len= 取值集合封闭 {10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 8.0, …}；username= 取值 `[a-z]{4}(\.[a-z])?`，username= 取值集合封闭 {jack, mike, mike.w, rose}；viewstate= 取值 `[A-Za-z0-9\-_]{511,4094}`；请求头 content-type 取值 `[a-z]{11}/[a-z]\-[a-z]{3}\-[a-z]{4}\-[a-z]{10}`；请求头 x-client-ver 取值 `[0-9]\.[0-9]\.[0-9]`；username=jack 只来自 -|chrome/126|win|128|w16；username=mike.w 只来自 -|edge/125|win|128|w16；username=rose 只来自 -|chrome/126|win|128|w16；绑定：10.168.7.121 → username=mike.w、192.168.1.21 → username=jack、192.168.1.23 → username=rose（g3 = 0.00，各 ≥ 5 次）。流程：POST /login → GET /home（间隔 0 秒–4 秒）。置信 0.35 · 首次 2025-09-10 · 最近 2025-09-21 · v1.0
- `stable`, confidence 0.86: 【oa】工作日 08:32–09:20（覆盖 89 %，8 个工作日），综合部（10.168.7.121、192.168.1.21、192.168.1.23）访问 POST /login：viewstate.len= 90 % 在 500–1500，全部在 400–2600（n = 674，下次越界概率 ≤ 0.3 %）；提交数据量 90 % 在 0.5–1.5 KB，全部在 0.6–2.6 KB（n = 674，下次越界概率 ≤ 0.3 %）；下行字节 90 % 在 320–580 B，全部在 300–600 B（n = 645，下次越界概率 ≤ 0.4 %）；上行字节 90 % 在 1–2 KB，全部在 1–3.2 KB（n = 645，下次越界概率 ≤ 0.4 %）；时长 90 % 在 10–100，全部在 5–180（n = 691，下次越界概率 ≤ 0.3 %）；net.pkts_down 90 % 在 2–2，全部在 2–2（n = 566，下次越界概率 ≤ 0.4 %）；每 IP 每小时次数 90 % 在 1–1；body.fmt 取值 `[a-z]{4}`，body.fmt 取值集合封闭 {form}；表单键必含 captcha=、csrf=、password=、username=、viewstate=；password= 取值 `[A-Za-z0-9]{7,30}`；password.len= 取值集合封闭 {10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 8.0, …}；username= 取值 `[a-z]{3,8}(\.[a-z])?`；viewstate= 取值 `[A-Za-z0-9\-_]{511,4094}`；客户端栈 取值集合封闭 {-|chrome/126|win|128|w16, -|edge/125|win|128|w16, -|edge/126|win|128|w16, -|firefox/125|linux|64|w15, -|firefox/126|linux|64|w15, -|firefox/127|linux|64|w15, -|firefox/128|linux|64|w15, -|safari/17|mac|64|w16}；请求头 user-agent.len 取值集合封闭 {111.0, 114.0, 125.0, 70.0}；请求头 x-client-ver 取值 `[0-9]\.[0-9]\.[0-9]`；username=amy 只来自 -|firefox/127|linux|64|w15；username=brandon 只来自 -|chrome/126|win|128|w16；username=brian 只来自 -|edge/125|win|128|w16；username=carol 只来自 -|chrome/126|win|128|w16；username=david 只来自 -|chrome/126|win|128|w16；username=emma 只来自 -|chrome/126|win|128|w16；username=frank 只来自 -|chrome/126|win|128|w16；username=geclgk 只来自 -|firefox/127|linux|64|w15；绑定：10.168.7.121 → username=mike.w、192.168.1.21 → username=jack、192.168.1.23 → username=rose、192.168.2.10 → username=lucy、192.168.2.11 → username=tom、192.168.2.12 → username=kate、192.168.3.20 → username=amy、192.168.3.21 → username=brian（g3 = 0.00，各 ≥ 5 次）。流程：POST /login → GET /home（间隔 0 秒–4 秒）。置信 0.86 · 首次 2025-09-02 · 最近 2025-09-21 · v2.0
- `stale`, confidence 0.28: 【oa】工作日 09:34–11:29（覆盖 96 %，8 个工作日），综合部·oa GET /approval/list（192.168.1.21）访问 POST /approval/{num}/approve：viewstate.len= 90 % 在 800–1400，全部在 700–1500（n = 50，下次越界概率 ≤ 3.9 %）；提交数据量 90 % 在 0.8–1.4 KB，全部在 0.8–1.5 KB（n = 50，下次越界概率 ≤ 3.9 %）；下行字节 90 % 在 2–18 KB，全部在 1–20 KB（n = 52，下次越界概率 ≤ 3.8 %）；上行字节 90 % 在 1.4–2 KB，全部在 1.3–2 KB（n = 42，下次越界概率 ≤ 4.7 %）；时长 90 % 在 10–100，全部在 10–140（n = 46，下次越界概率 ≤ 4.3 %）；net.pkts_down 90 % 在 3–12，观测范围 2–12（n = 7）；每 IP 每小时次数 90 % 在 1–5；body.fmt 取值 `[a-z]{4}`，body.fmt 取值集合封闭 {form}；表单键必含 id=、opinion=、sign=、viewstate=；opinion= 取值 `[\u0080-\U0010ffff]{2,8}`，opinion= 取值集合封闭 {同意, 同意，请尽快办理, 退回}；viewstate= 取值 `[A-Za-z0-9\-_]{511,2046}`；请求头 x-client-ver 取值 `[0-9]\.[0-9]\.[0-9]`，请求头 x-client-ver 取值集合封闭 {5.2.1}。置信 0.28 · 首次 2025-09-02 · 最近 2025-09-12 · v1.0（近期未出现）
- `stale`, confidence 0.06: 【oa】工作日 09:32–11:28（覆盖 96 %，8 个工作日），综合部·oa GET /approval/list（192.168.1.21）访问 GET /approval/{num}：下行字节 90 % 在 10–40 KB，全部在 5–40 KB（n = 40，下次越界概率 ≤ 4.9 %）；上行字节 90 % 在 505–506 B，观测范围 505–506 B（n = 10）；时长 90 % 在 10–100，全部在 5–180（n = 53，下次越界概率 ≤ 3.8 %）；net.pkts_down 90 % 在 10–25，观测范围 8–26（n = 10）；每 IP 每小时次数 90 % 在 0–4；请求头 user-agent 取值 `[A-Z][a-z]{6}/[0-9]\.[0-9] \([A-Z][a-z]{6} [A-Z]{2} [0-9]{2}\.[0-9]; [A-Z][a-z]{2}[0-9]{2}; [a-z][0-9]{2}\) [A-Za-z0-9]{11}/[0-9]{3}\.[0-9]{2} \([A-Z]{5}, [a-z]{4} [A-Z][a-z]{4}\) [A-Z][a-z]{5}/[0-9]{3}\.[0-9]\.[0-9]\.[0-9] [A-Z][a-z]{5}/[0-9]{3}\.[0-9]{2}`；请求头 x-client-ver 取值 `[0-9]\.[0-9]\.[0-9]`，请求头 x-client-ver 取值集合封闭 {5.2.1}。置信 0.06 · 首次 2025-09-02 · 最近 2025-09-12 · v1.0（近期未出现）
- `stale`, confidence 0.12: 【oa】工作日 09:28–11:25（覆盖 96 %，8 个工作日），综合部·oa GET /approval/list（192.168.1.21）访问 GET /approval/list：下行字节 90 % 在 6–19 KB，全部在 4–20 KB（n = 53，下次越界概率 ≤ 3.8 %）；上行字节 90 % 在 505–505 B，观测范围 505–505 B（n = 10）；时长 90 % 在 20–100，全部在 10–140（n = 53，下次越界概率 ≤ 3.8 %）；net.pkts_down 90 % 在 5–15，全部在 6–16（n = 40，下次越界概率 ≤ 4.9 %）；每 IP 每小时次数 90 % 在 0–4；请求头 user-agent 取值 `[A-Z][a-z]{6}/[0-9]\.[0-9] \([A-Z][a-z]{6} [A-Z]{2} [0-9]{2}\.[0-9]; [A-Z][a-z]{2}[0-9]{2}; [a-z][0-9]{2}\) [A-Za-z0-9]{11}/[0-9]{3}\.[0-9]{2} \([A-Z]{5}, [a-z]{4} [A-Z][a-z]{4}\) [A-Z][a-z]{5}/[0-9]{3}\.[0-9]\.[0-9]\.[0-9] [A-Z][a-z]{5}/[0-9]{3}\.[0-9]{2}`；请求头 x-client-ver 取值 `[0-9]\.[0-9]\.[0-9]`，请求头 x-client-ver 取值集合封闭 {5.2.1}。置信 0.12 · 首次 2025-09-02 · 最近 2025-09-12 · v1.0（近期未出现）
- `confirmed`, confidence 0.21: 【oa】工作日 17:01–17:14（覆盖 91 %，11 个工作日），综合部（10.168.7.121、192.168.1.23）访问 POST /report/generate：提交数据量 90 % 在 20–55 KB，观测范围 20–60 KB（n = 25）；下行字节 90 % 在 2.2–4.4 KB，观测范围 2–4.5 KB（n = 15）；上行字节 90 % 在 20–55 KB，观测范围 20–60 KB（n = 15）；时长 90 % 在 10–100，观测范围 5–140（n = 27）；每 IP 每小时次数 90 % 在 1–1；body.fmt 取值 `[a-z]{4}`；表单键必含 dept=、items[]=、period=；dept= 取值 `[A-Z]{2}`；请求头 content-type 取值 `[a-z]{11}/[a-z]{4}`；请求头 x-client-ver 取值 `[0-9]\.[0-9]\.[0-9]`。流程：GET /report/form → POST /report/generate（间隔 74 秒–4 分钟）。置信 0.21 · 首次 2025-09-02 · 最近 2025-09-19 · v1.0
- `confirmed`, confidence 0.45: 【oa】工作日 17:00–17:10（覆盖 90 %，10 个工作日），综合部（10.168.7.121、192.168.1.23）访问 GET /report/form：下行字节 90 % 在 6–12 KB，观测范围 5–12 KB（n = 26）；上行字节 90 % 在 503–517 B，观测范围 502–518 B（n = 12）；时长 90 % 在 20–100，观测范围 10–120（n = 26）；net.pkts_down 90 % 在 6–10，观测范围 6–10（n = 22）；请求头 user-agent 取值 `[A-Z][a-z]{6}/[0-9]\.[0-9] \([A-Z][a-z]{6} [A-Z]{2} [0-9]{2}\.[0-9]; [A-Z][a-z]{2}[0-9]{2}; [a-z][0-9]{2}\) [A-Za-z0-9]{11}/[0-9]{3}\.[0-9]{2} \([A-Z]{5}, [a-z]{4} [A-Z][a-z]{4}\) [A-Z][a-z]{5}/[0-9]{3}\.[0-9]\.[0-9]\.[0-9] [A-Z][a-z]{5}/[0-9]{3}\.[0-9]{2}( [A-Z][a-z]{2}/[0-9]{3}\.[0-9]\.[0-9]\.[0-9])?`；请求头 x-client-ver 取值 `[0-9]\.[0-9]\.[0-9]`。流程：GET /report/form → POST /report/generate（间隔 74 秒–4 分钟）。置信 0.45 · 首次 2025-09-02 · 最近 2025-09-19 · v1.0
- `stable`, confidence 0.19: 【oa】工作日 09:30–17:30（覆盖 100 %，14 个工作日），综合部（10.168.7.121、192.168.1.21、192.168.1.23）访问 GET /docs：下行字节 90 % 在 10–30 KB，全部在 5–30 KB（n = 374，下次越界概率 ≤ 0.6 %）；上行字节 90 % 在 460–510 B，全部在 450–510 B（n = 776，下次越界概率 ≤ 0.3 %）；时长 90 % 在 10–100，全部在 5–200（n = 776，下次越界概率 ≤ 0.3 %）；net.pkts_down 90 % 在 8–22，全部在 6–24（n = 374，下次越界概率 ≤ 0.6 %）；每 IP 每小时次数 90 % 在 1–2；客户端栈 取值集合封闭 {-|chrome/126|win|128|w16, -|edge/125|win|128|w16, -|edge/126|win|128|w16, -|firefox/126|linux|64|w15, -|firefox/127|linux|64|w15, -|safari/17|mac|64|w16}；请求头 user-agent 取值 `[A-Za-z0-9\ \(\),\./:;_]{70,125}`；请求头 user-agent.len 取值集合封闭 {111.0, 114.0, 125.0, 70.0}；请求头 x-client-ver 取值 `[0-9]\.[0-9]\.[0-9]`，请求头 x-client-ver 取值集合封闭 {5.1.9, 5.2.1}；waf.score 取值集合封闭 {0.0, 1.0, 2.0}；net.win 取值集合封闭 {29200.0, 64240.0, 65535.0}。流程：GET /docs → GET /docs/{num}（间隔 10 秒–56 秒）；POST /docs/{num}/comment → GET /docs（间隔 6 分钟–29 分钟）；GET /docs/{num} → GET /docs（间隔 3 分钟–28 分钟）；GET /home → GET /docs（间隔 10 分钟–31 分钟）。置信 0.19 · 首次 2025-09-02 · 最近 2025-09-19 · v1.0

**System view, finance approvals:**

- `stable`, confidence 0.35: 【finance】工作日 10:00–11:27、15:01–15:56（覆盖 93 %，13 个工作日），192.168.2.10访问 GET /fin/approval/list：下行字节 90 % 在 12–27 KB，观测范围 12–28 KB（n = 5）；时长 90 % 在 10–100，全部在 5–140（n = 57，下次越界概率 ≤ 3.5 %）；每 IP 每小时次数 90 % 在 1–3。流程：GET /fin/approval/list → POST /fin/approval/{num}/approve（间隔 30 秒–2 分钟）；POST /fin/approval/{num}/approve → GET /fin/approval/list（间隔 6 分钟–28 分钟）。置信 0.35 · 首次 2025-09-02 · 最近 2025-09-21 · v1.0
- `stable`, confidence 0.35: 【finance】工作日 10:02–11:28、15:03–15:58（覆盖 92 %，13 个工作日），192.168.2.10访问 POST /fin/approval/{num}/approve：viewstate.len= 90 % 在 915–1500，观测范围 915–915（n = 1）；voucher= 90 % 在 20000000–100000000，全部在 10000000–100000000（n = 52，下次越界概率 ≤ 3.8 %）；提交数据量 90 % 在 1–1.6 KB，全部在 0.9–1.6 KB（n = 55，下次越界概率 ≤ 3.6 %）；下行字节 90 % 在 11.2–17 KB，观测范围 11–17 KB（n = 7）；时长 90 % 在 10–100，全部在 5–140（n = 55，下次越界概率 ≤ 3.6 %）；每 IP 每小时次数 90 % 在 1–3；body.fmt 取值 `[a-z]{4}`，body.fmt 取值集合封闭 {form}；表单键必含 amount=、opinion=、sign=、viewstate=、voucher=；opinion= 取值 `[\u0080-\U0010ffff]{2,8}`，opinion= 取值集合封闭 {同意, 同意，请尽快办理, 退回}；viewstate= 取值 `[A-Za-z0-9\-_]{511,2046}`；请求头 content-type 取值 `[a-z]{11}/[a-z]\-[a-z]{3}\-[a-z]{4}\-[a-z]{10}`，请求头 content-type 取值集合封闭 {application/x-www-form-urlencoded}。流程：GET /fin/approval/list → POST /fin/approval/{num}/approve（间隔 30 秒–2 分钟）；POST /fin/approval/{num}/approve → GET /fin/approval/list（间隔 6 分钟–28 分钟）。置信 0.35 · 首次 2025-09-02 · 最近 2025-09-19 · v1.0

**User view — the department (`class:grp:dept:综合部`, composed of its learned roles):**

- header: 综合部（3 个 IP，2 个行为群组）使用 mail、oa
- 综合部 访问 mail（占其活动 18 %）：邮件（TLS mail.corp.local）
- 综合部 访问 oa（占其活动 82 %）：文档（GET /docs）、文档（GET /docs/{num}）、文档（POST /docs/{num}/comment）、GET /home、登录（POST /login）、查看报告（GET /report/form）[10.168.7.121、192.168.1.23]、提交报告（POST /report/generate）[10.168.7.121、192.168.1.23]、查看审批（GET /approval/list）[192.168.1.21]、查看审批（GET /approval/{num}）[192.168.1.21]、审批（POST /approval/{num}/approve）[192.168.1.21]
- (negative) 综合部 在 crm 中从未执行写操作（15 天、0 次）（封闭的写操作：客户（POST /crm/visit））
- (negative) 综合部 在 finance 中从未执行写操作（21 天、0 次）（封闭的写操作：凭证（POST /fin/voucher/create）、审批（POST /fin/approval/{num}/approve）、登录（POST /fin/login））；192.168.1.23 的尝试被判定为越权（未学习）
- (negative) 综合部 在 portal 中从未执行写操作（21 天、0 次）（封闭的写操作：登录（POST /login）、评论（POST /comment））

**User view — the learned groups holding 综合部's addresses** (192.168.1.21 → G22, 192.168.1.23 → G10, 10.168.7.121 → G10):

- G10 `综合部` (10.168.7.121, 192.168.1.23): 综合部（2 个 IP）使用 mail、oa
  - (negative) 综合部 在 crm 中从未执行写操作（15 天、0 次）（封闭的写操作：客户（POST /crm/visit））
  - (negative) 综合部 在 finance 中从未执行写操作（21 天、0 次）（封闭的写操作：凭证（POST /fin/voucher/create）、审批（POST /fin/approval/{num}/approve）、登录（POST /fin/login））；192.168.1.23 的尝试被判定为越权（未学习）
  - 综合部 访问 mail（占其活动 21 %）：邮件（TLS mail.corp.local）
  - 综合部 访问 oa（占其活动 79 %）：文档（GET /docs/{num}）、文档（GET /docs）、文档（POST /docs/{num}/comment）、GET /home、登录（POST /login）、提交报告（POST /report/generate）、查看报告（GET /report/form）
  - (negative) 综合部 在 portal 中从未执行写操作（21 天、0 次）（封闭的写操作：登录（POST /login）、评论（POST /comment））
- G22 `综合部·oa GET /approval/list` (192.168.1.21): 综合部·oa GET /approval/list（1 个 IP）使用 mail、oa
  - (negative) 综合部·oa GET /approval/list 在 crm 中从未执行写操作（15 天、0 次）（封闭的写操作：客户（POST /crm/visit））
  - (negative) 综合部·oa GET /approval/list 在 finance 中从未执行写操作（21 天、0 次）（封闭的写操作：凭证（POST /fin/voucher/create）、审批（POST /fin/approval/{num}/approve）、登录（POST /fin/login））
  - 综合部·oa GET /approval/list 访问 mail（占其活动 14 %）：邮件（TLS mail.corp.local）
  - 综合部·oa GET /approval/list 访问 oa（占其活动 86 %）：文档（GET /docs/{num}）、文档（GET /docs）、审批（POST /approval/{num}/approve）、查看审批（GET /approval/{num}）、查看审批（GET /approval/list）、文档（POST /docs/{num}/comment）、GET /home、登录（POST /login）
  - (negative) 综合部·oa GET /approval/list 在 portal 中从未执行写操作（21 天、0 次）（封闭的写操作：登录（POST /login）、评论（POST /comment））

Reading (against the requirement's example): the 综合部 login node is isolated and names the
three addresses with their three bindings (192.168.1.21 still `jack` on day 21 although A2
borrowed `rose` from it on days 17–21: M18 / G4 kept the borrowed credential out), the closed
username set {jack, mike, mike.w, rose} (D2's rename included) and the `[a-z]{4}(\.[a-z])?`
grammar; its window 08:32–08:51 is D1's new window, and the extra 09:10–09:11 window is A2's
logins (flagged and incident-opened, but their minute was learned: open). The same login node's
parent states the department as one "某类人" part (E2) with the parent's (all-department)
constraints; GET /docs, done alike by three departments, is stated once per department (E1,
E2, E9). Finance's approval is "192.168.2.10 only" (M29, M29b). In the user view the
department composes its two learned roles (approver .21; report writers .23, .121) and states
"综合部 在 finance 中从未执行写操作 … 192.168.1.23 的尝试被判定为越权（未学习）": A1's write
was judged foreign and not learned, which is the requirement's "访问财务系统去审批就是异常".
The approval statements are `stale` on day 21: D3 renamed the routes on day 14 and the new
`/flow/...` nodes are still candidates (§16.10.8, D3).


#### 16.10.7 Variants, non-regression and resources

- **O-red seed 0** (29 min, peak RSS 1175.2 MB): recall by day d5 0.14, d7 0.15, d8 0.17, d10 0.42, d14 0.41, d21 0.41; precision by day d7 0.52, d8 0.30, d14 0.40, d21 0.59; ECE@14 0.20; false splits per system-month 0.0; anomalies 7/10 (missed: A2, A3, A4); GA login who = the truth IPs True, finance approver only True, ARI 0.89, portal bindings 23; FAR >= LOW / >= MEDIUM 0.01 / 0.0056
- **O60 seed 0** (17 min, peak RSS 1074.4 MB): recall by day d5 0.29, d7 0.25, d8 0.26; precision by day d7 0.47, d8 0.40

- **O60 (aggregated vs 60-s event mode, PG1 at day 8 within 0.05):** days 1–7 are the same
  aggregated stream as pack O and agree (day 7 recall 0.25 against pack O's 0.24, seed 0); on
  day 8, in 60-s event mode, recall is 0.26 against pack O's 0.21 on its aggregated day 8
  (difference 0.053, just outside 0.05) and precision 0.40 against 0.41. The truth sets of the
  two packs differ on day 8 (O60's last day closes its lineages), so the comparison is
  indicative, not a clean paired test; `compute_pgates` has no O60 check yet (open).
- **O-red** is reported, never tuned on; its false-split probe (20 attributes independent of
  every truth constraint) gives the PG2 false-split figure above.
- **O-real (PG11)** was not run: with R8 the pack is 35 days (≈ 50 min of wall time, over the
  ~40-min budget of a single run in this environment).
- **PG9** (packs A, E seed 0, full vs bounded) was measured by the adaptation owner on the
  final B-library code of this round (§16.9); the evaluator's changes do not touch the
  B-library or the bounded paths, and packs A and E in full mode reproduce round 4 exactly.
- **PG4** points are the 7-day points of §16.9 (`reports/progressive/scale7/`); the evaluator's
  changes add no per-event work (E1 reads ≤ 64 signatures per group part at rendering time).
- **Full suite** on the final tree: 2 881 passed, 4 skipped (21 min).
- **Per run:** 29–31 min wall (seeds 0–4; O-red 28 min) with three runs in parallel on four cores, peak RSS 1.26–1.30 GB (O-red 1.18 GB, O60 1.07 GB); ≈ 261 000 behaviour events per pack-O run (582 anomalous).


#### 16.10.8 Diagnosis of what still fails (by engine)

1. **PG1 recall (target 0.90).** The remaining misses on seed 0 at day 14 (`ev/diag/recall_*`
   driver, per truth pattern, the candidate statements and which component fails):
   - *Mail (4 patterns, P04)*: opaque TLS whose departments differ only in their 20–40-minute
     windows; the tree now splits mail (M37/M38) but the nodes are /16 or mixed-department, so
     who and when both miss. Needs per-department mail nodes (time-only separation).
   - *Login closed sets (P07, 3 patterns)*: the truth requires the usernames' closed set;
     finance's 3 users have 16 decayed value observations on day 14 (< `CLOSED_N` = 20, U 0.029 >
     `CLOSED_U` = 0.02) and close by day 21; 销售部's 20 usernames exceed `TEXT_VALUES_K` = 16,
     so a 20-value closed set can never be stated (a capacity decision, not a defect).
   - *Required keys not stated (P05/P07, crm visit, portal comment)*: `body.keys` is not a P05
     target at those nodes, so P07 has no key-set summary to fit (`request_targets` only asks for
     attributes P05 rated `split` / `target`).
   - *17:00 report (P04)*: confirmed on day 18 (N_CONF = 20 decayed units; two reporters, one
     report a workday) — after PG1's day-14 check although the truth is eligible from 20
     opportunities.
   - *Portal (4 patterns)*: windows (normal arrival law over 07:00–23:00) give IoU 0.68 on
     non-workdays; POST /comment's band from 1 observation.
   - *DEV oa login*: the truth's 60-name closed set.
2. **PG2 calibration (ECE ≈ 0.30).** Statement confidence is P04's held-out test-pass frequency
   (M46): mean stated 0.39 against a 0.52 evaluator hold rate on seed 0 day 14 (under-confident).
   Of the failing statements, `when` fails in 20 of 28: a truth STEP row's window is the
   activity's window, while the step's arrivals start minutes later (finance approve: truth
   10:00–11:32, arrivals 10:22–11:26); the evaluator draws held-out minutes uniformly in the truth
   window. Evaluator and engine measure different distributions (ptree open issue, kept).
3. **A4 (03:05 login, missed on seed 3) and A2 (borrowed credential, missed on seed 4).** A2 needs the 综合部 login node's three bindings before day 17; on seed 4 that node is split off late (day 21: two of the three addresses, n = 4–5 logins, no binding confirmed) — the who split of the login node is still evidence-limited on some seeds (P04). For A4: P03's when p-value is the HDR p of the
   empty slot under the covering node's density, which floors at ≈ α / (N + α): the young 综合部
   login node (N ≈ 40) gives ≈ 0.02, the /login route node (N ≈ 400–900) ≈ 1e-3 — the finding
   bound. Detection therefore depends on which node covers the event on the day. A
   distribution-free distance bound (Cantelli: P(|X − μ| ≥ d) ≤ σ² / (σ² + d²); 5.5 h from a
   window of σ ≈ 6 min gives 3e-4) would make it robust; proposed, not implemented (it changes
   the when finding's FAR on every node and needs its own measurement).
4. **PG5 D3 (approval route rename).** The `/flow/...` nodes stay candidates: one source
   alone never establishes an action (P10 ActionLedger, protection against probes), so P03 damps
   .21's renamed routes as `new_action` every day and their evidence (× 0.1) never reaches
   N_CONF. A rename (old routes stop the day the new ones start, same source, same shape) is a
   recognisable successor; needs a design decision.
5. **PG5 D4 / D5.** 2–4 LOW incidents, mostly B-library `alarm` incidents (B26/B27), one 22:15
   portal login LOW.
6. **PG10 day 11 / 21.** The GA login statement fails the band (1–2.5 KB displayed; the day-21
   window and grammar pass) and the 0.5–3 KB range: the 综合部 login node is created on day 10–15
   and starts with an empty NumSummary, so logins below 1 KB before the split are not in its range
   (content_time open issue 1: seed the child's numeric summaries from the parent's per-branch
   statistics).
7. **PG8 (0.83).** Finance's P08 arm: the username pair is screened only after day 7–14
   (3 users, 1 login a day each), so its measured gain is 0 at the decision time.
8. **PG4 per-event cost.** P03 scoring and P04 learning per event 2–10× over target
   (adapt_cost open issues: batch the target-summary updates, score per covering node).

#### 16.10.9 Open issues for the owners (round 2 → round 3)

- **P04 (tree):** a split child starts with empty numeric summaries (the 0.5–3 KB range of the
  综合部 login node misses the logins before the split); the login who split is still late on
  some seeds (seed 4); per-event learning cost (batch the target-summary updates). Suspect
  sources are never cleared while they keep sending rows (pnode `WhoSummary.sus`): kept as
  designed by M29 (a slow poisoner whose damping P03 stops must not become a member), so a
  member falsely damped once stays out of a node's who — needs a rule that distinguishes the
  two (e.g. P03 accepting the source as a member through its group's colleagues on N days).
- **P03 (conformity):** the when p-value's floor α / (N + α) on young nodes (A4); per-event
  scoring cost. `pbounds.p_value` still returns 1e-9 at the observed extremes of integer or
  log-transformed attributes — P03 is its only consumer and floors it (G2), so no other effect.
- **P05 / P07:** `body.keys` is not requested where P05 does not rate it a split/target, so
  required keys are missing on some POST actions; `TEXT_VALUES_K` = 16 caps closed sets.
- **P10 / P03:** a single source's route rename (D3) is never adopted (new-action damping).
- **P11:** the 研发 DHCP pool is not one group; the finance approver has no learned group
  (E9 covers the department part, not the group view).
- **P08:** finance's username pair is screened only after day 7–14 (PG8 P08 arm, bindings 6/6
  at day 14).
- **Evaluator:** the truth's step windows are the activity's windows (held-out arrivals for
  later steps start too early; most `when` precision failures); no O60 agreement check in
  `compute_pgates`; PG1 eligibility (20 raw opportunities) and P04's N_CONF (20 decayed units)
  disagree for daily two-person actions (the 17:00 report is eligible before it can confirm).
- **B28 governor:** trust = 0 while any incident is open stops all learning from a source after
  one false finding (G1 removed the false findings that triggered it on pack O).
- **Lead decisions:** class-shared decision-chain rings (§16.9 A5, off); a successor rule for
  renamed routes (D3); the when distance bound (A4); `TEXT_VALUES_K`.


### 16.11 Round 3: results on the final code (evaluator, 2026-10-02)

Everything here is measured on ONE code state: HEAD 2c7d335 (round 2) plus the four owners'
round-3 changes (P04 / P05 / P02: R3-1 – R3-14; P06 – P10: R1 – R5; P03 / P11 / P14 / B28:
G8 – G15; generator, scorer and cost: E-T1 – E-T3, E-C1 – E-C2) plus the evaluator's changes
V1 – V9 below. Runs: pack O seeds 0–4, 21 days at 900 s, registry `progressive_decision`
(the pack's default), `--no-series`, resumable with a checkpoint every ~10 minutes of wall
time (§11.6; one run was resumed after the disk filled, see 16.11.7); O-red and O60 seed 0;
packs A and E seed 0 with the default registry. Seeds 0–1 were used for development, seeds
2–4 are held out. The evaluator's engine changes (V2, V3, V5, V6) were derived on seeds 0–1
(seeds 2–3 were run once on an intermediate snapshot); the groups / views owner's G14 and G15 came
from its seed-2 confirmation run, and one scorer change (V8, the example checklist's choice of
statement) was found on seed 3. Seed 4 was never looked at before the final runs.

**Before / after on the same scorer.** This round changed the generator's truth and the
scorer (E-T1, E-T2: each step's own arrival law and the traffic mix of held-out events; V1:
TLS framing in the truth; V4: hard ranges against what the data could show; V7: windows at the
coverage they state; V8 – V9: the example checklist's statement choice and PG1's 6/6 bindings check). So that the comparison measures the engines only, "before" is the
round-2 runs (the round-2 FINAL pickles of §16.10) with their truth regenerated by this
round's generator (same seeds; the traffic itself is unchanged — the generator changes are
truth bookkeeping only) and re-scored by this round's scorer. The round-2 figures as
published in §16.10 are quoted where they matter. Medians are over the 5 seeds with
[min–max]; per-seed results are in `reports/progressive/runs/O_<seed>.json`, the round-2 runs
as published in `reports/progressive/round2/runs/`, and the re-scored round-2 runs in
`reports/progressive/round2/rescored/`.

Full suite on the final tree: 2 955 passed, 4 skipped (2 tests failed in the full run only
with `OSError: [Errno 28] No space left on device` while the disk was full, and pass when re-run).
Packs A and E seed 0 with the default registry are identical, key for key apart from timings,
to `reports/round4/runs/{A,E}_0_full.json`.

#### 16.11.1 Deviations made in round 3

Every row has a regression test that fails without it (owners' files: `tests/engines/test_p04_round3.py`,
`test_p07_value_capacity.py`, `test_p08_pooling.py`, `test_p09_window_acceptance.py`, `test_p10_rename.py`,
`test_p06_requests.py`, `test_p03_round3.py`, `test_p11_round3.py`, `test_p14_round3.py`,
`test_b28_trust_scope.py`, `tests/eval/test_orggen.py`, `test_pmetrics.py`, `test_resumable.py`,
`tests/lib/test_pevent_get_equivalence.py`, `test_phier_shape_equivalence.py`; evaluator:
`tests/engines/test_eval3_fixes.py`, `tests/eval/test_orggen.py::test_tls_truth_states_the_observed_upstream_bytes`,
`tests/eval/test_pmetrics.py::test_pg1_range_is_judged_on_what_the_emitted_data_could_show`), each checked
against the pre-change files (evaluator: also `tests/eval/test_progressive_report_label.py::test_checklist_reads_the_most_specific_statement_of_the_department`, `test_pmetrics.py::test_pg1_window_is_compared_at_the_coverage_it_states`, `::test_login_bindings_ask_for_what_could_be_learned_by_the_day`).

R3-1 – R3-14 (P04 pattern tree, P05 selection, P02 registry):

| # | Where | Was | Changed to | Why (measured) |
|---|---|---|---|---|
| R3-1 | `pnode.SplitRows`, P04 `_keep_row` / `_inherit_rows` | split children start empty | each learning leaf keeps ≤ 128 rows (weighted to recent), replayed into the child each row belongs to at a split (size digest and daily min/max, text / set / categorical summaries, arrivals, source summary); long text not kept | the 综合部 login node's range began at the split (logins before it missing) |
| R3-2 | P04 `_note_extremes` / `_inherit_extremes`, `NumSummary.seed_extreme` | – | per split-candidate value the exact min / max of each numeric field and its day; the child's range starts from them | a rare small value is easily missing from a 128-row sample |
| R3-3 | P04 `_do_split` | a source-split child's observations since the parent started learning | from the leaf's source summary over its whole life | – |
| R3-4 | P04 `_keep_route_row` / `_replay_route_rows` | a route node starts learning empty | it receives the ≤ 8 rows it waited for | the first two days of every route were lost; the 17:00 report had no statement by day 14 |
| R3-5 | `pnode.Node.n_obs` | confirmation on a decayed sum (N_CONF = 20 decayed units) | 20 undecayed observations on ≥ 3 dates | 20 reports over two weeks counted ~17 |
| R3-6 | `pnode.WhoSummary` mark / observe / support | a suspect source stays suspect while it sends rows (M29) | a sequential evidence test: +4.64 bits per row damped as foreign, −4.64 per row supported by colleagues (its /24 or learned group at the node), cleared at ≤ −2 bits; repetition alone never clears | a member falsely damped once stayed out of the node's who; A9 slow poisoning still kept out |
| R3-7 – R3-9 | P04 `_hold_check` / `_hold_constraints` | shape-only values compared as text; exact band comparison; unstated attributes checked | shape-only values checked as an instance of their shape; bands tolerate float noise; attributes the views never state are not checked | password checks passed 0 of 289; a constant 409 B request failed a band of 409.00000000000017 |
| R3-10 | `pnode.fit_hold_prior` / `hold_kind`, P04 `_hold_priors` | flat Beta(1, 1) per node | the prior of a node's pass count fitted daily across all trees from statements with a similar number of constraints | median confidence fell with time |
| R3-11 – R3-13 | `pselect` | constant key sets skipped (assumed P04 invariants); unrecorded values counted; client properties ranked with the action's fields | constant form key sets per action get the target role (≤ 2 extra node slots); unrecorded probe values not counted; client properties rank after the action's own fields | login / CRM visit / portal forms never got their required keys stated |
| R3-14 | `psketch.WeightedReservoir.offer_lazy` | – | a row is built only when it enters the sample | cost of R3-1 |

R1 – R5 (P06 – P10, content / time / workflow):

| # | Where | Was | Changed to | Why (measured) |
|---|---|---|---|---|
| R1 | P07 `pgrammar.adapt_values` | fixed 16-value sketch (`TEXT_VALUES_K`) | the sketch doubles (≤ 64) when the values seen stop growing while it keeps evicting; closure judged on post-growth arrivals; a young child takes its ancestor's capacity; members must be clean and match the stated grammar | 销售部's 20 user names could never close; A3's injected name entered a grown set in an intermediate run |
| R2 | P08 `pfd` / `binding` pooling | n_x ≥ 5 at the node | source-level clean days across the tree, seeded from the probe rows before the sketch existed (≤ 64 sources per pair seen on ≥ 2 days, kept 7 days while untracked), leave-one-out empirical-Bayes prior | finance's three users bound only on a probe day (seed 1: day 15) |
| R3 | P10 `pdfg` route-rename adoption | a renamed route was a new action | a new route replacing an established one of the same source (old route stopped, one literal segment changed, same workflow position, 2 dates) takes the old action id; `successor_candidate()` exposed for P03 | D3: /approval/ → /flow/ never adopted |
| R4 | P09 `pwindows` | single-source windows accepted like others | at a multi-source node a window supported by one source needs `REGIME_SINGLE_DATES` (5) dates | A2's 09:10 logins became a login window (IoU 0.86 → 0.90–0.95) |
| R5 | P06 / P07 content requests | requests could target ctx / ev / sess; transport before payload | derived context attributes are never content targets; the payload is requested before its transport measures; a quantity measured twice takes one slot | the portal comment body size was never fitted |

G8 – G15 (P03 conformity, P11 groups, P14 views, B28 governor):

| # | Where | Was | Changed to | Why (measured) |
|---|---|---|---|---|
| G8 | P03 `when` p of an empty slot (`pscore`) | the covering node's HDR p (floors at α / (N + α)) | `max(p_hier, min(p_own, p_parent))` recursively up the tree; p_hier = the node's counts under an empirical-Bayes Dirichlet prior from the parent | A4 (03:05 login) missed on seeds 0–2 once the young-node false 'below range' signal was gone; p_hier alone raised FAR (+6/+10 portal incidents ≥ LOW on day 5) |
| G9 | B28 `_incident_trust` | trust = 0 while any incident is live | factor 1 when every live incident's evidence is pattern-scoped (P03 findings ≤ MEDIUM on ONE learned pattern + conformity-family alarms), else 0; quarantine unchanged | one false finding stopped all learning from a source |
| G10 | P11 pools (`_pool_nets`, `plouvain.merge_local`) | 研发 leases split over 5–10 groups | configured `dhcp_scopes` and learned turnover prefixes (≥ 50 % ephemeral, ≥ 8 addresses, outside `ip_classes`) are pools: everything inside merges into one group covering the prefix | PG3 DEV pool grouped 0/5 in round 2 |
| G11 | P11 `_roles` | singletons ungrouped | a stable address holding ≥ 50 % of an action's org-wide mass that joins no group is a one-address role group | the finance approver was never grouped |
| G12 | P11 `_inherit_ids` | Hungarian matching on all sources | matched on addresses; dissolved ids reused within 7 days; a merged group keeps its largest predecessor's id; sticky display names; DHCP scope names never override a department | 综合部 changed id on day 17/20 (a pool-only group won the tie) |
| G13 | P14 views | – | the department of an IP-level who is named; the action word follows the page; a binding is stated when it distinguishes its sources (BIND_MIN_CARD alone dropped finance); a group view states the group's part of a shared node; negatives consistent with the group's own actions; per-system "never does" statements composed for departments | the operator's "某类人 … 做什么" wording |
| G14 | P03 `_cross_up` | forward binding checked at the covering node only | a value an ancestor's record binds to another source counts as `cross_binding` | A2 missed on seed 2 after a who split of the 综合部 login node |
| G15 | P03 labels / pool standing, P11 `lineage` | group labels checked as-is | `system_new` / new-label tested through the group's lineage; a pool group's standing from group-level evidence; configured-pool members qualify without a colleague | 6 MEDIUM who incidents of 研发 leases in OA (seed 2, days 11–18) |

E-T1 – E-C2 (generator, scorer, cost; truth / cost / scale owner):

| # | Where | Was | Changed to | Why (measured) |
|---|---|---|---|---|
| E-T1 | `orggen.step_arrival_law` | a step's truth window = the activity's window widened by think-time sums; held-out minutes uniform in it | each step has its own arrival law (simulated sessions) published as quantiles; windows = its central 99 % (§11.5) | mail records 2–8: KS 0.23 between held-out minutes and generated arrivals |
| E-T2 | `pmetrics.holdout_events`, `PTruth.traffic` | every truth row of a route an equal share | held-out events follow the emitted traffic mix (rows, day types, sources; last 14 days, by lineage) | mail: 25 % per department while 销售部 + 研发 send ~90 % |
| E-T3 | `eval/resumable.py` | – | checkpointed `run_pack` (identity sentinels by reference) | O-real (35 d) and the 20k-IP / 300-system PG4 points exceed one slot |
| E-C1 | `pevent.EventBatch.get` | per-call column lookup | per-column list built once | 423 → 142 ns per call (≈ 6 M calls / 3 days) |
| E-C2 | `phier._shape` | per-character | per alphanumeric stretch, same output | 789 → 112 µs per 2 KB padding |

V1 – V6 (evaluator, this section's runs):

| # | Where | Was | Changed to | Why (measured) |
|---|---|---|---|---|
| V1 | `orggen` truth (`TLS_UP_FRAMING`) | a TLS row's `net.bytes_up` band / range stated on the payload | on the observed quantity (payload + 300 B framing), as the held-out sampler already did | learned mail / git bands were exactly truth + 300 B: 5 content misses per seed no engine could fix |
| V2 | P14 `node_statement` (part confidence) | a part with its own windows stated the min of its parts' NOMINAL coverages | min(node p_hold, the part's own window coverage) | mail parts stated 0.85–0.97 against node hold rates of ~0.5; /docs parts stated 0.097 (a workflow edge's dependency strength) and held 1.0 — both ends of PG2's reliability diagram |
| V3 | `pfd.binding_stated` shared by P14 and P04 `_hold_constraints` | P04's held-out test checked every bound pair | only the pairs the views state (identifier-like payload or sources bound to different values) | constant-of-action pairs (a client-version header → one value) were tested but never stated |
| V4 | `orggen` `ptruth.extremes`, `PTruth.observable`; PG1 content, PG10, checklist | a learned hard range compared with the generator's support only | support OR the support cut to the emitted extremes up to the scored day (§11.5) | seed 0: no 综合部 login below 1 KB was ever generated; "100 % in 0.5–3 KB" was unrecoverable |
| V5 | P14 `group_parts`, `_restrict_closed` | no signature members at a node with an address context; a part stated the node's closed sets | signature members admitted when they satisfy the node's net.src context; a part's closed set = its members' P08-bound values when all are bound | the 销售部 part of the 财务部+销售部 login node listed 5 of 20 members and the node's 23 user names |
| V6 | P12 `fitted_gain`, `bindings_pending` | any judged P08 pair (gain 0 included) measured the arm | constant pairs (one value across sources) are no measurement; the arm stays unmeasured while a pair of ≥ 2 recurring sources with different values is still gathering n_bind | finance's `net.src → body.fmt` (gain 0) switched P08 off on day 10, before its three users' 5th login (bindings 0.67 at day 14) |
| V7 | `pmetrics.when_compatible`, `law_windows` (PG1 'when') | learned windows compared (IoU ≥ 0.7) with the truth's windows = the central 99 % of the step's arrival law | also with the law's central interval at the coverage the statement states (0.5 ≤ c < 0.99), accepting either | the portal's normal arrival law over 07:00–23:00: a correct 89 % window [09:32, 20:04] had IoU 0.66 against the 99 % window [07:00, 23:01] (0.82 against the law's central 89 %); PG1 'when' failed for the portal on every seed whatever the engine learned |
| V8 | `scripts/progressive_report.py` checklist | the first statement with the best who Jaccard | among equal who, the address-level and deeper statement | seed 3 read the department's part of the all-department login node (window spanning D1, IoU 0.34) although the 综合部 node's own statement existed |
| V9 | `pmetrics.login_bindings` (PG1 "GA + FIN bindings 6/6 at day 14") | the bindings of the segment valid on the day | a pair whose value changed recently (< 5 events on 2 dates under the new value, P08's rename evidence) also accepts the previous value | D2 renames 10.168.7.121's user on day 13; the check asked for the new name on day 14 — 5/6 on every seed — while PG5 D2 gives the rename its own latency |

Scoring changes (V1, V4, V7 – V9) apply to "before" and "after" alike (16.11.2). V2 / V3 / V5 / V6 are
engine changes; their tests are in `tests/engines/test_eval3_fixes.py`. Cross-owner open issues
of the owners resolved here: the TLS truth (content owner), the part confidence and the
held-out test's bindings (tree owner), P12's judging of the binding arm (content owner),
`tests/eval/test_content_integration.py` (finance bindings now bound on day 7 by R2's pooling,
LB 0.98 — the test now asks for none on day 3 and all three on days 7 and 14).


#### 16.11.2 Gates (pack O, 5 seeds; status from `compute_pgates`)

| Gate | Status | Failing checks | Not measured |
|---|---|---|---|
| PG1 pattern recovery | fail | recall@14; precision@14; recall@14 (O-red); precision@14 (O-red); recall_when@14 (O-red); recall_content@14 (O-red); recall_bindings@14 (O-red); GA+FIN bindings 6/6 at day 14 (O-red) | – |
| PG2 convergence and calibrated confidence | fail | recall non-decreasing (±0.05) outside days 12-15; daily patterns: 80 % recall by day 7; weekly patterns: 80 % recall by day 21; mean depth non-decreasing before day 12; median confidence non-decreasing; median unseen-IP mass non-increasing; ECE | – |
| PG3 specificity reached | fail | GA login who = the 3 IPs | – |
| PG4 resources sublinear in IPs, metrics and servers | fail | memory slope vs systems (12 families); scoring p95 us/event; learning p95 us/event | – |
| PG5 drift adaptation latency | fail | D1; D3; D4; D5 | portal rate.ip_h p99 within 10 % of pre-A7 |
| PG6 anomaly detection of the requirement's examples | fail | B29 top reason = violated constraint | KS D of conf_* p on clean ticks |
| PG7 open schema | pass | – | noise dropped; constants -> invariants |
| PG8 scenario adaptation | fail | chosen arms in strategy truth (day 14); who level code length within 5 % of best | – |
| PG9 non-regression | not measured | – | comparison |
| PG10 views | fail | OA statement day 11 (before D1); OA statement day 21 | – |
| PG11 real-world robustness | not measured | – | – |

Per check, median [min–max] over the seeds: round 3 (after) against the round-2 runs re-scored by this round's scorer (before; same generator truth, same held-out law):

| Check | After (median [min–max], n=5) | Before (n=5) |
|---|---|---|
| PG1 recall @14 | 0.789 [0.737–0.816] | 0.684 [0.658–0.684] |
| PG1 precision @14 | 0.687 [0.667–0.721] | 0.690 [0.636–0.725] |
| PG1 who @14 | 0.921 [0.868–0.974] | 0.868 [0.842–0.895] |
| PG1 when @14 | 0.921 [0.895–0.974] | 0.895 [0.842–0.895] |
| PG1 content @14 | 0.868 [0.816–0.868] | 0.737 [0.711–0.763] |
| PG1 bindings @14 | 1.000 [1.000–1.000] | 0.667 [0.667–0.667] |
| PG1 workflow @14 | 1.000 [0.923–1.000] | 0.923 [0.923–0.923] |
| PG1 GA + FIN bindings (of 6) @14 | 6.000 [6.000–6.000] | 3.000 [0.000–3.000] |
| PG1 recall @21 | 0.690 [0.595–0.872] | 0.690 [0.619–0.769] |
| PG1 precision @21 | 0.809 [0.781–0.843] | 0.774 [0.702–0.836] |
| PG2 ECE @14 | 0.231 [0.180–0.259] | 0.344 [0.315–0.458] |
| PG2 ECE @21 | 0.343 [0.269–0.386] | 0.473 [0.367–0.540] |
| PG3 GA login who = 3 IPs | 4/5 | 4/5 |
| PG3 GA bindings (of 3) | 3.000 [3.000–3.000] | 3.000 [3.000–3.000] |
| PG3 finance approval = {.2.10} | 5/5 | 5/5 |
| PG3 portal login who level | 5/5 | 5/5 |
| PG3 DEV who ok | 5/5 | 5/5 |
| PG3 DEV pool grouped | 5/5 | 0/5 |
| PG3 ARI | 0.974 [0.974–0.974] | 0.974 [0.974–0.974] |
| PG5 D1 | 2/5 | 3/5 |
| PG5 D2 | 5/5 | 5/5 |
| PG5 D3 | 0/5 | 0/5 |
| PG5 D4 | 0/5 | 0/5 |
| PG5 D5 | 1/5 | 0/5 |
| PG5 .21 still jack | 5/5 | 5/5 |
| PG6 FAR inc >= LOW | 0.005 [0.004–0.008] | 0.005 [0.004–0.007] |
| PG6 FAR inc >= MEDIUM | 0.003 [0.003–0.007] | 0.004 [0.003–0.004] |
| PG6 pv >= LOW | 0.016 [0.009–0.017] | 0.016 [0.012–0.021] |
| PG7 type correct | 1.000 [1.000–1.000] | 1.000 [1.000–1.000] |
| PG8 arms in truth | 0.833 [0.667–1.000] | 0.833 [0.667–0.833] |
| PG8 max switches | 1.000 [1.000–1.000] | 1.000 [1.000–2.000] |
| PG10 day 11 | 0/5 | 0/5 |
| PG10 day 21 | 1/5 | 0/5 |
| PG10 finance single IP | 5/5 | 5/5 |
| PG10 GA negative (finance) | 5/5 | 5/5 |

#### 16.11.3 Progress with time (PG1 / PG2)

| PG1 (median over seeds) | day 3 | day 5 | day 7 | day 10 | day 14 | day 18 | day 21 |
|---|---|---|---|---|---|---|---|
| recall after | 0.00 | 0.35 | 0.42 | 0.73 | 0.79 | 0.79 | 0.69 |
| recall before | 0.00 | 0.25 | 0.32 | 0.65 | 0.68 | 0.66 | 0.69 |
| precision after | 1.00 | 0.54 | 0.58 | 0.77 | 0.69 | 0.79 | 0.81 |
| precision before | – | 0.48 | 0.66 | 0.76 | 0.69 | 0.75 | 0.77 |

Per seed, recall by day (7 / 10 / 14 / 18 / 21): s0 0.42 / 0.78 / 0.82 / 0.89 / 0.87; s1 0.47 /
0.63 / 0.74 / 0.76 / 0.74; s2 0.38 / 0.72 / 0.74 / 0.79 / 0.69; s3 0.48 / 0.76 / 0.82 / 0.80 / 0.69;
s4 0.42 / 0.73 / 0.79 / 0.76 / 0.60. Daily patterns reach 80 % recall on day 11 / 17 / 16 / 13 / 17
(round 2: never by day 21); weekly patterns never (PG2 "80 % by day 21" fails). The drop after
day 18 on seeds 1–4 is D3 (day 14: the approval pages move to `/flow/`; their three rows are
eligible from about day 19 and are never confirmed — 16.11.8 item 4), the 01:00 backup (eligible
on day 20 with 20 daily events, confirmed a day later, item 2) and seed-specific misses
(16.11.8).

The median stated confidence no longer falls with time (day 7 / 14 / 21: s0 0.58 / 0.54 / 0.50,
s1 0.45 / 0.59 / 0.62, s2 0.50 / 0.53 / 0.59, s3 0.55 / 0.60 / 0.67, s4 0.65 / 0.66 / 0.70: it rises
on four seeds; round 2 re-scored: 0.43–0.63 on day 7 falling to 0.34–0.40 on day 21). PG2's strict
"non-decreasing" check still fails on day-to-day dips around the drift days.


#### 16.11.4 The anomalies of the requirement (PG6; finding types per seed)

| Anomaly | s0 | s1 | s2 | s3 | s4 | before s0/s1/s2/s3/s4 |
|---|---|---|---|---|---|---|
| A1 | yes who | yes who | yes who | yes who | yes who | y/y/y/y/y |
| A2 | yes content | yes content | yes content | yes content | yes content | y/y/y/y/n |
| A3 | yes content,when | yes content,when | yes content,when | yes content,when | yes content,when | y/y/y/y/y |
| A4 | yes when | yes when | yes when | yes when | yes when | y/y/y/n/y |
| A5 | yes seq | yes seq | yes seq | yes seq | yes seq | y/y/y/y/y |
| A6 | yes novel | yes novel | yes novel | yes novel | yes novel | y/y/y/y/y |
| A7 | yes content,seq | yes content | yes content,seq | yes content,seq | yes content,seq | y/y/y/y/y |
| A8 | yes content,who | yes content,who | yes content,who | yes content,who | yes content,who | y/y/y/y/y |
| A9 | yes who | yes who | yes who | yes who | yes who | y/y/y/y/y |
| A10 | yes content,seq,who | yes seq,who | yes content,seq,who | yes content,seq,who | yes content,seq,who | y/y/y/y/y |

detected 50/50 (after); 48/50 (before)

#### 16.11.5 The requirement's example: checklist per seed (day 21)

| Clause (truth valid on day 21) | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| 综合部 3 个 IP 登录 OA（who = 192.168.1.21、192.168.1.23、10.168.7.121） | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 | ✗ who Jaccard 0.67 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 |
| 工作日登录时间窗 08:30–08:51 | ✓ IoU 0.90 | ✓ IoU 0.95 | ✗ IoU 0.00 | ✓ IoU 0.72 | ✗ IoU 0.45 |
| 提交数据量 90 % 在 1–2 KB | ✓ band [1024.0, 2048.0] | ✓ band [1024.0, 2048.0] | ✗ band None | ✗ band [1024.0, 3072.0] | ✗ band [512.0, 1536.0] |
| 100 % 在 0.5–3 KB | ✓ range [1024.0, 2662.4] | ✗ range [1024.0, 2867] | ✗ range None | ✓ range [512.0, 3072.0] | ✓ range [614, 2048.0] |
| 提交内容含 username=，取值不超过 10 个字符 | ✓ grammar [a-z]{4}(\.[a-z])? | ✓ grammar [a-z]{4}(\.[a-z])? | ✗ grammar None | ✓ grammar [a-z]{4}(\.[a-z])? | ✓ grammar [a-z]{3,8}(\.[a-z])? |
| 绑定 10.168.7.121→mike.w、192.168.1.21→jack、192.168.1.23→rose | ✓ 3/3 | ✓ 3/3 | ✗ 0/3 | ✓ 3/3 | ✓ 3/3 |
| 192.168.1.21 访问业务审批页面 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 |
| 下午 5 点提交报告（.23、.121） | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 |
| 财务系统审批只有财务部 192.168.2.10 访问 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 | ✓ who Jaccard 1.00 |
| 用户视角：综合部在财务系统中从未执行写操作（否定陈述） | ✓ present | ✓ present | ✓ present | ✓ present | ✓ present |


Reading: 40 of 50 clause checks pass, as in round 2 (40 of 50 as published), but distributed
differently: seed 0 now holds the whole example (10/10; round 2: 8), seed 4 gained the who and
the bindings (8; round 2: 6), seed 2 lost the login clauses (4; round 2: 9). Seed 1 misses only
the hard range: the generator drew one 643 B login of 192.168.1.21 on day 2, before the route
node of `/login` existed, and no node ever saw it (the first rows of a route are lost, 16.11.8
item 2); on seed 0, where no login below 1 KB was generated at all, the learned range
[1 053, 2 616] B is the data's own range and passes (V4). Seed 2: the login node splits on the
learned-group level and puts the one-address approver group of 192.168.1.21 (`G13`, ~1 login a
day) into the `other` branch with 销售部, so the 综合部 node names .23 and .121 only (item 1).
Seed 3: the address-level 综合部 node states D1's window plus A2's 09:10–09:14 (IoU 0.72) and a
band whose fitted upper edge is 2.85 KB from 41 logins (truth 2 KB). Seed 4: no address-level
综合部 node by day 21; the department's part of the all-department login node is the most
specific statement (who and bindings pass; its window spans D1, IoU 0.45, and its band is the
mixed node's).


#### 16.11.6 The example as learned (seed 0, day 21, both views; P14's text verbatim)

**System view, OA (statements naming a 综合部 address):**

- `stable`, confidence 0.31: 【oa】工作日 08:32–08:51（覆盖 74 %，6 个工作日），综合部（10.168.7.121、192.168.1.21、192.168.1.23）访问 POST /login（登录）：viewstate.len= 90 % 在 1000–2000，全部在 800–2600（n = 39，下次越界概率 ≤ 5.0 %）；提交数据量 90 % 在 1–2 KB，全部在 1–2.6 KB（n = 39，下次越界概率 ≤ 5.0 %）；下行字节 90 % 在 320–580 B，观测范围 300–600 B（n = 21）；上行字节 90 % 在 1.5–3 KB，观测范围 1.4–3.2 KB（n = 23）；时长 90 % 在 10–70，全部在 10–120（n = 46，下次越界概率 ≤ 4.3 %）；每 IP 每小时次数 90 % 在 1–1；body.fmt 取值 `[a-z]{4}`，body.fmt 取值集合封闭 {form}；表单键必含 captcha=、csrf=、password=、username=、viewstate=；password= 取值 `[A-Za-z0-9]{7,30}`；password.len= 取值集合封闭 {10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 8.0, …}；username= 取值 `[a-z]{4}(\.[a-z])?`，username= 取值集合封闭 {jack, mike, mike.w, rose}；viewstate= 取值 `[A-Za-z0-9\-_]{511,4094}`；请求头 content-type 取值 `[a-z]{11}/[a-z]\-[a-z]{3}\-[a-z]{4}\-[a-z]{10}`，请求头 content-type 取值集合封闭 {application/x-www-form-urlencoded}；username=jack 只来自 -|chrome/126|win|128|w16；username=mike 只来自 -|edge/125|win|128|w16；username=mike.w 只来自 -|edge/125|win|128|w16；username=rose 只来自 -|chrome/126|win|128|w16；绑定：10.168.7.121 → username=mike.w、192.168.1.21 → username=jack、192.168.1.23 → username=rose（g3 = 0.00，各 ≥ 5 次）。流程：POST /login → GET /home（间隔 0 秒–4 秒）。置信 0.31 · 首次 2025-09-09 · 最近 2025-09-21 · v1.0
- `stable`, confidence 0.68: 【oa】工作日 08:32–09:20（覆盖 88 %，8 个工作日），综合部（10.168.7.121、192.168.1.21、192.168.1.23）访问 POST /login（登录）：viewstate.len= 90 % 在 500–1500，全部在 400–2600（n = 674，下次越界概率 ≤ 0.3 %）；提交数据量 90 % 在 0.5–1.5 KB，全部在 0.6–2.6 KB（n = 674，下次越界概率 ≤ 0.3 %）；下行字节 90 % 在 320–580 B，全部在 300–600 B（n = 603，下次越界概率 ≤ 0.4 %）；上行字节 90 % 在 1–2 KB，全部在 1–3.2 KB（n = 645，下次越界概率 ≤ 0.4 %）；时长 90 % 在 10–100，全部在 5–180（n = 700，下次越界概率 ≤ 0.3 %）；每 IP 每小时次数 90 % 在 1–1；body.fmt 取值 `[a-z]{4}`，body.fmt 取值集合封闭 {form}；表单键必含 captcha=、csrf=、password=、username=、viewstate=；password= 取值 `[A-Za-z0-9]{7,30}`；password.len= 取值集合封闭 {10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 8.0, …}；username= 取值 `[a-z]{3,8}(\.[a-z])?`；viewstate= 取值 `[A-Za-z0-9\-_]{511,4094}`；请求头 content-type 取值 `[a-z]{11}/[a-z]\-[a-z]{3}\-[a-z]{4}\-[a-z]{10}`，请求头 content-type 取值集合封闭 {application/x-www-form-urlencoded}；请求头 x-client-ver 取值 `[0-9]\.[0-9]\.[0-9]`，请求头 x-client-ver 取值集合封闭 {5.2.1}；net.pkts_up 取值集合封闭 {11.0, 2.0, 3.0, 4.0}；username=amy 只来自 -|firefox/127|linux|64|w15；username=brandon 只来自 -|chrome/126|win|128|w16；username=brian 只来自 -|edge/125|win|128|w16；username=carol 只来自 -|chrome/126|win|128|w16；username=cdthc 只来自 -|firefox/127|linux|64|w15；username=david 只来自 -|chrome/126|win|128|w16；username=emma 只来自 -|chrome/126|win|128|w16；username=frank 只来自 -|chrome/126|win|128|w16；绑定：10.168.7.121 → username=mike.w、192.168.1.21 → username=jack、192.168.1.23 → username=rose、192.168.2.10 → username=lucy、192.168.2.11 → username=tom、192.168.2.12 → username=kate、192.168.3.20 → username=amy、192.168.3.21 → username=brian（g3 = 0.00，各 ≥ 5 次）。流程：POST /login → GET /home（间隔 0 秒–4 秒）。置信 0.68 · 首次 2025-09-01 · 最近 2025-09-21 · v2.0
- `stale`, confidence 0.45: 【oa】工作日 09:34–11:29（覆盖 96 %，8 个工作日），综合部（192.168.1.21）访问 POST /approval/{num}/approve（审批）：viewstate.len= 90 % 在 800–1400，全部在 700–1500（n = 46，下次越界概率 ≤ 4.3 %）；提交数据量 90 % 在 0.8–1.4 KB，全部在 0.8–1.5 KB（n = 47，下次越界概率 ≤ 4.2 %）；下行字节 90 % 在 2–18 KB，全部在 1–20 KB（n = 55，下次越界概率 ≤ 3.6 %）；上行字节 90 % 在 1.4–2 KB，全部在 1.3–2 KB（n = 42，下次越界概率 ≤ 4.7 %）；时长 90 % 在 10–100，全部在 10–180（n = 55，下次越界概率 ≤ 3.6 %）；net.pkts_down 90 % 在 4–16，全部在 2–16（n = 55，下次越界概率 ≤ 3.6 %）；每 IP 每小时次数 90 % 在 1–5；body.fmt 取值 `[a-z]{4}`，body.fmt 取值集合封闭 {form}；表单键必含 id=、opinion=、sign=、viewstate=；opinion= 取值 `[\u0080-\U0010ffff]{2,8}`，opinion= 取值集合封闭 {同意, 同意，请尽快办理, 退回}；viewstate= 取值 `[A-Za-z0-9\-_]{511,2046}`；请求头 content-type 取值 `[a-z]{11}/[a-z]\-[a-z]{3}\-[a-z]{4}\-[a-z]{10}`，请求头 content-type 取值集合封闭 {application/x-www-form-urlencoded}；请求头 user-agent 取值 `[A-Z][a-z]{6}/[0-9]\.[0-9] \([A-Z][a-z]{6} [A-Z]{2} [0-9]{2}\.[0-9]; [A-Z][a-z]{2}[0-9]{2}; [a-z][0-9]{2}\) [A-Za-z0-9]{11}/[0-9]{3}\.[0-9]{2} \([A-Z]{5}, [a-z]{4} [A-Z][a-z]{4}\) [A-Z][a-z]{5}/[0-9]{3}\.[0-9]\.[0-9]\.[0-9] [A-Z][a-z]{5}/[0-9]{3}\.[0-9]{2}`。置信 0.45 · 首次 2025-09-01 · 最近 2025-09-12 · v1.0（近期未出现）
- `stale`, confidence 0.45: 【oa】工作日 09:32–11:28（覆盖 96 %，8 个工作日），综合部（192.168.1.21）访问 GET /approval/{num}（查看审批）：下行字节 90 % 在 10–40 KB，全部在 5–40 KB（n = 40，下次越界概率 ≤ 4.9 %）；上行字节 90 % 在 505–506 B，观测范围 505–506 B（n = 10）；时长 90 % 在 10–100，全部在 5–180（n = 57，下次越界概率 ≤ 3.5 %）；net.pkts_down 90 % 在 15–30，观测范围 12–30（n = 10）；每 IP 每小时次数 90 % 在 0–4；请求头 user-agent 取值 `[A-Z][a-z]{6}/[0-9]\.[0-9] \([A-Z][a-z]{6} [A-Z]{2} [0-9]{2}\.[0-9]; [A-Z][a-z]{2}[0-9]{2}; [a-z][0-9]{2}\) [A-Za-z0-9]{11}/[0-9]{3}\.[0-9]{2} \([A-Z]{5}, [a-z]{4} [A-Z][a-z]{4}\) [A-Z][a-z]{5}/[0-9]{3}\.[0-9]\.[0-9]\.[0-9] [A-Z][a-z]{5}/[0-9]{3}\.[0-9]{2}`；请求头 x-client-ver 取值 `[0-9]\.[0-9]\.[0-9]`。置信 0.45 · 首次 2025-09-01 · 最近 2025-09-12 · v1.0（近期未出现）
- `stale`, confidence 0.45: 【oa】工作日 09:28–11:25（覆盖 96 %，8 个工作日），综合部（192.168.1.21）访问 GET /approval/list（查看审批）：下行字节 90 % 在 6–19 KB，全部在 4–20 KB（n = 59，下次越界概率 ≤ 3.4 %）；上行字节 90 % 在 505–505 B，观测范围 505–505 B（n = 11）；时长 90 % 在 20–100，全部在 10–140（n = 59，下次越界概率 ≤ 3.4 %）；net.pkts_down 90 % 在 5–15，全部在 6–16（n = 42，下次越界概率 ≤ 4.7 %）；每 IP 每小时次数 90 % 在 0–4；请求头 user-agent 取值 `[A-Z][a-z]{6}/[0-9]\.[0-9] \([A-Z][a-z]{6} [A-Z]{2} [0-9]{2}\.[0-9]; [A-Z][a-z]{2}[0-9]{2}; [a-z][0-9]{2}\) [A-Za-z0-9]{11}/[0-9]{3}\.[0-9]{2} \([A-Z]{5}, [a-z]{4} [A-Z][a-z]{4}\) [A-Z][a-z]{5}/[0-9]{3}\.[0-9]\.[0-9]\.[0-9] [A-Z][a-z]{5}/[0-9]{3}\.[0-9]{2}`；请求头 x-client-ver 取值 `[0-9]\.[0-9]\.[0-9]`。置信 0.45 · 首次 2025-09-01 · 最近 2025-09-12 · v1.0（近期未出现）
- `confirmed`, confidence 0.45: 【oa】工作日 17:01–17:14（覆盖 91 %，11 个工作日），综合部（10.168.7.121、192.168.1.23）访问 POST /report/generate（提交报告）：提交数据量 90 % 在 20–55 KB，观测范围 20–60 KB（n = 24）；下行字节 90 % 在 2.2–4.4 KB，观测范围 2–4.5 KB（n = 15）；时长 90 % 在 10–100，观测范围 5–140（n = 27）；每 IP 每小时次数 90 % 在 1–1；body.fmt 取值 `[a-z]{4}`；表单键必含 dept=、items[]=、period=；dept= 取值 `[A-Z]{2}`。流程：GET /report/form → POST /report/generate（间隔 75 秒–4 分钟）。置信 0.45 · 首次 2025-09-01 · 最近 2025-09-19 · v1.0
- `confirmed`, confidence 0.45: 【oa】工作日 17:00–17:10（覆盖 91 %，11 个工作日），综合部（10.168.7.121、192.168.1.23）访问 GET /report/form（查看报告）：下行字节 90 % 在 6–12 KB，观测范围 5–12 KB（n = 28）；上行字节 90 % 在 503–517 B，观测范围 502–518 B（n = 6）；时长 90 % 在 10–100，观测范围 10–120（n = 28）；net.pkts_down 90 % 在 6–10，观测范围 6–10（n = 28）；每 IP 每小时次数 90 % 在 1–1；请求头 user-agent 取值 `[A-Z][a-z]{6}/[0-9]\.[0-9] \([A-Z][a-z]{6} [A-Z]{2} [0-9]{2}\.[0-9]; [A-Z][a-z]{2}[0-9]{2}; [a-z][0-9]{2}\) [A-Za-z0-9]{11}/[0-9]{3}\.[0-9]{2} \([A-Z]{5}, [a-z]{4} [A-Z][a-z]{4}\) [A-Z][a-z]{5}/[0-9]{3}\.[0-9]\.[0-9]\.[0-9] [A-Z][a-z]{5}/[0-9]{3}\.[0-9]{2}( [A-Z][a-z]{2}/[0-9]{3}\.[0-9]\.[0-9]\.[0-9])?`；请求头 x-client-ver 取值 `[0-9]\.[0-9]\.[0-9]`。流程：GET /report/form → POST /report/generate（间隔 75 秒–4 分钟）。置信 0.45 · 首次 2025-09-01 · 最近 2025-09-19 · v1.0
- `stable`, confidence 0.60: 【oa】工作日 09:30–16:29（覆盖 96 %，11 个工作日），综合部（10.168.7.121、192.168.1.21、192.168.1.23）访问 GET /docs（文档）：下行字节 90 % 在 10–30 KB，全部在 5–30 KB（n = 784，下次越界概率 ≤ 0.3 %）；上行字节 90 % 在 460–510 B，全部在 450–510 B（n = 644，下次越界概率 ≤ 0.4 %）；时长 90 % 在 10–100，全部在 5–200（n = 784，下次越界概率 ≤ 0.3 %）；net.pkts_down 90 % 在 8–22，全部在 6–24（n = 722，下次越界概率 ≤ 0.3 %）；每 IP 每小时次数 90 % 在 1–2；客户端栈 取值集合封闭 {-|chrome/126|win|128|w16, -|edge/125|win|128|w16, -|edge/126|win|128|w16, -|firefox/126|linux|64|w15, -|firefox/127|linux|64|w15, -|safari/17|mac|64|w16}；请求头 user-agent 取值 `[A-Za-z0-9\ \(\),\./:;_]{70,125}`；请求头 user-agent.len 取值集合封闭 {111.0, 114.0, 125.0, 70.0}；请求头 x-client-ver 取值 `[0-9]\.[0-9]\.[0-9]`，请求头 x-client-ver 取值集合封闭 {5.2.1}；waf.score 取值集合封闭 {0.0, 1.0, 2.0}；net.win 取值集合封闭 {29200.0, 64240.0, 65535.0}。流程：GET /docs → GET /docs/{num}（间隔 10 秒–56 秒）；POST /docs/{num}/comment → GET /docs（间隔 6 分钟–29 分钟）；GET /docs/{num} → GET /docs（间隔 3 分钟–28 分钟）；GET /home → GET /docs（间隔 10 分钟–31 分钟）。置信 0.60 · 首次 2025-09-01 · 最近 2025-09-19 · v2.0
- `confirmed`, confidence 0.45: 【oa】工作日 09:30–16:56（覆盖 97 %，12 个工作日），10.168.7.121、192.168.1.21、192.168.1.23、192.168.2.10、192.168.2.11访问 GET /docs（文档）：下行字节 90 % 在 10–30 KB，全部在 5–30 KB（n = 76，下次越界概率 ≤ 2.6 %）；上行字节 90 % 在 496–510 B，全部在 496–510 B（n = 32，下次越界概率 ≤ 6.1 %）；时长 90 % 在 10–100，全部在 5–200（n = 82，下次越界概率 ≤ 2.4 %）；net.pkts_down 90 % 在 8–22，全部在 6–24（n = 36，下次越界概率 ≤ 5.5 %）；每 IP 每小时次数 90 % 在 1–2.59；客户端栈 取值集合封闭 {-|chrome/126|win|128|w16, -|edge/125|win|128|w16}；请求头 user-agent 取值 `[A-Z][a-z]{6}/[0-9]\.[0-9] \([A-Z][a-z]{6} [A-Z]{2} [0-9]{2}\.[0-9]; [A-Z][a-z]{2}[0-9]{2}; [a-z][0-9]{2}\) [A-Za-z0-9]{11}/[0-9]{3}\.[0-9]{2} \([A-Z]{5}, [a-z]{4} [A-Z][a-z]{4}\) [A-Z][a-z]{5}/[0-9]{3}\.[0-9]\.[0-9]\.[0-9] [A-Z][a-z]{5}/[0-9]{3}\.[0-9]{2}( [A-Z][a-z]{2}/[0-9]{3}\.[0-9]\.[0-9]\.[0-9])?`；请求头 user-agent.len 取值集合封闭 {111.0, 125.0}；请求头 x-client-ver 取值 `[0-9]\.[0-9]\.[0-9]`，请求头 x-client-ver 取值集合封闭 {5.2.1}；net.win 取值集合封闭 {64240.0}。流程：GET /docs → GET /docs/{num}（间隔 10 秒–56 秒）；POST /docs/{num}/comment → GET /docs（间隔 6 分钟–29 分钟）；GET /docs/{num} → GET /docs（间隔 3 分钟–28 分钟）；GET /home → GET /docs（间隔 10 分钟–31 分钟）。置信 0.45 · 首次 2025-09-16 · 最近 2025-09-19 · v1.0
- `confirmed`, confidence 0.45: 【oa】工作日 09:30–16:56（覆盖 97 %，12 个工作日），综合部（10.168.7.121、192.168.1.21、192.168.1.23）访问 GET /docs（文档）：下行字节 90 % 在 10–30 KB，全部在 5–30 KB（n = 76，下次越界概率 ≤ 2.6 %）；上行字节 90 % 在 496–510 B，全部在 496–510 B（n = 32，下次越界概率 ≤ 6.1 %）；时长 90 % 在 10–100，全部在 5–200（n = 82，下次越界概率 ≤ 2.4 %）；net.pkts_down 90 % 在 8–22，全部在 6–24（n = 36，下次越界概率 ≤ 5.5 %）；每 IP 每小时次数 90 % 在 1–2.59；客户端栈 取值集合封闭 {-|chrome/126|win|128|w16, -|edge/125|win|128|w16}；请求头 user-agent 取值 `[A-Z][a-z]{6}/[0-9]\.[0-9] \([A-Z][a-z]{6} [A-Z]{2} [0-9]{2}\.[0-9]; [A-Z][a-z]{2}[0-9]{2}; [a-z][0-9]{2}\) [A-Za-z0-9]{11}/[0-9]{3}\.[0-9]{2} \([A-Z]{5}, [a-z]{4} [A-Z][a-z]{4}\) [A-Z][a-z]{5}/[0-9]{3}\.[0-9]\.[0-9]\.[0-9] [A-Z][a-z]{5}/[0-9]{3}\.[0-9]{2}( [A-Z][a-z]{2}/[0-9]{3}\.[0-9]\.[0-9]\.[0-9])?`；请求头 user-agent.len 取值集合封闭 {111.0, 125.0}；请求头 x-client-ver 取值 `[0-9]\.[0-9]\.[0-9]`，请求头 x-client-ver 取值集合封闭 {5.2.1}；net.win 取值集合封闭 {64240.0}。流程：GET /docs → GET /docs/{num}（间隔 10 秒–56 秒）；POST /docs/{num}/comment → GET /docs（间隔 6 分钟–29 分钟）；GET /docs/{num} → GET /docs（间隔 3 分钟–28 分钟）；GET /home → GET /docs（间隔 10 分钟–31 分钟）。置信 0.45 · 首次 2025-09-16 · 最近 2025-09-19 · v1.0

**System view, finance approvals:**

- `stable`, confidence 0.38: 【finance】工作日 10:00–11:27、15:01–15:56（覆盖 93 %，13 个工作日），财务部（192.168.2.10）访问 GET /fin/approval/list（查看审批）：下行字节 90 % 在 8–27 KB，观测范围 8–28 KB（n = 9）；时长 90 % 在 10–100，全部在 5–140（n = 61，下次越界概率 ≤ 3.3 %）；每 IP 每小时次数 90 % 在 1–3。流程：GET /fin/approval/list → POST /fin/approval/{num}/approve（间隔 30 秒–2 分钟）；POST /fin/approval/{num}/approve → GET /fin/approval/list（间隔 6 分钟–28 分钟）。置信 0.38 · 首次 2025-09-01 · 最近 2025-09-21 · v1.0
- `stable`, confidence 0.33: 【finance】工作日 10:02–11:28、15:03–15:58（覆盖 92 %，13 个工作日），财务部（192.168.2.10）访问 POST /fin/approval/{num}/approve（审批）：viewstate.len= 90 % 在 915–1500，观测范围 915–915（n = 1）；voucher= 90 % 在 20000000–100000000，全部在 10000000–100000000（n = 48，下次越界概率 ≤ 4.1 %）；提交数据量 90 % 在 1–1.6 KB，全部在 0.9–1.6 KB（n = 55，下次越界概率 ≤ 3.6 %）；下行字节 90 % 在 11.2–17 KB，观测范围 11–17 KB（n = 7）；时长 90 % 在 10–100，全部在 5–140（n = 58，下次越界概率 ≤ 3.4 %）；每 IP 每小时次数 90 % 在 1–3；body.fmt 取值 `[a-z]{4}`，body.fmt 取值集合封闭 {form}；表单键必含 amount=、opinion=、sign=、viewstate=、voucher=；opinion= 取值 `[\u0080-\U0010ffff]{2,8}`，opinion= 取值集合封闭 {同意, 同意，请尽快办理, 退回}；viewstate= 取值 `[A-Za-z0-9\-_]{511,2046}`；请求头 content-type 取值 `[a-z]{11}/[a-z]\-[a-z]{3}\-[a-z]{4}\-[a-z]{10}`，请求头 content-type 取值集合封闭 {application/x-www-form-urlencoded}。流程：GET /fin/approval/list → POST /fin/approval/{num}/approve（间隔 30 秒–2 分钟）；POST /fin/approval/{num}/approve → GET /fin/approval/list（间隔 6 分钟–28 分钟）。置信 0.33 · 首次 2025-09-01 · 最近 2025-09-19 · v1.0

**User view — the department (`class:grp:dept:综合部`, composed of its learned roles):**

- header: 综合部（3 个 IP，2 个行为群组）使用 mail、oa
- 综合部 访问 mail（占其活动 18 %）：邮件（TLS mail.corp.local）
- 综合部 访问 oa（占其活动 82 %）：文档（GET /docs/{num}）、文档（GET /docs）、评论（POST /docs/{num}/comment）、首页（GET /home）、登录（POST /login）、查看报告（GET /report/form）[10.168.7.121、192.168.1.23]、提交报告（POST /report/generate）[10.168.7.121、192.168.1.23]、查看审批（GET /approval/list）[192.168.1.21]、查看审批（GET /approval/{num}）[192.168.1.21]、审批（POST /approval/{num}/approve）[192.168.1.21]
- (negative) 综合部 在 crm 中从未执行写操作（15 天、0 次）（封闭的写操作：客户（POST /crm/visit））
- (negative) 综合部 在 finance 中从未执行写操作（21 天、0 次）（封闭的写操作：审批（POST /fin/approval/{num}/approve）、登录（POST /fin/login））；192.168.1.23 的尝试被判定为越权（未学习）
- (negative) 综合部 在 portal 中从未执行写操作（21 天、0 次）（封闭的写操作：登录（POST /login）、评论（POST /comment））

**User view — the learned groups holding 综合部's addresses** (192.168.1.21 → G24, 192.168.1.23 → G7, 10.168.7.121 → G7):

- G24 `综合部·oa GET /approval/list` (192.168.1.21): 综合部·oa GET /approval/list（1 个 IP）使用 mail、oa
  - (negative) 综合部·oa GET /approval/list 在 crm 中从未执行写操作（15 天、0 次）（封闭的写操作：客户（POST /crm/visit））
  - (negative) 综合部·oa GET /approval/list 在 finance 中从未执行写操作（21 天、0 次）（封闭的写操作：审批（POST /fin/approval/{num}/approve）、登录（POST /fin/login））
  - 综合部·oa GET /approval/list 访问 mail（占其活动 13 %）：邮件（TLS mail.corp.local）
  - 综合部·oa GET /approval/list 访问 oa（占其活动 87 %）：文档（GET /docs/{num}）、文档（GET /docs）、审批（POST /approval/{num}/approve）、查看审批（GET /approval/{num}）、查看审批（GET /approval/list）、评论（POST /docs/{num}/comment）、首页（GET /home）、登录（POST /login）
  - (negative) 综合部·oa GET /approval/list 在 oa 中从未执行：提交报告（POST /report/generate）（21 天、0 次）
  - (negative) 综合部·oa GET /approval/list 在 portal 中从未执行写操作（21 天、0 次）（封闭的写操作：登录（POST /login）、评论（POST /comment））
- G7 `综合部` (10.168.7.121, 192.168.1.23): 综合部（2 个 IP）使用 mail、oa
  - (negative) 综合部 在 crm 中从未执行写操作（15 天、0 次）（封闭的写操作：客户（POST /crm/visit））
  - (negative) 综合部 在 finance 中从未执行写操作（21 天、0 次）（封闭的写操作：审批（POST /fin/approval/{num}/approve）、登录（POST /fin/login））；192.168.1.23 的尝试被判定为越权（未学习）
  - 综合部 访问 mail（占其活动 21 %）：邮件（TLS mail.corp.local）
  - 综合部 访问 oa（占其活动 79 %）：文档（GET /docs/{num}）、文档（GET /docs）、评论（POST /docs/{num}/comment）、首页（GET /home）、登录（POST /login）、提交报告（POST /report/generate）、查看报告（GET /report/form）
  - (negative) 综合部 在 portal 中从未执行写操作（21 天、0 次）（封闭的写操作：登录（POST /login）、评论（POST /comment））

Reading (against the requirement's example): the 综合部 login node names the three addresses,
their bindings (192.168.1.21 still `jack` on day 21 although A2 borrowed `rose` from it on days
17–21), the closed user-name set {jack, mike, mike.w, rose} with grammar `[a-z]{4}(\.[a-z])?`,
the band 1–2 KB and the data's range 1–2.6 KB; its window 08:32–08:51 is D1's new window (A2's
09:10 logins are no longer a window, R4). Its stated confidence 0.31 is the node's held-out hold
rate after the D1/D2 drifts. The department's view composes its two learned roles (the approver
192.168.1.21 and the report writers .23 / .121), and says that 综合部 never writes in finance —
"192.168.1.23 的尝试被判定为越权（未学习）": A1's write was judged foreign and not learned,
which is the requirement's "访问财务系统去审批就是异常". The role group of the approver also says
what it never does inside OA ("在 oa 中从未执行：提交报告", deviation G13). The OA approval statements are
`stale` on day 21: D3 renamed the routes on day 14; P10 adopted the rename on day 17 (R3), but the
`/flow/` nodes are still candidates (16.11.8 item 4). Finance's approval is "财务部（192.168.2.10）"
only, in its two workday windows 10:00–11:28 and 15:01–15:58.


#### 16.11.7 Variants, non-regression and resources

- **O-red seed 0** (43 min; peak RSS not kept by the re-scoring): recall by day d5 0.28, d7 0.35, d8 0.39, d10 0.58, d14 0.59, d21 0.68; precision by day d7 0.43, d8 0.53, d14 0.52, d21 0.75; ECE@14 0.29; false splits per system-month 0.0; anomalies 8/10 (missed: A4, A7); GA login who = the truth IPs True, finance approver only True, ARI 0.89, portal bindings 50; FAR >= LOW / >= MEDIUM 0.0056 / 0.0043
- **O60 seed 0** (22 min; peak RSS not kept by the re-scoring): recall by day d5 0.35, d7 0.47, d8 0.50; precision by day d7 0.56, d8 0.70

- **O60 (aggregated vs 60-s event mode, PG1 at day 8 within 0.05):** days 1–7 are the same
  aggregated stream as pack O (day 7 recall 0.47 against pack O's 0.42, seed 0); on day 8, in
  60-s event mode, recall 0.50 against pack O's 0.48 on its aggregated day 8 (difference 0.02,
  inside 0.05; round 2: 0.053), precision 0.70 against 0.56. Its ten anomalies fall after day 8.
- **O-red** (never tuned on): recall@14 0.59 (round 2: 0.41), 8 of 10 anomalies (round 2: 7;
  missed now A4 and A7, round 2 A2, A3, A4), no false splits.
- **O-real (PG11)** was run once this round by the truth / cost / scale owner on a code snapshot
  of 04:49Z (before the final code): 3 of 12 R items pass (R8, R10, R11), 4 of 10 anomalies over
  35 days, recall / precision @14 0.45 / 0.17 (`reports/progressive/round3_eval/O-real_0.json`).
  It was not re-run on the final code (35 days ≈ 85 min of wall time).
- **PG4** points are the 7-day points of `reports/progressive/round3_eval/scale7` (20 000 IPs,
  340 attributes, 20 / 100 / 300 systems; code snapshot of 04:49Z, timings taken with 3–4 heavy
  processes on 4 cores): memory slope vs IPs 0.040, CPU/event slope vs IPs −0.157, memory slope
  vs attributes 0.172, CPU/event slope vs attributes 0.149 (round 2: 0.236, fail → pass), idle
  system 0.45 MB, largest tree 14.2 MB pass; memory slope vs systems 0.45 (> 0.3) and the per-event
  p95 (scoring 6.4 ms, learning 5.6 ms against 100 / 250 µs; the maximum over all points,
  dominated by the servers packs' 5–30 k events) fail. The engines changed after the snapshot
  (V2–V6) add no per-event work.
- **PG9 / non-regression:** packs A and E seed 0 with the default registry are identical to
  round 4, key for key apart from timings (0 differences each).
- **Per run:** 38–45 min wall (seeds 0–4; O-red 43 min, O60 22 min) with 3–5 heavy processes on
  4 cores, peak RSS 1.58–1.93 GB (round 2: 1.26–1.30 GB). The peak includes the checkpoint
  serialisation of the resumable runner (a 300–420 MB pickle of the whole run state every ~10
  minutes); the engines' own state was not measured separately. Engine CPU per run 2 180–2 570 s
  (round 2: 1 760 s at lower machine load; P04 285–358 s against 253 s, P03 98–115 s against
  72 s).
- **Incident:** the disk filled at ~08:55 (old rounds' scratch data); seed 1's run stopped at a
  checkpoint write and was resumed from its day-10 checkpoint (a resumed run equals an uninterrupted one except the wall-clock-derived fields, §11.6), and two
  tests of the full suite failed with ENOSPC (pass when re-run).


#### 16.11.8 Diagnosis of what still fails (by engine)

Per truth pattern at day 14, the misses of the 5 seeds (190 eligible rows, 42 missed; driver
`prog3/evl/diag.py`, scratch logs `diagZ_s*.txt`):

| Miss (seeds) | Engine | Mechanism |
|---|---|---|
| FIN `POST /fin/login` (5/5) | P07 | the three users' closed set is not stated: 16–19 undecayed observations at day 14 against `CLOSED_N` = 20 (U 0.026–0.029 > `CLOSED_U` = 0.02); closes by day 21 |
| FIN `POST /fin/approval/{id}/approve` (5/5; and `GET …/list` on seed 1) | P07 | the opinion's 3-value closed set: U 0.021, just above 0.02 |
| FIN `POST /login` in OA (5/5) | P04 | no 财务部-only login node by day 14: FIN and SALES share one /24-split node until ~day 15–16 (evidence-limited); their parts state the node's band |
| DEV `POST /login` (5/5) | P07 / truth | the truth asks for the pool's ~60-name closed set; the sketch grows to 64 but U stays ≈ 0.9 at day 14 (each user logs in once a day: the set is honestly open) |
| GA 17:00 report (6 of 10 rows) | P04 | confirmed one to two days after PG1's eligibility: the route node misses the first day's reports (created on the second report), so `n_obs` reaches 20 two reports after the 20 opportunities |
| mail (GA / FIN / SALES; 9) | P04 / P09 | opaque TLS whose departments differ only by 10–30-minute windows: the /24 nodes mix GA and FIN, parts fit their own windows from a few reservoir points (FIN 09:22–09:29 against the truth's 09:15–09:33) |
| AUTO health monitors (3; seeds 1, 2) | P09 | on day 14 of seed 1 the 60-s monitors of OA and finance state a 23-minute window with 38 % coverage (also in round 2); the window is back to 00:00–24:00 by day 21 |
| FIN comment (2), SALES login (1) | P14 / P11 | the 财务部 part lists .10 and .11 only (.12's signature share below `SIG_STANDING`); seed 2's 销售部 part of the shared node |

At day 21, besides those: D3's three renamed approval rows (seeds 1–4, item 4), the 01:00
backup (all seeds: its node was created on the second backup, day 2, so it holds 20
observations only at the end of day 21 — one day after PG1's eligibility — and is still a
candidate; item 2), the finance voucher node `evolving` after a drift alarm with no drift in the
truth (seeds 0 and 2) and portal comment / login windows on some seeds.

1. **The 综合部 login node on seeds 2 and 4 (PG3 4/5, PG10, the checklist).** Seed 2: the OA
   login node splits on the learned-group level (`net.src` level 3); P11 has made
   192.168.1.21 a one-address role group (G13: it alone approves) with ~1 login a day, so
   its value goes to the split's `other` branch (销售部 + 研发 groups) and the 综合部 node holds
   .23 and .121 only. Before round 3 the same split happened with a different value grouping
   (round 2 seed 2 passed). Seed 4: the login node is not split below the department level by
   day 21. Both are P04 split-value grouping with low-evidence values; the groups / views owner
   proposed treating the roles of one configured department as one value at the group level.
2. **First rows of a route (P04).** A route node of the root partition is created on the
   route's second unit and replays ≤ 8 waiting rows (R3-4); on pack O the first day's 17:00
   reports and 01:00 backups and seed 1's day-2 643 B login (the hard-range clause) are still
   missing from their nodes.
3. **Calibration (PG2, ECE@14 0.23, target 0.05).** Statements are under-confident (seeds 0–2, day 14): nodes state
   0.47–0.55 and hold 0.58–0.61 on the evaluator's held-out events; group parts state 0.43–0.61
   and hold 0.79–0.81 (a part's held-out events are its members', which the node's own
   held-out test — P04 `p_hold` — does not separate). ECE@21 0.34. Before (round-2 engine, same
   scorer) 0.34 / 0.47.
4. **D3, the approval rename (PG5 0/5).** P10 adopts the rename on day 17 (R3), but the
   `/flow/` nodes (created day 15.4) stay candidates: 192.168.1.21's first `/flow/` page opened a
   HIGH incident (P03 `new_action`, p 1.4e-4) that held its rows until day 16.5, and from day 17
   the same address is A2's source (the borrowed credential, days 17–21): its HIGH incidents hold
   its rows again. The pack puts the only approver's rename and an attack from the same address
   in one week; P03 does not yet call `pdfg.successor_candidate` to cap the first renamed page.
5. **D4 / D5 / FAR (PG5, PG6).** D4 and D5 fail on 1–2 LOW incidents per seed (single content
   findings, e.g. on seed 0: a portal comment's 222 ms duration against an observed range of 4–200 ms; a git
   `net.pkts_down` of 2 968 inside the displayed range 4–3 000 but past the GPD tail's endpoint,
   p 1e-9). **Held-out seed 4 regressed:** FAR ≥ MEDIUM 0.0028 → 0.0067 and 8 incidents in D5:
   fresh 研发 leases (12-h leases from day 15) logging in to OA were `outsider_group,
   system_new` (MEDIUM / HIGH) — the pool path of G10 / G15 does not cover these leases on that
   seed. D1 2/5: the 综合部 node (item 1) or the D1 window confirm late.
6. **PG8 (0.83).** Finance's who arm is `prefix` where the strategy truth expects ip / grp
   (seeds 1, 3, 4), and the portal's binding arm P08 is on at day 14 where the truth expects it
   off (seeds 1, 2, 4; on seed 1 with a measured gain of 0 — the same measurement left it off in
   the run of the round's first integrated snapshot: P12's decisions use wall-clock engine costs,
   the reproducibility issue of the truth / cost owner).
7. **PG10.** Day 11 (0/5): no address-level 综合部 login node before D1 on any seed (the
   department's part states the mixed node's band). Day 21: seed 0 passes; the others fail on the
   clauses of the checklist (16.11.5).
8. **PG4.** Unchanged from the scale owner's points: the systems slope (0.45) and the per-event
   p95 (6.4 / 5.6 ms) fail; P04 learning and P05 selection dominate (owner proposals in the
   truth / cost report).
9. **PG6 B29.** "B29 top reason = violated constraint" 0/5 as in round 2 (not investigated
   this round).

#### 16.11.9 Open issues for the owners (round 3 → round 4)

- **P04 (tree):** split-value grouping at the group level loses a low-evidence role group of a
  configured department (seed 2, item 1); first rows of a new route (item 2); `p_hold` cannot
  calibrate a group's part (item 3); false drift alarms make a stable node `evolving` (finance
  vouchers, seeds 0 and 2).
- **P03 / P11 (who, pools):** fresh DHCP leases after D5 are `outsider_group, system_new` on
  seed 4 (item 5); `successor_candidate` is not used for D3's first renamed page (item 4); GPD
  tails give p 1e-9 just past an observed maximum (item 5).
- **P07:** `CLOSED_N` / `CLOSED_U` close a 3-user set only after ~20 logins (FIN, items above).
- **P09:** the health monitors' 23-minute window on seed 1 day 14; parts' windows from a few
  reservoir points (mail).
- **P12:** wall-clock costs make arm decisions differ between identical runs (item 6).
- **Evaluator:** the remaining PG2 "non-decreasing" checks fail on dips of ≤ 0.07 at drift days
  — a tolerance (as recall has, ±0.05) is a gate decision for the lead; O-real on the final code
  (85 min) and the O-servers PG4 points were not re-run.

---

## Appendix A. Review changes (adversarial review, 2026-09-29)

The first draft was reviewed sentence by sentence against the user requirement (§3) and
against the state of the art it cites. Each row lists what was wrong or missing, the change,
and where it now lives. Nothing here has been implemented or measured; every number added by
the review is either a default with its reason (§7.1) or an estimate, like the rest of the
document.

| # | Finding (what was wrong or missing) | Change | Where |
|---|---|---|---|
| RC1 | **Requirement gap: servers.** S1 forbids traversing all users *and servers*. Every system had a fixed ≈ 20–30 MB tree and fixed hourly/6-hourly fits, so memory and CPU grew linearly with the number of servers; a new server started from nothing. | System families (shared tree, `net.dst` as an ordinary split attribute: "一类服务器 → 某一台服务器"), activity-proportional tiers incl. XS, checkpoint-out of idle trees, dirty-node periodic work, cold start by family inheritance; PG4 server curves; pack O-servers; R9, R13. | PPC-10, §6.20, §6.19, §7, §11.6, §12 PG4 |
| RC2 | **Statistical error: evidence inflated by mass.** ω = H(r + m) − H(r) let one aggregated row count ≈ ln m units and a thinned HT row (weight up to ~1 000) ≈ 7 units; split statistics and Dirichlet predictives used mass counts, so HT weights made nodes look far more certain than the learner's observations justify. | Mass and evidence separated everywhere: ω ≤ 1 per observed row (harmonic within a run), evidence-scaled predictives, split statistics in evidence units, sensor sample rate scales mass only. | PPC-9, §5.2.3, §6.5.3–6.5.4, §6.13 |
| RC3 | **Statistical error: split significance.** The summed prequential saving (a ratio of two Bayes mixtures) is not a test martingale under a composite null, double-counts dependent targets, and "C_eval counts every evaluation" was an unjustified charge; the Bernstein check was a fixed-n bound applied repeatedly. | Rule (V): per-target universal-inference e-values (pooled ML code length vs prequential split code), averaged over targets (valid under any dependence), Ville's inequality ⇒ anytime-valid; multiplicity charged per candidate (log2 i ⇒ ≤ 2^−τ0 · H(C_ever)); MDL sum kept as gain (G); time-uniform Bernstein for (S); learning restarts after R_learn; system-level false-split rate stated; exceptions use the same test with a conservative leave-x-out construction; revision and alternates use an explicit τ0 margin. | PPC-11, §6.5.5, §6.6, §6.7, P04 tests (b)–(b″) |
| RC4 | **Statistical error: decay vs bounds ("越久越准" violated).** All confidences used H_m-decayed evidence, which plateaus at rate × 10 d: the finance approval node's U could never fall below ≈ 0.016, so its HIGH-candidate severity (U ≤ 0.01) and the P03 test fixture (n_eff = 200) were unreachable; single-IP exceptions (n_eff,x ≥ 20) and node invariants (n_eff ≥ 50) were unreachable for once-a-day sources. | Confidence channel: H_l evidence reset to the H_m state on an accepted change, capped while evolving; shape stays on H_m. Confirmation, closedness, bindings, bounds, invariants, exceptions and statement support use it; the remaining plateau is stated with examples. | §6.9.4, §5.5.3, §6.8.1, §7.1 |
| RC5 | **Statistical error: range bound.** The 30-day observed range was paired with the H_m n_eff; the rank bound 2/(n+1) needs the number of observations the extremes were taken from. | Daily ring of (min, max, n_obs) over the confidence segment; n_rng; render threshold 30 (was 50, unreachable for small nodes by day 14). | §6.10 |
| RC6 | **Multiple testing on emissions.** A busy IP was tested per event against fixed thresholds; FAR per entity-day grew with the IP's event count. | Emission thresholds on p_day (Šidák over the IP's scored events at the (node, type) that day); expected chance emissions per clean IP-day bounded. | §6.16.3, P03 test (h) |
| RC7 | **Who violations conflated DHCP and intruders.** A never-seen IP had grp = ∅ and was flagged `outsider_group` ⇒ HIGH candidate on every re-addressing; A8 had no rule distinguishing credential reuse from re-addressing. | `outsider_group` only for grouped IPs; `unknown_ip`; binding concurrency test (`concurrent_use` vs `readdress_candidate`); provisional P11 join; heavy-set definition; severity rows updated; A8 made explicit. | §6.12, §6.15, §6.16.2–6.16.3, §11.4 |
| RC8 | **Poisoning gap.** A single foreign source that persisted for 5 days could be accepted into a who-closed node (single-IP persistence rule), and an attacker not yet in an incident learned at full weight. | Who-set changes need a group already present, a readdress candidate, a label, or (σ < 2) ≥ 2 sources of one group; outlier damping for p ≤ 1e-4 events; P02/P10/P11 learn from t − D with trust; new anomaly A9 (low-and-slow) with a non-adoption check. | §6.9.2–6.9.3, §4.2, §11.4, PG6 |
| RC9 | **Hidden per-IP enumeration / unbounded state.** P01 built a window event for every active IP each grain (96 store reads each); P03's intensity counter beyond its cap sampled new pairs with weight 10 (false alarms possible); B04 shadow LRU of 100 000 IPs (≈ 60 MB per system); P13 composed a portrait for every IP every 2 h; batches of all events were retained 2 h (≈ GBs at 5 000 events/s) and not counted in §7; retired summaries and group views were uncapped; P05 swept all 512 attributes hourly. | W_max with priority sampling; intensity SpaceSaving scored on guaranteed counts plus a separate `pat.rate` batch; B04 shadow only for heavy candidates with shrinkage; P13 IP portraits earned-only; batches compacted to learned rows, retained D + 1 ticks, M_batch/M_tick in §7, R_tick cap; retired summaries ≤ N_max/2; G_max group views; P05 rotating slice; all LRU caps allocated by P15 from the sources seen. | §6.2.3, §6.16.4, §10.2, §6.17.1, §5.2.3, §5.6, §6.8.2, §6.15, §6.4, §7.2 |
| RC10 | **Infeasible budget.** `cpu_ms_per_tick` = 80 ms (the gate-14 latency figure) while lib-3 already needs ≈ 909 ms at 35 entities and the P-core ≈ 1–2 s per 900-s tick: the ladder would have been engaged permanently, and ladder step 5 touched B-engines even in full mode. | Budgets as CPU share and memory, P-core separate from lib-3; ladder step 5 only in bounded mode; step 7 priority scoring under overload; e_rate default 10/s. | §6.19, §7.1, §13.2 |
| RC11 | **Metrics still hard-coded.** Sampling strata, P05 context seeds and code-type hints were fixed lists. | Stratum from the tree's own root split; seeds are bootstrap only; type hints are configuration with a data test. | §6.2.2, §6.4, §5.3 |
| RC12 | **Schema changes looked like behaviour changes.** A missing split attribute sent every event to `other`. | Absence is a value (`⊥`); `attribute_gone` detection; splits on gone attributes collapse. | §5.2.2, §5.5.1, §6.3, §6.6, R10 |
| RC13 | **Missing real-world cases.** No treatment of SNAT reverse proxies, NAT/VDI, shared terminals, service accounts, IPv6 privacy addresses, holidays vs weekends, month-end jobs, scanners, sensor sampling, late logs, new servers, binary bodies. | Who resolution (XFF, trusted proxies, `snat_suspect`), `shared:<ip>`, `sess.key`, set bindings, `/64`, `ctx.dayclass`/`dom`/`mend`, normal-day counting, dormant patterns, outlier damping, sample-rate mass; a situation table with signals, responses and residual limits; pack O-real (R1–R13) and PG11. | §5.1.4, §5.4, §6.2.3, §6.8.1, §6.12, §6.21, §11.4, PG11 |
| RC14 | **Grammar example inconsistent with the algorithm.** `[a-z]{1,10}` was presented as learned from jack/rose/mike, which anti-unification cannot produce; PG10 required "at most `[a-z]{1,10}`" although D2 introduces `mike.w`. | Length bounds are observed with coverage; operator-pinned business bounds via B23; example shows `[a-z]{4}`; prefix factoring specified; PG10 split into day 11 and day 21 checks with `[a-z.]{1,10}`. | §6.11, §6.17.2, §3 S8, PG10 |
| RC15 | **Evaluation could not prove its claims.** PG10 at day 14 conflicted with D1 (day 12) and with the range render threshold; weekly patterns could not be confirmed in 14 d (3 dates needed); A9 was missing though recall was "over 10 anomalies"; nothing measured "precision grows with time", false splits, servers, or real-world conditions; the red-team variant was undefined. | PG1 opportunity rule with ≥ 3 dates; PG2 weekly ≤ 21 d, confidence-growth and false-split checks; PG4 server dimension; PG6 A9; PG10 re-timed; PG11; packs O-real, O-red (with 20 independent attributes), O-servers defined. | §11.6, §12 |
| RC16 | **Workflow poisoning and id reuse.** P10 counted every current event (including an attacker's) into the DFG; SpaceSaving eviction could reuse an action id. | DFG counts from learned trusted rows of t − D via `prev_act` in `pat.assign`; ids never reused; sessions keyed by (ip, sess.key). | §6.14 |
| RC17 | **Strategy selection maths.** Hedge ran on a gain/cost ratio (unbounded losses); the who-level code was not a complete code for unseen items. | Utility in one currency (bits − λ_c · µs), clipped for Hedge; ratio only for knapsack packing; who-level two-part code with escapes specified; B-engine applicability arm. | §6.18.2 |
| RC18 | **Bindings.** The empirical-Bayes prior reused x's own data; no representation for shared terminals or service accounts; the example numbers were not reproducible. | Leave-one-out prior (numbers recomputed with scipy: 5 pure logins ⇒ LB ≈ 0.88), set bindings both directions, `sess.key` bindings behind NAT. | §6.12, P08 tests |
| RC19 | **Value policy vs requirement.** Production default HMAC would never show `username=jack`, the requirement's own example; secrets were caught by key name only. | Default clear for non-secret values (legal out of scope), secrets shape-only by key glob and value randomness (reusing `lib/template._looks_random`), long values shape-only; lead confirmation kept as an open item. | §5.1.3 |
| RC20 | **"自动学到综合部".** Names were said to come from config only, which reads as manual upkeep; the requirement expects the department to be recognised. | Stated plainly: membership is learned, a name needs a source; IPAM / DHCP-scope / asset / directory import script. | §0 item 6, §2, §6.15 |
| RC21 | **No end-to-end trace.** An implementer could not see how the example emerges (which splits, when groups arrive, when each fitter can speak). | Worked trace of the OA example, day by day, with the portal contrast. | §6.22 |
| RC22 | Small fixes: `p_ref` NaN before the first snapshot; ancestor-sampling uniform keyed by `seeded_uniform`; `rate_Hs` defined; P02 statistics from t − D; lag list in §9.1; family pseudo-system store keys; new events (`attribute_gone`, `pattern_revived`, `family_changed`); references added (Ville, universal inference, e-value combination, time-uniform bounds, priority sampling). | as listed | §5.6, §6.5.1, §6.8.3, §9.1, §15 |

**Open items for the lead** (added or changed by the review, beyond the architect's list):

1. The value-policy default (RC19) supersedes the decisions.md privacy line for the PPC.
2. Contract additions grow: `compact_batch` on the store (RC9), the `pat.rate` batch, the
   `fam:<id>` pseudo-system key for shared trees (RC1), and `model.sysfam`.
3. The confidence plateau (H_l = 30 d, RC4) is a product trade-off: longer H_l means more
   confidence for unchanging behaviour and slower forgetting of changes no detector flags.
4. Pack O-real extends pack O to 35 days for R8; eval wall time grows accordingly.
