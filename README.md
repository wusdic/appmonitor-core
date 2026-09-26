# AppMonitor Core · 组织信息化业务系统画像平台

在**不在主机安装任何软件**的前提下（以旁路流量为主、主动探测为辅），为组织内每个业务系统
构建完整画像。画像由**四个联动的库**组成：

1. **原始指标库**（Raw Metrics）— 基于流量/主动探测可获取、按分类尽量全的原始指标，含**获取方式**。
2. **次生指标库**（Derived Metrics）— 对原始指标分析/组合派生出的新指标。
3. **行为库**（Behavior）— 每业务系统×每实体**动态生成**的行为画像；给出又准又全的识别算法，
   可区分某一个/某一类用户，并识别“行为与日常不一致”。实体粒度到 **IP / 一类 IP**（不定位到人）。
4. **行为特征库**（Behavior Signatures）— 指标组合→语义的**预设库**，识别“用户在干什么”。

> 设计与实现细节见 **[`docs/组织业务系统画像平台设计.md`](docs/组织业务系统画像平台设计.md)**。

## 架构一览

```
数据源(被动/主动) → 原始指标库(6引擎) → 次生指标库(7引擎) → 行为库(7引擎) → 行为特征库(2引擎)
                   全部落入统一 MetricStore，引擎只按“指标名”读写，彼此低耦合、可独立增删换
```

共 **22 个引擎**，每个聚焦单一职责、低耦合、可复用：

| 库 | 引擎 |
|---|---|
| 原始 | l2l3 · l4flow · http · tls · dns · active_probe |
| 次生 | aggregation · ratio · periodicity · entropy · session · graph · trend |
| 行为 | feature_vector · baseline · fingerprint · clustering · anomaly · drift · sequence |
| 特征 | rule_match · correlation |

## 运行

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --app-dir . --host 0.0.0.0 --port 8099
```

打开 `http://localhost:8099/`（自动跳转 `/app/` 仪表盘）。仪表盘含六视图：
总览 / 实体画像(下钻行为指纹) / 行为事件 / 行为特征库 / 指标库 / 引擎拓扑。

无需真实流量即可体验：内置合成流量源会生成 3 个业务系统、多类实体，并注入
外发/扫描/信标/DNS隧道/凭据填充 5 类威胁与 1 个行为漂移场景。

## 自检

```bash
python scripts/smoke.py          # 后端全链路冒烟（引擎产出/画像/事件/命中）
python scripts/render_check.py   # 用内置 Chromium 无头渲染前端并截图
```

验证结果：合成场景下 **5/5 注入威胁全部捕获，14 个正常实体 0 误报**。

## 目录

```
backend/app/
  models/schema.py        统一数据模型
  core/{store,engine}.py  时序存储 + 引擎框架/注册表
  engines/{raw,derived,behavior,signature}/   四库引擎
  pipeline/               编排器 · 合成流量源 · 运行时装配
  api/                    FastAPI 只读投影层
data/
  catalog.yaml            指标库目录（原始55 + 次生24，含获取方式）
  signatures/*.yaml       行为特征库（原子24 + 组合5）
frontend/                 零依赖原生 JS 仪表盘（内联 SVG 图表）
docs/                     设计文档
scripts/                  冒烟/渲染自检
```

## 环境变量

`APPMON_WARMUP_TICKS`（预热拍数，默认180）· `APPMON_LIVE_PERIOD_S`（实时拍间隔秒，默认3）·
`APPMON_SIGNATURE_DIR` · `APPMON_FRONTEND_DIR`。
