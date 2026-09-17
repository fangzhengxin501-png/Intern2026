# 在线 POMDP 求解器对比 — 实验记录

对比对象：**POMCPOW** / **AdaOPS** / **AR-DESPOT**
问题集：RockSample(11,11)、RockSample(15,15)、LaserTag、LightDark1D

> 用法：每跑一轮，复制文末的 [空白模板](#空白模板复制这一段) 到「实验轮次」区最上面，
> 填日期和本轮改动，把 CSV 里的 mean/std 抄进主表。**一轮只改一个变量**。

---

## 0. 固定实验环境（基准 = `test_POMCPOW/test_RS15_POMCPOW.jl`）

以下七项在三个目录里必须完全一致，改动任意一项都要在本文件里开新的一轮：

| 项目 | 值 | 三个包里对应的字段 |
|---|---|---|
| 搜索深度 | 40 | POMCPOW `max_depth` / AdaOPS `max_depth` / ARDESPOT `D` |
| 每步规划预算 | 3.0 s | POMCPOW `max_time` / AdaOPS `T_max` / ARDESPOT `T_max` |
| 单局最大步数 | 100 | `HistoryRecorder(max_steps = 100)` |
| 重复次数 | 100 | `nb_runs` |
| 跟踪粒子数 | 5000 | `BootstrapFilter(pomdp, 5000, ...)` |
| 迭代次数上限 | 不设限 | POMCPOW `tree_queries` / ARDESPOT `max_trials` = `typemax(Int)`，由时间预算决定停止 |
| 随机种子 | planner `MersenneTwister(1)`；belief filter `1000+i`；HistoryRecorder `2000+i` | 第 i 局 |

回报口径统一为 `discounted_reward(history)`（折扣回报）。

### 各问题的固定属性（均为包默认值，脚本中未覆盖）

| 问题 | 构造 | γ | 关键回报 | 观测空间 |
|---|---|---|---|---|
| RS11 | `RockSamplePOMDP(11, 11)` | 0.95 | good rock +10 / bad rock −10 / exit +10 | 离散 |
| RS15 | `RockSamplePOMDP(15, 15)` | 0.95 | 同上 | 离散 |
| LaserTag | `gen_lasertag(rng = MersenneTwister(7), robot_position_known = false)` | 0.95 | tag +10 / 误 tag −10 / 每步 −1 | 离散（DESPOTEmu / DMeas） |
| LightDark | `LightDark1D()` | 0.9 | correct +10 / incorrect −10 | **连续** |

> γ 跨问题不同是 benchmark 本身的属性，**绝对回报不可跨问题比较**，只在同一行内比三列。
> 深度 40 对 γ=0.95 的问题会截断约 13% 的价值（0.95⁴⁰≈0.129），三个算法同等截断，公平但需在报告中说明。

### 文件对照

| 问题 | POMCPOW | AdaOPS | AR-DESPOT |
|---|---|---|---|
| RS11 | `test_POMCPOW/test_RS11_POMCPOW.jl` | `test_Ada/test_RS11_Ada_v2.jl` | `test_ARDESPOT/test_RS11_ARDESPOT.jl` |
| RS15 | `test_POMCPOW/test_RS15_POMCPOW.jl` | `test_Ada/test_RS15_Ada_v2.jl` | `test_ARDESPOT/test_RS15_ARDESPOT.jl` |
| LaserTag | `test_POMCPOW/test_LaserTag_POMCPOW.jl` | `test_Ada/test_LaserTag_Ada.jl` | `test_ARDESPOT/test_LaserTag_ARDESPOT.jl` |
| LightDark | `test_POMCPOW/test_LightDark_POMCPOW.jl` | `test_Ada/test_LightDark_Ada_v2.jl` | `test_ARDESPOT/test_LightDark_ARDESPOT.jl` |

输出目录：`data_POMCPOW/` · `data_Ada/` · `data_ARDESPOT/`（脚本用相对路径，**必须在各自目录下启动 Julia**）

---

## 1. Bounds 口径登记

AdaOPS 和 AR-DESPOT 是 branch-and-bound，上下界决定搜索方向和停止时机；POMCPOW 没有上下界，
但 `estimate_value` 扮演**下界**的同一角色。因此规定：

- **下界 / rollout 策略：三方严格一致**
- **上界：只在 AdaOPS 与 AR-DESPOT 之间严格一致**（POMCPOW 无上界，属结构性差异，报告中说明）
- 做消融时**一次只切一个问题**，且该问题的三个文件同步切

| 问题 | 变体 | 下界 (POMCPOW `estimate_value` / AdaOPS `FORollout` / ARDESPOT `DefaultPolicyLB`) | 上界 (AdaOPS `FOValue`/常数 · ARDESPOT 同) |
|---|---|---|---|
| RS11 / RS15 | **A（当前启用）** | `RSExitSolver()` | `FOValue(RSMDPSolver())` |
| RS11 / RS15 | B（注释） | `RandomPolicy(rng = MersenneTwister(3))` | 常数 `10.0 / (1 - discount(pomdp))` = 200.0 |
| LaserTag | **A（当前启用）** | `RandomPolicy(rng = MersenneTwister(3))` | 常数 `10.0` (= `tag_reward`) |
| LaserTag | B（注释） | 自写 `lt_heuristic`（四向逼近 + tag，见文件内注释） | 常数 `10.0` |
| LightDark | **A（当前启用）** | `FunctionPolicy(s -> s.y < 0 ? 1 : -1)`（朝光源） | 常数 `10.0` (= `correct_r`) |
| LightDark | B（注释） | 常数 `-10.0` (= `incorrect_r`) | 常数 `10.0` |

> 不要用 LaserTag.jl 自带的 `MoveTowards()`：它只在差向量各分量 ∈ {−1,0,1} 时有定义，
> 否则 `DIR_TO_ACTION[...]` 抛 `KeyError`，且可能返回 6..9 的对角动作而 `diag_actions` 默认为 false。

---

## 2. 实验轮次

<!-- 新的一轮插在这里（最新的放最上面） -->

### 第 0 轮 — 基线 smoke test（未完成）

| 字段 | 内容 |
|---|---|
| 日期 | 2026-09-17 |
| git commit | `________` |
| 目的 | 九个脚本能否跑通（`nb_runs = 1`），不看数值 |
| 相对上一轮的改动 | — （首轮） |
| Bounds 口径 | 全部变体 A |
| 状态 | ⬜ 未跑 |

**主表 — 平均折扣回报 mean ± std（n = 1，仅验证跑通）**

| 实验 \ 算法 | POMCPOW | AdaOPS | AR-DESPOT |
|---|---|---|---|
| RS11 | | | |
| RS15 | | | |
| LaserTag | | | |
| LightDark | | | |

**跑通情况**

| 实验 \ 算法 | POMCPOW | AdaOPS | AR-DESPOT |
|---|---|---|---|
| RS11 | ⬜ | ⬜ | ⬜ |
| RS15 | ⬜ | ⬜ | ⬜ |
| LaserTag | ⬜ | ⬜ | ⬜ |
| LightDark | ⬜ | ⬜ | ⚠️ 连续观测，DESPOT 观测分支退化，结果仅作不兼容性对照 |

**本轮记录**

已知待办（跑之前要处理）：

- [ ] `test_LightDark_POMCPOW.jl` CSV 文件名里的 `alphao(1/15)` 含 `/`，会在写文件时抛 `SystemError`
- [ ] `test_RS15_POMCPOW.jl` / `test_LightDark_POMCPOW.jl` 补 `estimate_value`，与 Ada/ARDESPOT 的下界对齐
- [ ] `test_LaserTag_POMCPOW.jl` 的 `MaxUCB(20.0)` 偏大（LaserTag 回报量级 −20..+10），扫 [3, 5, 10, 20]
- [ ] 三个环境各 `] add https://github.com/JuliaPOMDP/LaserTag.jl.git`
- [ ] `test_RoombaL_POMCPOW.jl` 缺 `RoombaPOMDPs` 依赖，暂时搁置
- [ ] 所有脚本 `nb_runs` 改回 100 再出正式数据

观察：

```
（留空）
```

结论 / 下一轮要改什么：

```
（留空）
```

---

## 空白模板（复制这一段）

### 第 N 轮 — 一句话标题

| 字段 | 内容 |
|---|---|
| 日期 | YYYY-MM-DD |
| git commit | `________` |
| 目的 | |
| 相对上一轮的改动 | **只写改了的那一项** |
| Bounds 口径 | 变体 A / 变体 B（哪个问题切了） |
| 状态 | ⬜ 未跑 / 🟡 进行中 / ✅ 完成 |

**主表 — 平均折扣回报 mean ± std（n = 100）**

| 实验 \ 算法 | POMCPOW | AdaOPS | AR-DESPOT |
|---|---|---|---|
| RS11 | | | |
| RS15 | | | |
| LaserTag | | | |
| LightDark | | | |

**数据来源 — summary CSV 文件名（可追溯）**

| 实验 \ 算法 | POMCPOW | AdaOPS | AR-DESPOT |
|---|---|---|---|
| RS11 | | | |
| RS15 | | | |
| LaserTag | | | |
| LightDark | | | |

**本轮超参快照（只填与上一轮不同的格子，相同的写「同上」）**

| 算法 | 关键超参 | RS11 | RS15 | LaserTag | LightDark |
|---|---|---|---|---|---|
| POMCPOW | `criterion` (MaxUCB c) | | | | |
| POMCPOW | `k_observation` / `alpha_observation` | | | | |
| POMCPOW | `enable_action_pw` / `check_repeat_obs` | | | | |
| POMCPOW | `estimate_value` | | | | |
| AdaOPS | `delta` | | | | |
| AdaOPS | `m_min` / `m_max` / `zeta` | | | | |
| AdaOPS | `grid` | | | | |
| AdaOPS | `num_b` | | | | |
| AR-DESPOT | `K` | | | | |
| AR-DESPOT | `lambda` / `xi` | | | | |

**每格耗时（可选，用于衡量 3 s 预算是否被真正用满）**

| 实验 \ 算法 | POMCPOW | AdaOPS | AR-DESPOT |
|---|---|---|---|
| RS11 | | | |
| RS15 | | | |
| LaserTag | | | |
| LightDark | | | |

**异常 / 报错**

```
（留空）
```

**观察**

```
（留空）
```

**结论 / 下一轮要改什么**

```
（留空）
```

---
