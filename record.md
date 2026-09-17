# 实验记录 — POMCPOW / AdaOPS / AR-DESPOT

> 超参数以各 `test_*/test_*.jl` 脚本里当前启用（未注释）的值为准，最后核对日期：2026-09-17。
> 详细的轮次日志、结果主表见 `docs/experiment_log.md`；本文件只记「环境 + 超参数」。

---

## 0. 固定实验环境（三个算法、四个问题全部一致，不随实验改变）

基准脚本：`test_POMCPOW/test_RS15_POMCPOW.jl`。以下任意一项改动都要在 `docs/experiment_log.md` 里开新一轮。

| 项目 | 值 | 各包对应字段 |
|---|---|---|
| 搜索深度 | 40 | POMCPOW `max_depth` / AdaOPS `max_depth` / ARDESPOT `D` |
| 每步规划预算 | 3.0 s | POMCPOW `max_time` / AdaOPS `T_max` / ARDESPOT `T_max` |
| 单局最大步数 | 100 | `HistoryRecorder(max_steps = 100)` |
| 重复次数 | 100 | `nb_runs`（smoke test 时临时设为 1/2，出正式数据前必须改回） |
| 跟踪粒子数 | 5000 | `BootstrapFilter(pomdp, 5000, ...)` |
| 迭代次数上限 | 不设限 | POMCPOW `tree_queries` / ARDESPOT `max_trials` = `typemax(Int)`，由时间预算停止 |
| 随机种子 | planner `MersenneTwister(1)`；belief filter `1000 + i`；HistoryRecorder `2000 + i`；全局 `Random.seed!(1)` | 第 i 局 |
| 回报口径 | `discounted_reward(history)`（折扣回报） | — |

### 四个问题的固定属性（均为包默认值，脚本未覆盖）

| 问题 | 构造 | γ | 关键回报 | 观测空间 |
|---|---|---|---|---|
| RS11 | `RockSamplePOMDP(11, 11)` | 0.95 | good rock +10 / bad rock −10 / exit +10 | 离散 |
| RS15 | `RockSamplePOMDP(15, 15)` | 0.95 | 同上 | 离散 |
| LaserTag | `gen_lasertag(rng = MersenneTwister(7), robot_position_known = false)` | 0.95 | tag +10 / 误 tag −10 / 每步 −1 | 离散（DESPOTEmu / DMeas） |
| LightDark | `LightDark1D()` | 0.9 | correct +10 / incorrect −10 | **连续** |

> γ 跨问题不同是 benchmark 自身属性，**绝对回报不可跨问题比较**，只在同一问题内比三个算法。
> 深度 40 对 γ = 0.95 截断约 13% 的价值（0.95⁴⁰ ≈ 0.129），三个算法同等截断。

### 运行环境 / 文件对照

| 问题 | POMCPOW | AdaOPS | AR-DESPOT |
|---|---|---|---|
| RS11 | `test_POMCPOW/test_RS11_POMCPOW.jl` | `test_Ada/test_RS11_Ada_v2.jl` | `test_ARDESPOT/test_RS11_ARDESPOT.jl` |
| RS15 | `test_POMCPOW/test_RS15_POMCPOW.jl` | `test_Ada/test_RS15_Ada_v2.jl` | `test_ARDESPOT/test_RS15_ARDESPOT.jl` |
| LaserTag | `test_POMCPOW/test_LaserTag_POMCPOW.jl` | `test_Ada/test_LaserTag_Ada.jl` | `test_ARDESPOT/test_LaserTag_ARDESPOT.jl` |
| LightDark | `test_POMCPOW/test_LightDark_POMCPOW.jl` | `test_Ada/test_LightDark_Ada_v2.jl` | `test_ARDESPOT/test_LightDark_ARDESPOT.jl` |

Julia 环境：`Environments_library/Env_POMCPOW` · `Env_Ada` · `Env_ARDESPOT`（LaserTag 需先 `] add https://github.com/JuliaPOMDP/LaserTag.jl.git`）。
输出目录：`data_POMCPOW/` · `data_Ada/` · `data_ARDESPOT/`，**必须在各自目录下启动 Julia**（脚本用相对路径）。

---

## POMCPOW

固定项：`max_depth = 40`、`max_time = 3.0`、`tree_queries = typemax(Int)`、`rng = MersenneTwister(1)`。
POMCPOW 无上下界，`estimate_value` 承担与 AdaOPS/ARDESPOT **下界**相同的角色，必须与它们对齐。

| 实验 | criterion | k_observation | alpha_observation | check_repeat_obs | 其他 | estimate_value（= 下界） |
|---|---|---|---|---|---|---|
| RS11 | `MaxUCB(20.0)` | 10.0 | 0.5 | true | — | `FORollout(RSExitSolver())` |
| RS15 | `MaxUCB(20.0)` | 10.0 | 0.5 | true | — | `FORollout(RSExitSolver())` |
| LaserTag | `MaxUCB(20.0)` | 10.0 | 0.5 | true | `check_repeat_act = true`、`enable_action_pw = false` | 未设置 → 默认随机 rollout（等价于 Ada/ARDESPOT 的 `RandomPolicy` 下界） |
| LightDark | `MaxUCB(20.0)` | 10.0 | **1/15** | **false** | — | `FORollout(FunctionPolicy(s -> s.y < 0 ? 1 : -1))` |

备注：
- 默认值参考 POMCPOW.jl README（`criterion = MaxUCB(1.0)`、`k_observation = 10`、`alpha_observation = 0.5`）；本项目统一把 UCB 设为 20.0 便于横向对比。
- LightDark 连续观测：`alpha_observation = 1/15`、关闭 `check_repeat_obs`。
- 待办：LaserTag 的 `MaxUCB(20.0)` 相对其回报量级（−20..+10）偏大，计划扫 [3, 5, 10, 20]。

---

## AdaOPS

固定项：`max_depth = 40`、`T_max = 3.0`、`zeta = 0.1`、`num_b = 10_000`、`tree_in_info = false`、`rng = MersenneTwister(1)`；
bounds 统一用 `AdaOPS.IndependentBounds(..., check_terminal = true, consistency_fix_thresh = 1e-5)`。

| 实验 | delta | m_min | m_max | grid | 下界 | 上界 |
|---|---|---|---|---|---|---|
| RS11 | 0.3 | 30 | 200 | — | `FORollout(RSExitSolver())` | `FOValue(RSMDPSolver())` |
| RS15 | 0.3 | 30 | 200 | — | `FORollout(RSExitSolver())` | `FOValue(RSMDPSolver())` |
| LaserTag | 0.1 | 50 | 500 | — | `FORollout(RandomPolicy(pomdp, rng = MersenneTwister(3)))` | 常数 `10.0`（= tag_reward） |
| LightDark | 0.1 | 100 | 5000 | `StateGrid(collect(-20.0:1.0:20.0))` | `FORollout(FunctionPolicy(s -> s.y < 0 ? 1 : -1))` | 常数 `10.0`（= correct_r） |

备注：
- `delta` 是主要调参项，大致范围 [0.05, 0.3]；RockSample 取 AdaOPS.jl README 的 0.3。
- LaserTag 的 belief（对手位置）比 RockSample 分散，所以 m_min/m_max 放大到 50/500。
- LightDark 需要 `Base.convert(::Type{SVector{1,Float64}}, s::LightDark1DState)` 才能做 KLD-sampling 计数。
- 消融变体 B（已在脚本中注释）：RS11/RS15 换成 `FORollout(RandomPolicy(rng = MersenneTwister(3)))` + 常数上界 `10/(1-γ) = 200.0`；LightDark 换成常数 −10.0 / 10.0。切变体 B 时必须同步切对应的 ARDESPOT 文件。

---

## ARDESPOT

固定项：`D = 40`、`T_max = 3.0`、`K = 500`、`lambda = 0.01`、`xi = 0.95`、`epsilon_0 = 0.0`、
`max_trials = typemax(Int)`、`tree_in_info = false`、`rng = MersenneTwister(1)`（均为 ARDESPOT.jl 默认值，只改 T_max）；
bounds 统一用 `IndependentBounds(..., check_terminal = true, consistency_fix_thresh = 1e-5)`。

| 实验 | K | 下界 | 上界 | bounds_warnings |
|---|---|---|---|---|
| RS11 | 500 | `DefaultPolicyLB(RSExitSolver())` | `FullyObservableValueUB(RSMDPSolver())` | true |
| RS15 | 500 | `DefaultPolicyLB(RSExitSolver())` | `FullyObservableValueUB(RSMDPSolver())` | true |
| LaserTag | 500 | `DefaultPolicyLB(RandomPolicy(pomdp, rng = MersenneTwister(3)))` | 常数 `10.0` | true |
| LightDark | 500 | `DefaultPolicyLB(FunctionPolicy(s -> s.y < 0 ? 1 : -1))` | 常数 `10.0` | **false** |

备注：
- `K`（场景数）是主要调参项，常用 [100, 500]。
- ⚠️ LightDark + AR-DESPOT 原理上不兼容：DESPOT 按观测精确相等分组，连续观测导致每个观测分支只剩 1 条 scenario，搜索树退化为 K 条独立单粒子轨迹。该文件只作「不兼容性」对照，数字不进正式对比表；`bounds_warnings = false` 是为了避免刷屏。
- 上界口径只要求与 AdaOPS 严格一致（POMCPOW 无上界，属结构性差异）。

---

## 待办

- [ ] `test_Ada/test_RS11_Ada_v2.jl`、`test_RS15_Ada_v2.jl`、`test_LaserTag_Ada.jl` 的 `nb_runs` 仍为 1，出正式数据前改回 100
- [ ] `test_LightDark_POMCPOW.jl` 的 CSV 文件名里 `alphao(1/15)` 含 `/` 会抛 `SystemError`（当前已写成 `1_15`，确认无遗留）
- [ ] LaserTag `MaxUCB` 扫 [3, 5, 10, 20]
- [ ] `test_RoombaL_POMCPOW.jl` 缺 `RoombaPOMDPs` 依赖，暂时搁置
