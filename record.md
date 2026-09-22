# 实验记录 — POMCPOW / AdaOPS / AR-DESPOT

> 超参数以各 `test_*/test_*.jl` 脚本里当前启用（未注释）的值为准，最后核对日期：2026-09-22。
> 详细的轮次日志、结果主表见 `docs/experiment_log.md`；本文件只记「环境 + 超参数」。

---

## 0. 固定实验环境（三个算法、四个问题全部一致，不随实验改变）

基准脚本：`test_POMCPOW/test_RS15_POMCPOW.jl`。以下任意一项改动都要在 `docs/experiment_log.md` 里开新一轮。

| 项目 | 值 | 各包对应字段 |
|---|---|---|
| 搜索深度 | 40 | POMCPOW `max_depth` / AdaOPS `max_depth` / ARDESPOT `D` |
| 每步规划预算 | 3.0 s | POMCPOW `max_time` / AdaOPS `T_max` / ARDESPOT `T_max` |
| 单局最大步数 | 100 | `HistoryRecorder(max_steps = 100)` |
| 重复次数 | 100 | `nb_runs`（12 个脚本当前均为 100；smoke test 时临时设 1，之后必须改回） |
| 跟踪粒子数 | 5000 | `BootstrapFilter(pomdp, 5000, ...)` |
| 迭代次数上限 | 不设限 | POMCPOW `tree_queries` / ARDESPOT `max_trials` = `typemax(Int)`，由时间预算停止 |
| 随机种子 | planner `MersenneTwister(1)`；belief filter `1000 + i`；HistoryRecorder `2000 + i`；全局 `Random.seed!(1)` | 第 i 局 |
| 回报口径 | `discounted_reward(history)`（折扣回报） | — |
| 观测分箱（仅 AR-DESPOT × LightDark） | `OBS_BIN = 1.0` | `BinnedObsPOMDP` wrapper，只作用于 planner 模型 |

### 四个问题的固定属性（均为包默认值，脚本未覆盖）

| 问题 | 构造 | γ | 关键回报 | 观测空间 |
|---|---|---|---|---|
| RS11 | `RockSamplePOMDP(11, 11)` | 0.95 | good rock +10 / bad rock −10 / exit +10 | 离散 |
| RS15 | `RockSamplePOMDP(15, 15)` | 0.95 | 同上 | 离散 |
| LaserTag | `gen_lasertag(rng = MersenneTwister(7), robot_position_known = false)` | 0.95 | tag +10 / 误 tag −10 / 每步 −1 | 离散（DESPOTEmu / DMeas） |
| LightDark | `LightDark1D()` | 0.9 | correct +10 / incorrect −10 | **连续**（AR-DESPOT 经 planner 侧分箱后适用） |

> LaserTag 默认 7×11 地图 → 77² + 1 = **5930 个状态 × 5 个动作**，`transition` 有显式分布、
> `actionindex(p,a)=a`、`reward(p,s,a,sp)` 齐全，所以通用 `ValueIterationSolver` 直接可解（秒级）。
>
> **`ValueIterationSolver(max_iterations = 1000, belres = 1e-4, include_Q = false)` 的取值依据**
> ——不是来自任何论文，是按 VI 的收敛速度算的；包默认值是 `100 / 1e-3 / true`：
> - `max_iterations`：`‖V_n − V*‖ ≤ γⁿ‖V₀ − V*‖`。LaserTag γ = 0.95、值域约 [−20, +10]，
>   要压到 1e-4 需 `n ≥ ln(1e-4/30)/ln 0.95 ≈ 248` 轮。**包默认的 100 轮会在残差
>   ≈ 0.95¹⁰⁰×30 ≈ 0.18 处被截断**——这才是必须改的一项，1000 是留足余量（VI 会先撞 belres 停）。
> - `belres`：1e-4 相对 ±10 的回报量级是 1e-5 的相对误差，足够。改回默认 1e-3 差别可忽略
>   （约少跑 45 轮），**这一项不关键**。
> - `include_Q = false`：`FOValue` 只调 `value(policy, s)`，不需要 Q 矩阵；与 RockSample.jl
>   `RSMDPSolver` 同款写法。**副作用：`action(policy, s)` 不可用**——将来若改写成
>   `FORollout(ValueIterationSolver(...))` 会直接报错，那时必须打开 `include_Q`。

> γ 跨问题不同是 benchmark 自身属性，**绝对回报不可跨问题比较**，只在同一问题内比三个算法。
> 深度 40 对 γ = 0.95 截断约 13% 的价值（0.95⁴⁰ ≈ 0.129），三个算法同等截断。

### 运行环境 / 文件对照

| 问题 | POMCPOW | AdaOPS | AR-DESPOT |
|---|---|---|---|
| RS11 | `test_POMCPOW/test_RS11_POMCPOW.jl` | `test_Ada/test_RS11_Ada_v2.jl` | `test_ARDESPOT/test_RS11_ARDESPOT.jl` |
| RS15 | `test_POMCPOW/test_RS15_POMCPOW.jl` | `test_Ada/test_RS15_Ada_v2.jl` | `test_ARDESPOT/test_RS15_ARDESPOT.jl` |
| LaserTag | `test_POMCPOW/test_LaserTag_POMCPOW.jl` | `test_Ada/test_LaserTag_Ada.jl` | `test_ARDESPOT/test_LaserTag_ARDESPOT.jl` |
| LightDark | `test_POMCPOW/test_LightDark_POMCPOW.jl` | `test_Ada/test_LightDark_Ada_v2.jl` | `test_ARDESPOT/test_LightDark_ARDESPOT.jl` |
| RoombaB | `test_POMCPOW/test_RoombaB_POMCPOW.jl` | `test_Ada/test_RoombaB_Ada.jl` | `test_ARDESPOT/test_RoombaB_ARDESPOT.jl` |
| RoombaL | `test_POMCPOW/test_RoombaL_POMCPOW.jl` | `test_Ada/test_RoombaL_Ada.jl` | `test_ARDESPOT/test_RoombaL_ARDESPOT.jl` |

Julia 环境：`Environments_library/Env_POMCPOW` · `Env_Ada` · `Env_ARDESPOT`。
- LaserTag 需先 `] add https://github.com/JuliaPOMDP/LaserTag.jl.git`（三个环境都要）。
- **`Env_POMCPOW` 需 `] add DiscreteValueIteration`**（2026-09-22 新增：LaserTag 的 MDP estimator 用它解 `UnderlyingMDP`。
  它只是 RockSample.jl 的间接依赖，不显式 add 会报 `not found in current path`）。
输出目录：`data_POMCPOW/` · `data_Ada/` · `data_ARDESPOT/`，**必须在各自目录下启动 Julia**（脚本用相对路径）。

---

## 0.5 超参数分层：哪些固定、哪些按问题调

后面三张表要配合这个分层读。**只有第三层是真正的调参旋钮**，前两层跨实验一律固定。

### 第一层：协议参数 —— 全部实验、全部算法锁死

即上面第 0 节的七项，加上 `tree_in_info = false`、bounds 的 `check_terminal = true` 与
`consistency_fix_thresh = 1e-5`。这层不是超参数，是「给多少资源、怎么量结果」。
改任意一项都要在 `docs/experiment_log.md` 里开新的一轮。

### 第二层：由问题性质决定 —— 按明文规则固定，不许当旋钮调

这层参数跨问题确实取不同值，但那**不是调参，是算法要正确实例化的前提**。
它们由规则决定，不该试出来：

| 算法 | 参数 | 规则 |
|---|---|---|
| POMCPOW | `check_repeat_obs` | 离散观测 → `true`；连续观测 → `false` |
| POMCPOW | `check_repeat_act` | 离散动作 → `true` |
| POMCPOW | `enable_action_pw` | 动作空间小且离散 → `false`（不对动作做渐进加宽） |
| AdaOPS | `grid` | 仅连续状态需要（LD 用 `StateGrid`）；离散状态不设 |
| AdaOPS | `Base.convert` 方法 | 连续状态需要 `convert(::Type{SVector{N,Float64}}, s)` 才能做 KLD-sampling 计数 |
| AdaOPS | `m_min` / `m_max` | 按该问题 belief 的宽度定（LaserTag 的对手位置比 RockSample 分散 → 50/500），**扫参时不动** |
| ARDESPOT | `bounds_warnings` | 默认 `true`；仅在已知界不一致会刷屏时关掉 |
| ARDESPOT | `OBS_BIN`（LD 专有） | 连续观测必须分箱才能用 DESPOT；宽度本身属第三层 |
| 三者 | bounds 变体 A / B | **同一问题的三个文件必须同步切** |

**上下界属于这一层，不属于第三层。** 它们不是用来刷分的旋钮，而是保证三个算法拿到同等
先验信息的契约；换个更强的下界去提高某一列的分数，等于给那个算法开小灶。
各格的具体口径、以及 2026-09-22 起 RS / LaserTag 上刻意打破统一的那一处，见 §「叶子估值口径」。

> 唯一无法消除的实现层差异：DESPOT 在**确定化的 K 条 scenario** 上算界，
> AdaOPS 在**带权自适应粒子集**上算界，估计量的方差结构不同，属算法结构性差异。

### 第三层：真正的调参旋钮 —— 按问题调，但投入必须对等

**每个算法只认一个主旋钮**，每个问题扫同样数量的值（4–5 个点），用与正式实验**分开**的
调参种子（建议 filter `5000+i`、recorder `6000+i`），扫参结果全部留档。
这样「三个算法调参投入对等」才是一句可核实的话。

| 算法 | 主旋钮（每问题扫同样数量的值） | 次要旋钮（锁在包默认值，声明不调） |
|---|---|---|
| POMCPOW | `criterion` 的 UCB 常数 c | `k_observation` = 10.0 · `alpha_observation`（离散 0.5，连续另定） |
| AdaOPS | `delta`（约 [0.05, 0.3]） | `zeta` = 0.1 · `num_b` = 10000 |
| AR-DESPOT | `K` | `lambda` = 0.01 · `xi` = 0.95 · `epsilon_0` = 0.0 |

**当前调参投入并不对等**，出正式对比前需要拉平：

| 算法 | 主旋钮现状 |
|---|---|
| POMCPOW | 已从默认 1.0 改到 20.0，但**未扫参**；且 RS 上疑似过大（见下方零回报问题） |
| AdaOPS | RS 用 README 原值 0.3，LaserTag / LD 调到 0.1，**未系统扫参** |
| AR-DESPOT | 四个问题全为默认 `K = 500`，**完全未调** |

---

## POMCPOW

固定项：`max_depth = 40`、`max_time = 3.0`、`tree_queries = typemax(Int)`、`rng = MersenneTwister(1)`。
**POMCPOW 没有上界也没有下界**，只有一个叶子点估值 `estimate_value`（BasicPOMCP docstring:
"an initial *unbiased* estimate of the value at belief node h"）。它不参与任何剪枝或最优性保证，
估歪了只是有偏，不会报错。所以下表里「= 上界 / 同 lower 策略」是**口径对照**——说的是
「这个量与 AdaOPS/ARDESPOT 的哪一侧界数值上是同一个东西」，不是说 POMCPOW 有界。
当前各格不再统一，见 §「叶子估值口径」。

| 实验 | criterion | k_observation | alpha_observation | check_repeat_obs | 其他 | estimate_value（当前启用） |
|---|---|---|---|---|---|---|
| RS11 | `MaxUCB(1.0)` | 10.0 | 0.5 | true | — | **`FOValue(RSMDPSolver())`**（MDP 值，= 上界） |
| RS15 | `MaxUCB(1.0)` | 10.0 | 0.5 | true | — | **`FOValue(RSMDPSolver())`**（MDP 值，= 上界） |
| LaserTag | `MaxUCB(1.0)` | 10.0 | 0.5 | true | `check_repeat_act = true`、`enable_action_pw = false` | **`FOValue(ValueIterationSolver(1000, 1e-4))`**（MDP 值，= 上界） |
| LightDark | `MaxUCB(1.0)` | 10.0 | 0.5 | **false** | — | `FORollout(FunctionPolicy(s -> s.y < 0 ? 1 : -1))`（同另两算法的 lower 策略） |
| RoombaB | `MaxUCB(1.0)` | 10.0 | 0.5 | true | `check_repeat_act = true`、`enable_action_pw = false` | **`FOValue(RoombaMDPUpper(pomdp))`**（闭式 MDP 启发式，= Ada 的 upper） |
| RoombaL | `MaxUCB(1.0)` | 10.0 | **1/15** | **false** | 同上 | **`FOValue(RoombaMDPUpper(pomdp))`**（同上） |

备注：
- R1（2026-09-17）：UCB 常数统一回到 POMCPOW.jl README 默认值 **1.0**（原为 20.0）。
- LightDark 的 `alpha_observation` 实际是 **0.5**，不是 1/15（本文件 09-17 版记错；09-20 已把 CSV 文件名里的
  `alphao(1_15)` 改成 `alphao(0.5)` 与实参一致，参数本身没动）。连续观测的放慢加宽目前只在 **RoombaL** 上用 1/15。
- 2026-09-17：LaserTag 原先未设 `estimate_value`，靠包默认随机 rollout（rng 由包内部构造，不是统一种子 3），已改为显式写出。
- 2026-09-20：删除 `test_RoombaL2_POMCPOW.jl`（漏写 `enable_action_pw=false` 等四处错误）；
  给 `test_RoombaL_POMCPOW.jl` 补上未定义的 `step_time`（修 bug，非调参）。
- 待办：UCB 常数尚未扫参，见 §0.5 第三层。

---

## 叶子估值口径（2026-09-22 起不再全线统一，报告必须逐格写明）

> 起因：POMCGraphSearch.jl issue #1 —— 「不给叶子一个 MDP value estimator，POMCPOW 在 RockSample 上
> 即便 500000 次模拟也复现不出论文成绩」。原因：随机 / 固定策略 rollout 在稀疏奖励 + 长时域下
> 叶子值彼此几乎相等，UCB 看到的 Q 差全是噪声，树退化成近似均匀展开。
> 完整论证见 `work_record/wk_6/estimate_value_wk_6.md`。

**关键事实：MDP value estimator 是上界，不是下界。**
对任意 belief b，`E_{s~b}[V*_MDP(s)] ≥ V*_POMDP(b)`（完全可观测只会更好，即 QMDP / hindsight 界）。
ARDESPOT 把它命名为 `FullyObservableValueUB`、AdaOPS 用 `FOValue` 放 upper 槽，都是这个意思。
所以它**只能进 `upper` 槽，绝不能进 `lower` 槽**——下界必须是某个可执行策略的可达值，
放上界进去会让 AdaOPS 的 `excess_uncertainty` 剪枝和 DESPOT 的 regret bound 全部失效。

POMCPOW 的 `estimate_value` 语义上要的是**无偏点估计**，不是界。塞上界进去的三个后果：
1. 叶子被系统性高估，且不确定性越大高估越多 —— 信息收集动作拿不到任何信用（QMDP 盲点）。
2. Q 值量纲变大 → `MaxUCB(1.0)` 相对变小 → 实际更贪心，c 值得另外扫一遍。
3. RockSample 能扛住（探测收益在树前几层就能被采样到）；**LightDark 是这个盲点的教科书反例，故明确不换**。

各格现状：

| 问题 | POMCPOW `estimate_value` | AdaOPS `lower / upper` | ARDESPOT `lower / upper` | 三算法是否同口径 |
|---|---|---|---|---|
| RS11 / RS15 | `FOValue(RSMDPSolver())` = 上界（精确 MDP 值） | `FORollout(RSExitSolver())` / `FOValue(RSMDPSolver())` | `DefaultPolicyLB(RSExitSolver())` / `FullyObservableValueUB(RSMDPSolver())` | ✅ 同一对界，POMCPOW 取上沿 |
| LaserTag | `FOValue(ValueIterationSolver)` = 上界（精确 MDP 值） | `FORollout(RandomPolicy(seed 3))` / 常数 10.0 | `DefaultPolicyLB(RandomPolicy(seed 3))` / 常数 10.0+1e-3 | ❌ 仅 POMCPOW 换了，**本轮只作内部对照** |
| LightDark | `FORollout(朝光源启发式)`（同 lower 策略） | `FORollout(同启发式)` / 常数 10.0 | `DefaultPolicyLB(同启发式)` / 常数 10.0 | ✅ 三边同策略 |
| RoombaB | `FOValue(RoombaMDPUpper)` = 上界（乐观闭式） | `FORollout(RandomPolicy(seed 3))` / `FOValue(RoombaMDPUpper)` | 同下界 / 常数 10.0+1e-3 | ⚠ POMCPOW 与 Ada 的 upper 同量；ARDESPOT 仍是常数 |
| RoombaL | `FOValue(RoombaMDPUpper)` = 上界（乐观闭式） | `FORollout(RandomPolicy(seed 3))` / 常数 10.0 | 同下界 / 常数 10.0+1e-3 | ❌ 三边上界各不相同，**仅作 POMCPOW 内部对照** |

CSV 标签：换了 MDP estimator 的 POMCPOW 跑次一律用 **`MDP_esti_trial`** 后缀，
与 R1 / R1b 分开存，报告里单列一行「POMCPOW + MDP estimator」，**不要与 R1 混**。

各包的包装类型名不同但语义相同：
`FORollout`（AdaOPS / BasicPOMCP / POMCPOW）≡ `DefaultPolicyLB`（ARDESPOT）——同一策略的 rollout；
`FOValue`（AdaOPS / POMCPOW）≡ `FullyObservableValueUB`（ARDESPOT）——同一个完全可观测 MDP 值。

---

## AdaOPS

固定项：`max_depth = 40`、`T_max = 3.0`、`zeta = 0.1`、`num_b = 10_000`、`tree_in_info = false`、`rng = MersenneTwister(1)`；
bounds 统一用 `AdaOPS.IndependentBounds(..., check_terminal = true, consistency_fix_thresh = 1e-5)`。

| 实验 | delta | m_min | m_max | grid | 下界 | 上界 |
|---|---|---|---|---|---|---|
| RS11 | 0.3 | 30 | 200 | — | `FORollout(RSExitSolver())` | `FOValue(RSMDPSolver())` |
| RS15 | 0.3 | 30 | 200 | — | `FORollout(RSExitSolver())` | `FOValue(RSMDPSolver())` |
| LaserTag | 0.3 | 50 | 500 | — | `FORollout(RandomPolicy(pomdp, rng = MersenneTwister(3)))` | 常数 `10.0`（= tag_reward） |
| LightDark | 0.3 | 100 | 5000 | `StateGrid(collect(-20.0:1.0:20.0))` | `FORollout(FunctionPolicy(s -> s.y < 0 ? 1 : -1))` | 常数 `10.0`（= correct_r） |
| RoombaB | 0.3 | 100 | 1000 | `StateGrid(collect(1.5:1.0:4001.5))`（状态索引每个一个 bin） | `FORollout(RandomPolicy(pomdp, rng = MersenneTwister(3)))` | `FOValue(RoombaMDPUpper(pomdp))`（闭式，标签 **R1b**） |
| RoombaL | 0.3 | 100 | 1000 | 同上 | `FORollout(RandomPolicy(pomdp, rng = MersenneTwister(3)))` | 常数 `10.0`（= goal_reward，标签 R1） |

备注：
- R1（2026-09-17）：四个问题的 `delta` 统一为 **0.3**（AdaOPS.jl README RockSample 例子的值；LaserTag/LD 原为 0.1）。`delta` 是主旋钮，范围约 [0.05, 0.3]，R2 扫参。
- LaserTag 的 belief（对手位置）比 RockSample 分散，所以 m_min/m_max 放大到 50/500。
- LightDark 需要 `Base.convert(::Type{SVector{1,Float64}}, s::LightDark1DState)` 才能做 KLD-sampling 计数。
- 消融变体 B（已在脚本中注释）：RS11/RS15 换成 `FORollout(RandomPolicy(rng = MersenneTwister(3)))` + 常数上界 `10/(1-γ) = 200.0`；LightDark 换成常数 −10.0 / 10.0；Roomba 换成「朝目标直行」的启发式下界。切变体 B 时必须同步切对应的 ARDESPOT 文件。
- **Roomba 两个脚本加了 `default_action = RoombaAct(0.0, 0.0)`（2026-09-20）**：AdaOPS 建树前 `strip_terminals` 若把 belief 全部清零就 `error("All states in the current belief are terminal.")`。Roomba 转移是 `Deterministic`、观测无过程噪声、`BootstrapFilter` 不扰动粒子 → 粒子只减不增，跑几十步后常塌成同一个状态，一起走进 goal/stairs 时整个 belief 全终止（真实状态还没终止）。`default_action` 只在原本会崩的那一步生效（原地停一步），其余步策略与随机数流不变。**这是止血不是根治**；根治方案是包自带的 `RoombaParticleFilter`（注入动作噪声 + 跳过终止粒子），但那改的是 belief 更新本身，要六个 Roomba 脚本同步改并重跑，本轮不做。
- **Roomba 的 `Base.convert` 是从 `Int` 出发的一维网格**：`Base.convert(::Type{SVector{1,Float64}}, s::Int) = SVector{1,Float64}(s)` + `StateGrid(1.5:1.0:4001.5)`，每个离散状态索引自成一个 bin —— 正是 KLD-sampling 要数的东西（旧的 `SVector{3}` 版本按 (x,y,θ) 分箱，已随 `_Ada_v2` 文件一起删除）。
- **AdaOPS × Roomba 搜索深度只有 1（2026-09-20 实测，未解决）**：`Times of exploration: 1` / `Depth of exploration: 1.00`，退化成「一步前瞻 + 随机 rollout」，与 POMCPOW 的对比不成立。瓶颈是单次 expand 的成本：286 动作 × `m_max=1000` 粒子 ≈ 28.6 万次 generative 调用，Lidar 版再按 δ 分出 ~21 个观测分支 → 12000+ 个子 belief 各算一次界（下界是 40 步 rollout）≈ 50 万步模拟。**RoombaL 试过换紧上界（闭式启发式），深度仍是 1，已回退常数 10.0**；RoombaB 保留闭式上界作对照（标签 R1b）。结论固化：换上界不是解，要动成本项 —— 按杠杆排 `m_max` 1000→100 → `delta` 0.3→1.0 → 缩小动作空间 286→15（最后一项要六个 Roomba 脚本同步改）。
- AdaOPS 自身的超时警告（`planner.jl`，单步 > `timeout_warning_threshold = 2·T_max` 时 dump 整棵树）**与 `tree_in_info` 无关、关不掉**，只能用 `timeout_warning_threshold = 1e9` 消掉症状。

---

## ARDESPOT

固定项：`D = 40`、`T_max = 3.0`、`K = 500`、`lambda = 0.01`、`xi = 0.95`、`epsilon_0 = 0.0`、
`max_trials = typemax(Int)`、`tree_in_info = false`、`rng = MersenneTwister(1)`（均为 ARDESPOT.jl 默认值，只改 T_max）；
bounds 统一用 `IndependentBounds(..., check_terminal = true, consistency_fix_thresh = 1e-5)`。

| 实验 | K | 下界 | 上界 | bounds_warnings |
|---|---|---|---|---|
| RS11 | 500 | `DefaultPolicyLB(RSExitSolver())` | `FullyObservableValueUB(RSMDPSolver())` | true |
| RS15 | 500 | `DefaultPolicyLB(RSExitSolver())` | `FullyObservableValueUB(RSMDPSolver())` | true |
| LaserTag | 500 | `DefaultPolicyLB(RandomPolicy(pomdp, rng = MersenneTwister(3)))` | 常数 `10.0 + 1e-3` | true |
| LightDark | 500 | `DefaultPolicyLB(FunctionPolicy(s -> s.y < 0 ? 1 : -1))` | 常数 `10.0` | true |
| RoombaB | 500 | `DefaultPolicyLB(RandomPolicy(pomdp, rng = MersenneTwister(3)))` | 常数 `10.0 + 1e-3` | true |
| RoombaL | 500 | `DefaultPolicyLB(RandomPolicy(planner_pomdp, rng = MersenneTwister(3)))` | 常数 `10.0 + 1e-3` | true |

备注：
- `K`（场景数）是主要调参项，常用 [100, 500]。
- **LightDark 观测分箱（2026-09-17 改）**：DESPOT 按观测精确相等分组，连续观测会让每个观测分支只剩 1 条 scenario，树退化为 K 条独立单粒子轨迹。现用 `BinnedObsPOMDP` wrapper 把观测离散成 bin 索引，**只交给 planner**；`HistoryRecorder` 的仿真环境与 `BootstrapFilter` 仍用原始连续 `LightDark1D()`。问题本身没变，三列仍可比；离散化算作算法的一部分，报告时须写明这是「AR-DESPOT + 观测分箱」变体。
  - `OBS_BIN = 1.0`（依据：判定成功容差 |y| < 1，分辨率与决策精度同量级），属第三层调参项，扫 [0.5, 1.0, 2.0]；设 `OBS_BIN = nothing` 可退回退化基线做对照。
  - 观测噪声随 y 变化（离光源越远越糊），固定宽度必然有偏；文件内注释了按噪声尺度缩放的**相对分箱**变体。
  - 同时修正了该文件把 `DESPOTSolver` 的 `D` 误写成 `max_depth` 的问题（`@with_kw` 会判为未知关键字直接报错），并把 `bounds_warnings` 恢复为 `true`。
- **LaserTag 上界放松（2026-09-18）**：R1 中途抛 `lower and upper bounds for the root belief were both 10.0, so no tree was created`。常数上界 10.0 是**可达**的，当 belief 收敛到「下一步 tag 必中」时 rollout 下界也会到 10.0，gap 塌成 0。上界改为 `10.0 + 1e-3`（大于 `consistency_fix_thresh`），gap 恒 > 0；下界保持随机 rollout 不变，**三方下界一致性未被破坏**。另加 `default_action = 5`（TAG）兜底，规划失败不会中断整局。
- **Roomba（2026-09-20 新增）**：两个脚本都加了 `default_action = RoombaAct(0.0, 0.0)` 兜底；RoombaL 带只作用于 planner 的观测分箱 wrapper `BinnedObsRoomba`，`OBS_BIN = 0.5`（扫 [0.25, 0.5, 1.0]），仿真环境与 `BootstrapFilter` 仍用原始连续 pomdp。286 个动作 × K=500 个 scenario 的 `expand!` 成本极高，**K = 100 的对照必须一并跑**，否则容易得出「DESPOT 不适合 Roomba」的错误结论。
- 上界口径只要求与 AdaOPS 严格一致（POMCPOW 无上界，属结构性差异；2026-09-22 起 RS / LaserTag 的 POMCPOW 叶子估值实际取的就是这个上界，见 §叶子估值口径）。
- **对比公平性（报告必须写明）**：POMCPOW 靠 UCB 每次只碰**一个**动作，AdaOPS / AR-DESPOT 每次 expand 都要遍历**全部**动作。286 个动作的 Roomba 对后两者是结构性劣势，不是调参能抹平的。

---

## R1（第 1 轮，初始版）配置口径

三个主旋钮一律取**包默认 / README 值**，跨四个问题统一，不做任何按问题的调参 —— 即 §0.5 里说的
「主表：默认配置（零调参）」。CSV 文件名统一带 `_R1` 后缀。

| 算法 | 主旋钮 | R1 取值 | 来源 |
|---|---|---|---|
| POMCPOW | `criterion` | `MaxUCB(1.0)` | POMCPOW.jl README 默认值 |
| AdaOPS | `delta` | 0.3 | AdaOPS.jl README（RockSample 例子） |
| AR-DESPOT | `K` | 500 | ARDESPOT.jl 默认值；DESPOT 论文 §5.1「we chose K = 500」 |

次要旋钮（锁定，声明不调）：`lambda = 0.01`、`xi = 0.95`、`epsilon_0 = 0.0`（ARDESPOT.jl 默认值）。
注：DESPOT 论文 JAIR 版 Table 2 在 RockSample 上选的是 `lambda = 0.0`、LaserTag 为 `0.01`；R1 未采用，留待 R2。

| 状态 | 问题 × 算法 | 情况 |
|---|---|---|
| ✅ 已出数 | RS11 / RS15 / LaserTag / LightDark × POMCPOW、AdaOPS；RS11 / RS15 × AR-DESPOT | — |
| ❌ 未出数 | LaserTag × AR-DESPOT | 跑到中途报错 |
| ❌ 未出数 | LightDark × AR-DESPOT | 观测分箱尚未正常工作 |

第二层（由问题性质决定）的参数仍按各自规则取值，未统一：
POMCPOW 的 `alpha_observation`（LD = 1/15）、`check_repeat_obs`；
AdaOPS 的 `m_min`/`m_max`（RS 30/200、LaserTag 50/500、LD 100/5000）与 `grid`；
ARDESPOT 的 `OBS_BIN`（LD = 1.0）。

---

## Roomba (RoombaPOMDPs.jl) — 2026-09-19 新增

**安装**：`RoombaPOMDPs` 没有注册进 General registry，必须
`] add https://github.com/sisl/RoombaPOMDPs.git`（包名是复数）。

**问题实例**（六个文件逐字一致，与 `test_POMCGS/experiments_online/test_{Lidar,Bumper}Roomba.jl` 对齐）：
`config = 3`，`DiscreteRoombaStateSpace(25, 16, 10)`，`v_max = 5.0`，
aspace = 286 个离散动作（26 档速度 `0:0.2:5` × 11 档转向 `-1:0.2:1`）。

> 注意：`RoombaPOMDPs` 默认的 `aspace = RoombaActions()` 是**空标记 struct**，
> 没有 `rand` / `length` / `iterate`，POMCPOW 的动作采样和 `RandomPolicy` 都用不了。
> 想做连续动作变体必须自己补 `Base.rand(rng, ::RoombaActions)`，并单独声明为一个变体。

**奖励**：`time_pen = -0.1` / 步，`contact_pen = -1.0`（新撞墙），
`goal_reward = +10`（终止），`stairs_penalty = -10`（终止），`discount = 0.95`。
单步最大 = 9.9 < 10.0，所以常数上界 10.0 严格可用，不会出现 LaserTag 那种 gap 塌成 0。

| 实验 | 下界 | 上界 | 观测类型 | 观测处理 |
|---|---|---|---|---|
| RoombaL (Lidar) | `RandomPolicy(pomdp, rng = MersenneTwister(3))` | 常数 `10.0` | 连续 Float64 | POMCPOW: `alpha_observation = 1/15` + `check_repeat_obs = false`；AR-DESPOT: `BinnedObsPOMDP`，`OBS_BIN = 0.5`；AdaOPS: 无（δ-packing 原生支持） |
| RoombaB (Bumper) | `RandomPolicy(pomdp, rng = MersenneTwister(3))` | 常数 `10.0` | 离散 Bool（2 个） | 无；POMCPOW 观测参数回 README 默认（k=10, α=0.5, 开查重） |

**动作 PW**：三个算法都不做动作渐进加宽（POMCPOW `enable_action_pw = false` +
`check_repeat_act = true`），与 POMCGS 的 `bool_APW = false` 对齐。
代价是 286 个动作在每个节点全展开，3 s 内 trial 数会很少 —— 六个脚本都记了
`sec_per_step`，第一次出数前先看这个数。

**状态类型的坑（2026-09-19）**：用 `DiscreteRoombaStateSpace` 时，这个 POMDP 的
**状态类型是 `Int`（索引），不是 `RoombaState`**：

```julia
RoombaPOMDP{SS, AS, S, T, O} <: POMDP{S, RoombaAct, O}
Base.eltype(::Type{DiscreteRoombaStateSpace}) = Int
POMDPs.states(m) = vec(1:n_states(m))   # n_states = 25*16*10 + 2 = 4002
```

索引与 `RoombaState(x, y, theta, status)` 之间靠 `convert_s(RoombaState, si, m)` /
`convert_s(Int, s, m)` 互转（编码是 `1 + xi + num_x*yi + num_x*num_y*thi`，
最后两个索引 `states_num` / `states_num - 1` 是到达目标 / 掉楼梯）。这一条坑了两处：

- `test_Ada/test_Roomba*_Ada_v2.jl` 原来写 `Base.convert(::Type{SVector{3,Float64}}, ::RoombaState)`
  → 报 `MethodError: Cannot convert an object of type Int64 to an object of type SVector{3, Float64}`。
  已改成对 `::Int` 定义，并先 `convert_s` 解码回 (x, y, theta) —— 索引本身在网格上没有几何意义
  （x 快变、theta 慢变），直接拿索引分箱是错的。`convert_s` 需要模型而 `Base.convert` 拿不到，
  所以用一个 `const _ADAOPS_GRID_MODEL = pomdp` 捞进来（对 `Int` 的类型盗用，只在独立脚本里可接受）。
- `test_ARDESPOT/test_RoombaL_ARDESPOT.jl` 的 `BinnedObsPOMDP` 原来声明成
  `<: POMDP{RoombaState, RoombaAct, Int}`，已改成 `POMDP{Int, RoombaAct, Int}`，并补了
  `states` / `stateindex` 转发。

**AdaOPS 专有（2026-09-20 改）**：`Base.convert(::Type{SVector{1,Float64}}, s::Int)` +
`StateGrid(collect(1.5:1.0:4001.5))` —— 按**状态索引**做一维网格，每个离散状态自成一个 bin。
（旧的「按 (x,y,θ) 三维分箱」方案已随 `_Ada_v2` 文件一起删除。）
`m_min/m_max` 取 **100 / 1000**（不是 100/5000：286 个动作 × 3 s 预算，粒子再多会跑不满深度）。

**文件清单（2026-09-22 核对）**：
- `test_POMCPOW/test_RoombaB_POMCPOW.jl`、`test_RoombaL_POMCPOW.jl` —— 保留，R1 已全部跑完出数
- `test_Ada/test_RoombaB_Ada.jl`、`test_RoombaL_Ada.jl` —— **2026-09-20 重写新建**（旧的 `_Ada_v2` 三个文件已删）
- `test_ARDESPOT/test_RoombaB_ARDESPOT.jl`、`test_RoombaL_ARDESPOT.jl` —— **2026-09-20 重写新建**（旧的三个文件已删）
- ~~`test_POMCPOW/test_RoombaL2_POMCPOW.jl`~~ —— 2026-09-20 删除（漏写 `enable_action_pw=false`、缺 `check_repeat_act`、
  下界错用 RockSample 专用的 `FORollout(RSExitSolver())`、CSV 写入被注释掉；`test_RoombaL_POMCPOW.jl` 是四点都正确的版本）

**AdaOPS / AR-DESPOT 这两列的状态（2026-09-22）：脚本齐了，但正式实验暂缓。**
`data_Ada/` 和 `data_ARDESPOT/` 里仍没有 Roomba 的 CSV。搁置范围仅限这四个 Roomba 脚本的满跑，
其他问题（RS11 / RS15 / LaserTag / LightDark）不受影响。重启时直接从这几条继续：

- **AdaOPS**：`convert` 与 `StateGrid` 的坑已解决，belief 全终止崩溃已用 `default_action` 止血，
  但**搜索深度仍是 1**（换紧上界无效，已验证）。重启第一步：两个脚本各跑 1 run 冒烟，
  看 `Times of exploration` / `Depth of exploration` 有没有动；再按杠杆依次试
  `m_max` 1000→100 → `delta` 0.3→1.0 → 动作空间 286→15。
- **AR-DESPOT**：`expand!` 要遍历**全部 286 个动作 × K 个 scenario**（每展开一个节点约 14.3 万次 `gen`，
  还要给每个子节点算 rollout 下界），3 s 内树会很浅。**K = 100 的对照必须一并跑**。
  另注意旧版观察到的症状：每步耗时极短、各局 return 完全相同 = `action_info` 的 `try/catch`
  静默吞异常、每步回落到 `default_action`（源码这条路径不打印警告），冒烟时要专门排查。
- 重启顺序：**先把 POMCPOW 的 MDP estimator 验证完**（跑得通、闭环快），启发式被证明有效之后
  再回头调这两列 —— 那时上界已经有可信参照，调 `m_max` / `delta` / 动作空间才有判断依据。

**结构性差异（报告必须写明）**：Lidar 上三条线的观测处理本来就不是同一种
（POMCPOW 放慢加宽 / AR-DESPOT 分箱 / AdaOPS 原生），Bumper 上才是完全对齐的。
所以"Lidar vs Bumper"这一对受控对照，只在同一个算法内部才是干净的。
Roomba 这一行目前实际出数的只有 **POMCGS vs POMCPOW** 两列；AdaOPS / AR-DESPOT 的脚本已就位但未满跑。

**问题一览 PDF**：`docs/benchmark_problems.pdf`（S/A/Z 维度、奖励机制、实验意图，
以及上下界口径与统一协议的字段对照表）。

## 待办

**优先：MDP estimator 这条线（2026-09-22 开）**

- [ ] `test_RS11_POMCPOW.jl` / `test_RS15_POMCPOW.jl`：先 `nb_runs = 1` 冒烟，确认
      `solve(solver, pomdp)` 能建表且不 OOM（RS15 约 737 万状态，util + policy 两个数组约 120 MB，预计数十秒），
      再跑满 100 runs
- [ ] `test_LaserTag_POMCPOW.jl`：同样先冒烟（5930 状态 × 5 动作，VI 秒级），注意 `@warn_requirements` 有无缺项
- [ ] 三份结果在 `docs/experiment_log.md` 里单列「POMCPOW + MDP estimator」一行，**不与 R1 混**
- [x] ~~Roomba 的 POMCPOW 换 `FOValue(RoombaMDPUpper(pomdp))`~~ 2026-09-22 完成（两个脚本，
      `RoombaMDPUpper` 定义与 `test_Ada/test_RoombaB_Ada.jl` 逐字相同，**改一处必须同步改另一处**）
- [ ] `test_RoombaB_POMCPOW.jl` / `test_RoombaL_POMCPOW.jl`：冒烟 1 run，确认 `FOValue` 走通
      （状态是 Int 索引，`value(::RoombaMDPUpper, ::Int)` 要能被分派到）
- [ ] 那是**乐观闭式**而非精确 MDP 值；4002 状态 × 286 动作的真 VI 也够得着（一轮 ≈ 1.1M 个 (s,a)、
      转移确定性），但要先确认自定义 aspace 上有 `actionindex`。值得小规模试一次做对照
- [ ] 若 MDP estimator 明显优于 R1，把同一个 `FOValue` 换进 LaserTag / Roomba 的
      AdaOPS·ARDESPOT `upper` 槽，使三算法同口径
- [ ] 换了 estimator 后 `MaxUCB` 的 c 要重扫：Q 值量纲变大会让 c=1.0 相对变小、实际更贪心
- [ ] **LightDark 明确不换 MDP estimator**（QMDP 盲点的教科书反例）。那边值得改的是下界：
      现在的 `FunctionPolicy(s -> s.y < 0 ? 1 : -1)` 永不执行 `a = 0`，rollout 跑满 40 步回报 ≈ 0；
      改成 `s -> abs(s.y) < 1 ? 0 : (s.y < 0 ? 1 : -1)` 才有信息（三个 LightDark 文件同步换）

**R1 收尾**

- [ ] `test_LaserTag_ARDESPOT.jl`：上界已放松为 `10.0 + 1e-3` 并加 `default_action = 5`，需重跑补上这一格
- [ ] `test_LightDark_ARDESPOT.jl`：观测分箱（`BinnedObsPOMDP`，`OBS_BIN = 1.0`）调通后补 R1 数据
- [ ] 十二个既有脚本都还没做过冒烟测试（每个 1 run），改完注释后未实际运行过
- [ ] `Env_ARDESPOT` 仍需 `] add https://github.com/JuliaPOMDP/LaserTag.jl.git`

**R2 调参（§0.5 第三层）**

- [ ] 拉平调参投入：POMCPOW 扫 `MaxUCB` [1, 3, 5, 10]、AdaOPS 扫 `delta` [0.05, 0.1, 0.2, 0.3]、
      AR-DESPOT 扫 `K` [125, 250, 500, 1000]，各问题同等点数、用调参专用种子（filter `5000+i` / recorder `6000+i`）
- [ ] 核 `lambda`：论文 RS = 0.0、LaserTag = 0.01，与当前统一的 0.01 不符
- [ ] LightDark 分箱宽度扫 [0.5, 1.0, 2.0]，并与 `OBS_BIN = nothing` 的退化基线对照
- [ ] 三套脚本加单局计时，确认 3 s 预算是否被用满（目前只有两个 Roomba POMCPOW 脚本记了 `sec_per_step`）

**Roomba**

- [ ] AdaOPS × Roomba：两个脚本各跑 1 run 冒烟，看 `Times of exploration` / `Depth of exploration` 有没有动；
      深度仍为 1 就按杠杆动成本项（`m_max` 1000→100 → `delta` 0.3→1.0 → 动作空间 286→15）
- [ ] AR-DESPOT × Roomba：K = 100 的对照；RoombaL 扫 `OBS_BIN` [0.25, 0.5, 1.0]
- [ ] AdaOPS × RoombaL 扫 `delta` [0.1, 0.3, 1.0]
- [ ] Roomba 下界 R2 换成「朝目标直行」的启发式（变体 B 注释块已写好，六个脚本同步切）；
      `RoombaPOMDPs.get_goal_xy` 已确认导出，可安全启用
- [ ] 粒子退化的根治方案：改用包自带的 `RoombaParticleFilter`（注入动作噪声 + 跳过终止粒子）；
      要六个 Roomba 脚本同步改并重跑 R1，本轮不做
- [ ] Roomba-Lidar 的 POMCPOW `alpha_observation` 做 1/15 vs 0.5 对照
- [x] ~~`test_RoombaL_POMCPOW.jl` 缺 `RoombaPOMDPs` 依赖~~ 2026-09-19 解决：改用 git URL 安装；
      原 `FORollout(RSExitSolver())` 是从 RS 文件复制来的 RockSample 专用启发式（对 Roomba 的
      UnderlyingMDP 没有 solve 方法），已换成随机 rollout（种子 3）
