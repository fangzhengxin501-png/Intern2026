using ARDESPOT
using POMDPModels, POMDPTools, POMDPs, Distributions, StatsBase
using CSV, DataFrames
using Random
using Dates
using ParticleFilters
using Statistics

# ============================================================================
# 实验环境一致性基准 (baseline = test_POMCPOW/test_RS15_POMCPOW.jl)
#   mx_depth    = 40    -> POMCPOW: max_depth / AdaOPS: max_depth / ARDESPOT: D
#   mx_steps    = 100   -> HistoryRecorder(max_steps = ...)
#   nb_runs     = 100   -> 正式实验值 (当前文件若为 1/2 是 smoke test, 出数据前必须改回 100)
#   n_particles = 5000  -> 仿真中跟踪真实 belief 的 BootstrapFilter 粒子数
#   规划预算     = 3.0s  -> POMCPOW: max_time / AdaOPS: T_max / ARDESPOT: T_max
#   种子         = planner MersenneTwister(1), filter 1000+i, recorder 2000+i
#   LaserTag 环境实例 = gen_lasertag(rng = MersenneTwister(7), robot_position_known=false)
# 以上七项在 test_POMCPOW / test_Ada / test_ARDESPOT 三个目录里必须完全一致。
# ============================================================================

# ============================================================================
# LightDark1D 的观测是连续的, 原始形式与 AR-DESPOT 不兼容; 本文件用**观测分箱**修复。
#
# 【为什么原始形式不行】
# DESPOT 用 K 条 scenario 做确定化采样, 然后按观测把 scenario 分组成观测分支。
# ARDESPOT.jl 的 src/tree.jl 里这一步是:
#       odict = Dict{O, Int}()
#       sp, o, r = @gen(:sp, :o, :r)(p.pomdp, s, a, rng)
#       bp = get(odict, o, 0)
# 分组靠的是观测的**精确相等 / 哈希**。LightDark1D 的观测是连续 Float64 (y 加高斯噪声),
# K 条 scenario 会产生 K 个互不相同的观测 -> 每个观测分支里只剩 1 条 scenario,
# 搜索树退化成 K 条互相独立的单粒子轨迹, belief 完全没被共享, 下界极度乐观、方差极大。
#
# 【修复方式: 只在 planner 的模型里分箱 (做法 A)】
# 下面的 BinnedObsPOMDP 把观测离散成 bin 索引 (Int), **只交给 DESPOTSolver**;
# HistoryRecorder 的仿真环境和 BootstrapFilter 的 belief 更新仍然用原始的连续
# LightDark1D()。于是:
#   - 被评价的**问题**没变 (仍是连续观测的 LightDark), 三个算法的回报可以直接比;
#   - 离散化成了**算法的一部分** —— 报的是 "AR-DESPOT + 观测分箱" 这个变体,
#     这正是 DESPOT 原论文处理连续观测的做法;
#   - 代价是 planner 的模型与真实环境有偏差 (planner 以为观测是离散的),
#     属于"用近似模型规划", 必须在报告里说明, 但不破坏可比性。
# 反面做法 B (不采用): 把环境本身离散化。那 LightDark 就是另一个问题了,
# POMCPOW 和 AdaOPS 必须在离散版上重跑, 否则三列不同源。
#
# 【bin 宽度是一个新的超参数】
# LightDark1D 的观测噪声随 y 变化 (sigma = |y - light_loc|/sqrt(2) + eps, 离光源越远越糊),
# 所以单一的全局 bin 宽度必然是偏的: 太粗会抹掉"走到亮处能看得更准"这个信息增益,
# 太细则 K 条 scenario 仍各自落进不同的箱, 退化问题没解决。
# OBS_BIN = 1.0 的依据: LightDark1D 判定成功的容差是 |y| < 1, 所以 ~1 的分辨率
# 与决策所需精度同量级。把它当成 DESPOT 在本问题上的调参项, 扫 [0.5, 1.0, 2.0],
# 投入与 POMCPOW 扫 alpha_observation / AdaOPS 扫 delta 对等。
# 变体: 按噪声尺度做**相对分箱** (见下方注释), 让每个箱覆盖大致等量的概率质量。
# ============================================================================

Random.seed!(1)

#POMDP problem
# 真实环境 / belief filter 用的模型: 原始连续观测, 与 POMCPOW / AdaOPS 版本完全相同。
pomdp = LightDark1D()

mx_depth = 40 #对应 DESPOTSolver 的 D (package default 90)
mx_steps = 100
nb_runs   = 100
n_particles = 5000

# --- 观测分箱 wrapper: 只给 planner 用 ---
# 观测类型从 Float64 变成 Int (bin 索引), 于是 ARDESPOT 的 odict 能把多条 scenario
# 归到同一个观测分支, belief 得到共享。状态 / 动作 / 转移 / 回报 / 终止判定全部原样转发。
OBS_BIN = 1.0   # 分箱宽度; 设为 nothing 则退回原始连续观测 (退化基线, 用于对照)

struct BinnedObsPOMDP{P<:POMDP} <: POMDP{LightDark1DState, Int, Int}
    m::P
    bin::Float64
end

bin_obs(w::BinnedObsPOMDP, o::Real) = round(Int, o / w.bin)

# ---- 变体: 相对分箱 [已注释] ----
# 固定宽度在光源附近 (sigma ~ 1e-2) 太粗、在暗处 (sigma ~ 7) 太细。
# 下面按该观测处的噪声尺度缩放, 让每个箱覆盖大致等量的概率质量。
# rel 是新的超参数 (箱宽 = rel 个标准差), 建议扫 [0.25, 0.5, 1.0]。
# 注意: 依赖 LightDark1D 把噪声函数存在 .sigma 字段里。
#
# bin_obs(w::BinnedObsPOMDP, o::Real) = round(Int, o / max(w.bin * w.m.sigma(o), 1e-3))

POMDPs.discount(w::BinnedObsPOMDP)          = discount(w.m)
POMDPs.actions(w::BinnedObsPOMDP)           = actions(w.m)
POMDPs.actions(w::BinnedObsPOMDP, b)        = actions(w.m)
POMDPs.isterminal(w::BinnedObsPOMDP, s)     = isterminal(w.m, s)
POMDPs.initialstate(w::BinnedObsPOMDP)      = initialstate(w.m)
POMDPs.transition(w::BinnedObsPOMDP, s, a)  = transition(w.m, s, a)
POMDPs.reward(w::BinnedObsPOMDP, s, a)      = reward(w.m, s, a)

function POMDPs.gen(w::BinnedObsPOMDP, s, a, rng)
    sp, o, r = @gen(:sp, :o, :r)(w.m, s, a, rng)
    return (sp = sp, o = bin_obs(w, o), r = r)
end

# planner 看到的模型 (bounds 和 solve 都用这个); 仿真和 belief 更新仍用 pomdp
planner_pomdp = OBS_BIN === nothing ? pomdp : BinnedObsPOMDP(pomdp, OBS_BIN)
bin_tag = OBS_BIN === nothing ? "rawobs" : "bin($(OBS_BIN))"

# --- bounds ---
# lower: 简单启发式, 始终朝光源 (y = 0) 方向走, 与 AdaOPS 版本一致。
# (DefaultPolicyLB 的 rollout 在 planner 的模型即 planner_pomdp 上进行, 由 solve 传入;
#  策略只看状态, 不看观测, 所以分箱不影响下界的语义。)
# upper: LightDark1D 的单步最大回报 correct_r = 10.0 (POMDPModels 默认值), 作为常数上界。
lb_policy = FunctionPolicy(s -> s.y < 0 ? 1 : -1)
bounds = IndependentBounds(
    DefaultPolicyLB(lb_policy),
    10.0,
    check_terminal = true,
    consistency_fix_thresh = 1e-5,
)

# ---- 上下界变体 B [已注释]: 完全无信息的常数上下界, 与 test_Ada/test_LightDark_Ada_v2.jl 变体 B 对应 ----
# bounds = IndependentBounds(
#     -10.0,   # 常数下界 = incorrect_r = -10.0
#      10.0,   # 常数上界 = correct_r = 10.0
#     check_terminal = true,
#     consistency_fix_thresh = 1e-5,
# )

#POMDP solver: just for hyperparameter tuning, not for actual planning
solver = DESPOTSolver(
    bounds = bounds,
    K       = 500,        # ARDESPOT.jl default
    D       = mx_depth,   # 2026-09-17 修正: DESPOTSolver 的字段名是 D, 不是 max_depth
                          # (原来写成 max_depth 会被 @with_kw 判为未知关键字而报错)
    lambda  = 0.01,       # ARDESPOT.jl default
    xi      = 0.95,       # ARDESPOT.jl default
    epsilon_0 = 0.0,      # ARDESPOT.jl default
    T_max   = 3.0,        # 与 POMCPOW / AdaOPS 版本一致
    max_trials = typemax(Int),
    bounds_warnings = true,    # 分箱后观测分支不再退化, 恢复与其它三个 ARDESPOT 脚本一致
    tree_in_info = false,
    rng = MersenneTwister(1),
)

#policy,
# 注意: planner 用分箱后的模型, 下面 simulate 的环境和 BootstrapFilter 用原始 pomdp
planner = solve(solver, planner_pomdp)

results = Float64[]

for i in 1:nb_runs
    u_p = BootstrapFilter(pomdp, n_particles, MersenneTwister(1000 + i)) #belief update
    hr = HistoryRecorder(max_steps = mx_steps, rng = MersenneTwister(2000 + i)) #simulates policy execution in "real" environment, record  trajectory
    history = simulate(hr, pomdp, planner, u_p) #

    run_return = discounted_reward(history)
    push!(results, run_return)
    println("Run $i: $run_return")
end

# Save results to CSV
df = DataFrame(run_id = 1:nb_runs, return_value = results)
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
CSV.write("data_ARDESPOT/LD_$(bin_tag)_K(500)_ardespot_$(timestamp)_$(mean(results))_R1.csv", df)

# 均值和标准差单独存一个 CSV
summary_df = DataFrame(mean_return = mean(results), std_return = std(results))
CSV.write("data_ARDESPOT/LD_$(bin_tag)_K(500)_ardespot_$(timestamp)_summary_R1.csv", summary_df)

println("\nTotal return: $(sum(results))")
println("Average return: $(mean(results)), Std: $(std(results))")
if OBS_BIN === nothing
    println("\n[WARNING] OBS_BIN = nothing: 连续观测, DESPOT 观测分支退化, 结果仅作不兼容性对照。")
else
    println("\n[NOTE] planner 模型的观测已分箱 (bin = $(OBS_BIN)); 仿真环境与 belief filter 仍为连续观测。")
    println("       报告时须写明这是 \"AR-DESPOT + 观测分箱\" 变体, 且 bin 宽度是待调超参数。")
end




#changes 1st round:
#K=500, D=40, lambda=0.01, xi=0.95 (ARDESPOT.jl defaults), T_max=3.0s
#lower=DefaultPolicyLB(toward-light heuristic), upper=10.0 (= correct_r)
#100 runs, 5000 tracking particles, max_steps=100
#changes 2nd round (2026-09-17):
#观测分箱 wrapper (OBS_BIN=1.0, 只作用于 planner 模型) 修复观测分支退化
#修正 DESPOTSolver 关键字 max_depth -> D; bounds_warnings 恢复为 true
