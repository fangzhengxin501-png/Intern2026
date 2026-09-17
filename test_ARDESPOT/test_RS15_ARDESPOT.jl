using ARDESPOT
using POMDPModels, POMDPTools, POMDPs, Distributions, StatsBase
using RockSample
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

# 程序结构与 test_POMCPOW/test_RS15_POMCPOW.jl 完全一致
# (BootstrapFilter + HistoryRecorder + discounted_reward + 明细/summary 两个 CSV),
# 只把 solver 换成 AR-DESPOT, 方便和 POMCPOW / AdaOPS / POMCGS 横向对比。
# RockSample 是离散状态 + 离散动作 + 离散观测, 是 AR-DESPOT 最标准的适用场景。

Random.seed!(1)

#POMDP problem
pomdp = RockSamplePOMDP(15, 15)

mx_depth = 40 #对应 DESPOTSolver 的 D (ARDESPOT.jl 默认是 90), 这里与 POMCPOW 的 max_depth 对齐
mx_steps = 100
nb_runs   = 100
n_particles = 5000

# --- bounds: AR-DESPOT 必须提供上下界 ---
# 用 RockSample.jl 自带的两个启发式, 和 AdaOPS 版本用的是同一对上下界,
# 只是换成 ARDESPOT 自己的包装类型 (DefaultPolicyLB / FullyObservableValueUB)。
bounds = IndependentBounds(
    DefaultPolicyLB(RSExitSolver()),        # lower bound: 直接走向出口的默认策略 rollout
    FullyObservableValueUB(RSMDPSolver()),  # upper bound: 完全可观测 MDP 的最优值
    check_terminal = true,
    consistency_fix_thresh = 1e-5,
)

# ---- 上下界变体 B [已注释]: 无信息上下界, 与 test_Ada/test_RS15_Ada_v2.jl 的变体 B 一一对应 ----
# 切换时两个文件必须同步切, 否则两边先验信息量不同, 结果不可比。
# bounds = IndependentBounds(
#     DefaultPolicyLB(RandomPolicy(pomdp, rng = MersenneTwister(3))),  # 随机 rollout 下界
#     10.0 / (1.0 - discount(pomdp)),                                  # 常数上界 Vmax = Rmax/(1-gamma)
#     check_terminal = true,
#     consistency_fix_thresh = 1e-5,
# )

#POMDP solver: just for hyperparameter tuning, not for actual planning
# 超参数取自 JuliaPOMDP/ARDESPOT.jl 的 DESPOTSolver 默认值 (K=500, lambda=0.01, xi=0.95),
# 只把 T_max 改成 3.0s 以匹配本项目其它 solver 的每步规划预算。
solver = DESPOTSolver(
    bounds = bounds,
    K       = 500,        # ARDESPOT.jl default: 场景 (scenario) 数, 主要调这个, 常用 [100, 500]
    D       = mx_depth,   # 搜索树最大深度 (package default 90)
    lambda  = 0.01,       # ARDESPOT.jl default: 正则化系数, 惩罚策略树规模
    xi      = 0.95,       # ARDESPOT.jl default: gap 收缩因子
    epsilon_0 = 0.0,      # ARDESPOT.jl default
    T_max   = 3.0,        # 每步在线规划时间预算, 与 POMCPOW 的 max_time = 3.0 一致
    max_trials = typemax(Int),   # 让 T_max 成为实际的停止条件
    bounds_warnings = true,
    tree_in_info = false,
    rng = MersenneTwister(1),
)

#policy,
planner = solve(solver, pomdp)

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
CSV.write("data_ARDESPOT/RS15_K(500)_ardespot_$(timestamp)_$(mean(results))_R1.csv", df)

# 均值和标准差单独存一个 CSV
summary_df = DataFrame(mean_return = mean(results), std_return = std(results))
CSV.write("data_ARDESPOT/RS15_K(500)_ardespot_$(timestamp)_summary_R1.csv", summary_df)

println("\nTotal return: $(sum(results))")
println("Average return: $(mean(results)), Std: $(std(results))")




#changes 1st round:
#K=500, D=40, lambda=0.01, xi=0.95 (ARDESPOT.jl defaults), T_max=3.0s
#lower=DefaultPolicyLB(RSExitSolver()), upper=FullyObservableValueUB(RSMDPSolver())
#100 runs, 5000 tracking particles, max_steps=100
