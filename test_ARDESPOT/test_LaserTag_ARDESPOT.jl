using ARDESPOT
using POMDPModels, POMDPTools, POMDPs, Distributions, StatsBase
using LaserTag
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

# 程序结构与 test_POMCPOW/test_RS15_POMCPOW.jl 一致, solver 换成 AR-DESPOT。
# LaserTag 用默认的 DESPOTEmu 观测模型 -> 观测是离散的 DMeas, 所以完全适用 AR-DESPOT。
# (LaserTag 本来就是 DESPOT 原论文的 benchmark 之一。)
#
# 注意: LaserTag 不在 Env_ARDESPOT/Project.toml 的 [deps] 里, 运行前需要先加一次:
#   ] add https://github.com/JuliaPOMDP/LaserTag.jl.git

Random.seed!(1)

#POMDP problem
# 与 test_POMCGS / test_POMCPOW / test_Ada 的 LaserTag 用同一个问题实例 (rng=7)。
rng_env = MersenneTwister(7)
pomdp = gen_lasertag(rng = rng_env, robot_position_known = false)

mx_depth = 40 #对应 DESPOTSolver 的 D (package default 90), 与 POMCPOW 的 max_depth 对齐
mx_steps = 100
nb_runs   = 100
n_particles = 5000

upper_bound = 10.0 #

# --- bounds: AR-DESPOT 必须提供上下界 ---
# 上下界变体 A [当前启用]: 随机 rollout 下界 + 常数上界。
# 与 test_Ada/test_LaserTag_Ada.jl 的启用项一一对应 (同样的随机策略, 同样的种子 3),
# 这样 AdaOPS 和 AR-DESPOT 拿到的先验信息量完全相同, 结果才可比。
# upper: LaserTag 单步最大回报是 tag 成功的 +10 (其余每步 -1), 所以 10.0 是合法但偏松的上界。
bounds = IndependentBounds(
    DefaultPolicyLB(RandomPolicy(pomdp, rng = MersenneTwister(3))),
    upper_bound,
    check_terminal = true,
    consistency_fix_thresh = 1e-5,
)

# ---- 上下界变体 B [已注释]: 自写的 LaserTag 启发式下界 ----
# 不要用 LaserTag.jl 自带的 MoveTowards(): 它的 action() 只在 opponent-robot 的差向量
# 恰好是单位向量时才有定义, 对手稍远一点 (例如 diff = (5,3)) 就会
# DIR_TO_ACTION[SVector(5,3)] -> KeyError; 而且它可能返回对角动作 6..9,
# 但 gen_lasertag 默认 diag_actions=false, 动作空间只有 1:5。
# 下面这个策略用 sign 把方向压到四个主方向 + tag, 保证落在 1:5 内。
# 启用时必须同时在 test_Ada / test_ARDESPOT / test_POMCPOW 三个 LaserTag 文件里
# 一起切到变体 B (POMCPOW 那边是把它传给 estimate_value = FORollout(...)), 否则不可比。
#
# lt_heuristic = FunctionPolicy(function (s)
#     d = s.opponent - s.robot
#     dx, dy = d[1], d[2]
#     if dx == 0 && dy == 0
#         return 5                      # LaserTag.TAG_ACTION
#     elseif abs(dx) >= abs(dy)
#         return dx > 0 ? 2 : 4         # east : west
#     else
#         return dy > 0 ? 1 : 3         # north : south
#     end
# end)
#
# bounds = IndependentBounds(
#     DefaultPolicyLB(lt_heuristic),
#     upper_bound,
#     check_terminal = true,
#     consistency_fix_thresh = 1e-5,
# )


#POMDP solver: just for hyperparameter tuning, not for actual planning
solver = DESPOTSolver(
    bounds = bounds,
    K       = 500,        # ARDESPOT.jl default: 场景数, 主要调这个, 常用 [100, 500]
    D       = mx_depth,
    lambda  = 0.01,       # ARDESPOT.jl default
    xi      = 0.95,       # ARDESPOT.jl default
    epsilon_0 = 0.0,      # ARDESPOT.jl default
    T_max   = 3.0,        # 与 POMCPOW / AdaOPS 版本一致
    max_trials = typemax(Int),
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
CSV.write("data_ARDESPOT/LaserTag_K(500)_ardespot_$(timestamp)_$(mean(results))_R1.csv", df)

# 均值和标准差单独存一个 CSV
summary_df = DataFrame(mean_return = mean(results), std_return = std(results))
CSV.write("data_ARDESPOT/LaserTag_K(500)_ardespot_$(timestamp)_summary_R1.csv", summary_df)

println("\nTotal return: $(sum(results))")
println("Average return: $(mean(results)), Std: $(std(results))")




#changes 1st round:
#K=500, D=40, lambda=0.01, xi=0.95 (ARDESPOT.jl defaults), T_max=3.0s
#lower=DefaultPolicyLB(RandomPolicy(seed 3)), upper=10.0 (变体 A, 与 test_Ada 一致)
#100 runs, 5000 tracking particles, max_steps=100
