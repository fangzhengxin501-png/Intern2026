using AdaOPS
using POMDPModels, POMDPTools, POMDPs, ParticleFilters, StaticArrays
using Distributions, StatsBase
using CSV, DataFrames
using Random
using Dates

Random.seed!(1)

pomdp = LightDark1D()

# AdaOPS 的自适应粒子滤波 (KLD-sampling) 需要把状态映射到一个向量,
# 才能落到 StateGrid 的网格里计数。LightDark1D 的状态只有一个连续维度 y。
Base.convert(::Type{SVector{1, Float64}}, s::LightDark1DState) = SVector(convert(Float64, s.y))

# --- 上下界 (bounds), AdaOPS 必须提供, POMCGS 不需要 ---
# 下界: 一个简单启发式 rollout 策略, 让智能体始终往光源 (y=0) 方向移动。
# 上界: 用一个保守的常数上界。这两者都强烈影响 AdaOPS 的效果, 建议你根据实际
# 收敛效果替换为更紧的上下界 (例如用 FOValue/FORollout 结合已知的 MDP 求解器)。
lb_policy = FunctionPolicy(s -> s.y < 0 ? 1 : -1)
bound = AdaOPS.IndependentBounds(
    FORollout(lb_policy),
    0.0,
    check_terminal=true,
    consistency_fix_thresh=1e-5,
)

planning_time = 3.0   # 论文原文: "All online solvers are limited to 3 seconds of computation time per step."
max_depth = 100

solver = AdaOPSSolver(
    bounds=bound,
    delta=0.1,                                   # 对应 POMCGS 的 max_b_gap (0.1 given in table 2)
    D=30,                                         # 对应 POMCGS 的 max_search_depth
    grid=StateGrid(collect(-20.0:1.0:20.0)),      # 对应 POMCGS 的 state_grid = [1.0, 1.0]
    m_min=100,                                    # 自适应粒子数下限
    m_max=5000,                                   # 对应 POMCGS 的 nb_particles (5000 given in table 2)
    T_max=planning_time,                          # 每步在线规划时间预算, 与论文评测设置一致
    tree_in_info=false,
    rng=MersenneTwister(1),
)

planner = solve(solver, pomdp)

nb_runs = 1
n_particles = 5000   # 用于在仿真中跟踪真实信念的粒子滤波器 (与 AdaOPS 内部搜索粒子数是两回事)

results = Float64[]

for i in 1:nb_runs
    b0 = initialstate(pomdp)
    s0 = rand(b0)
    up = BootstrapFilter(pomdp, n_particles)

    # 用 RolloutSimulator 返回折扣回报 (discounted return), 与论文 Table 1
    # "average discounted returns" 的口径一致, 可直接与 AdaOPS on Light Dark
    # 的 3.76 ± 0.10 做对比。
    run_return = simulate(RolloutSimulator(max_steps=max_depth), pomdp, planner, up, b0, s0)
    push!(results, run_return)
    println("Run $i: $run_return")
end

# Save results to CSV
df = DataFrame(run_id = 1:nb_runs, return_value = results)
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
CSV.write("on_off_compare_data/LD_AdaOPS_$(timestamp).csv", df)

println("\nTotal return: $(sum(results))")
println("Average return: $(mean(results))")
println("Results saved to LD_AdaOPS_$(timestamp).csv")