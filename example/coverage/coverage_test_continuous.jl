using PyPlot

using SubmodularMaximization
using Distributions
using LinearAlgebra


pygui(false)
fig_path = "./fig/coverage_test"
mkpath(fig_path)

num_agents = 5
num_sensors = 5
nominal_area = 2.0

sensor_radius = sqrt(nominal_area / (num_agents * pi))
station_radius = sensor_radius

agent_specification = CircleAgentSpecification(sensor_radius, station_radius,
                                         num_sensors)

agents = generate_agents(agent_specification, num_agents)

f(x) = mean_area_coverage(x, 100)
problem = ExplicitPartitionProblem(f, agents)

###### Continuous Greedy Algorithm Test #######

# Define the gradient of the multilinear extension of the submodular function f
function gradient_multilinear_extension(f, n, x, m=5n^5)
    g = zeros(n)
    for i in 1:m
        S = rand(0:1, n)
        T = rand(0:1, n)
        S_val = f(findall(S .== 1))
        T_val = f(findall(T .== 1))
        delta_val = prod(1 .- (1 .- x) .* (S .- T))
        g = g .+ delta_val * (S .- T) * (S_val - T_val)
    end
    return g
end
# Define the continuous greedy algorithm for the submodular welfare problem using the multilinear extension with n^5 samples
function continuous_greedy_multilinear_extension_n5(f, n, v, ϵ)
    x = zeros(n)
    p = zeros(n)
    while true
        g = gradient_multilinear_extension(f, n, x)
        v_norm = norm(v)
        if v_norm == 0
            break
        end
        g_norm = norm(g)
        if g_norm == 0
            break
        end
        g = g ./ g_norm
        s = argmax(dot(v, g))
        if dot(v, g) / v_norm <= (1 - ϵ)
            break
        end
        x[s] = 1.0
        p[s] += 1.0
    end
    p = p / sum(p)
    return p
end

# example usage
# m = 3
# n = 4
# V = collect(1:m*n)
# f(S) = length(S)
# p = continuous_greedy(f, V, m, (m*n)^5)
# println(p)




function solve_continuous(p::PartitionProblem)
    # m = p.partition_matroid.size()
    # get the number of robots
    (size_robot,) = size(p.partition_matroid)
    # get the number of actions 
    (size_action,) = size(p.partition_matroid[1].sensors)
    action_set = p.partition_matroid[1].sensors;
    for i in size_robot
        if i != 1
            union(action_set, p.partition_matroid[i].sensors)
        end
    end

    p = continuous_greedy(p.objective, action_set, size_robot, (size_robot*size_action)^5)
    println("The size of x is $size_action")
  end


solve_continuous(problem)

# function evaluate_solver(solver, name)
#   println("$name solver running")
#   @time solution = solver(problem)

#   figure()
#   xlim([0, 1])
#   ylim([0, 1])
#   colors = generate_colors(agents)
#   visualize_agents(agents, colors)
#   visualize_solution(problem, solution, colors)

#   gca().set_aspect("equal")
#   savefig("$(fig_path)/$(to_file(name)).png", pad_inches=0.00, bbox_inches="tight")

#   coverage = solution.value

#   title("$name Solver ($coverage)")

#   @show coverage
# end



#@show mean_weight(problem)
#@show total_weight(problem)



