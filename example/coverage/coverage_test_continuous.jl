using PyPlot

using SubmodularMaximization
using Distributions
using LinearAlgebra
using Random


# pygui(false)
# fig_path = "./fig/coverage_test"
# mkpath(fig_path)

# num_agents = 5
# num_sensors = 5
# nominal_area = 2.0

# sensor_radius = sqrt(nominal_area / (num_agents * pi))
# station_radius = sensor_radius

# agent_specification = CircleAgentSpecification(sensor_radius, station_radius,
#                                          num_sensors)

# agents = generate_agents(agent_specification, num_agents)

# f(x) = mean_area_coverage(x, 100)
# problem = ExplicitPartitionProblem(f, agents)

###### Continuous Greedy Algorithm Test #######



# Define the number of players and items
function test_continuous()
    n_players = 5
    n_items = 5

    # Initialize y matrix
    y = zeros(n_players, n_items)

    # Initialize t and delta
    t = 0.0
    delta = 1.0 / (n_players * n_items)^2

    # Define the players' values for each item (random for demonstration purposes)
    w = rand(n_players, n_items)

    # Define a function to estimate the expected marginal profit of player i from item j
    function expected_marginal_profit(i, j, y, w)
        # Generate a random set Ri containing each item j independently with probability yij(t)
        R = [j for j = 1:n_items if rand() < y[i, j]]
        # Compute the expected marginal profit of player i from item j
        wi_Ri_j = sum(w[i, R]) + w[i, j]
        wi_Ri = sum(w[i, R])
        return wi_Ri_j - wi_Ri
    end

    # Initialize the distribution of all mn actions
    action_distribution = zeros(n_players, n_items)

    # Run the algorithm
    while t < 1
        # Estimate the expected marginal profits for all players and items
        
        ω = zeros(n_players, n_items)
        for i = 1:n_players
            for j = 1:n_items
                for k = 1:(n_players * n_items)^3
                    ω[i, j] += expected_marginal_profit(i, j, y, w)
                    # println("Loop4\n");
                end
                ω[i, j] /= (n_players * n_items)^3
                # println("Loop3\n");
            end
            # println("Loop2\n");
        end
        # Update y matrix
        for j = 1:n_items
            i_star = argmax(ω[:, j])
            for i = 1:n_players
                if i == i_star
                    y[i, j] += delta
                else
                    # y[i, j] -= delta / (n_players - 1)
                    y[i,j] = y[i,j]
                end
            end
        end
        # Increment t
        t += delta
        # println("Loop1: $t");
    end

    # println("OutLoop1\n");

    # Compute the distribution of all mn actions
    # for i = 1:n_players
    #     for j = 1:n_items
    #         y[i, j] = y[i, j] * prod(1 - y[k, j] for k = 1:n_players if k != i)
    #     end
    # end

    # Print the action distribution matrix
    println("W:")
    println(w)
    println("Action distribution matrix:")
    println(y)
end

test_continuous()



# function solve_continuous(p::PartitionProblem)
#     # m = p.partition_matroid.size()
#     # get the number of robots
#     (size_robot,) = size(p.partition_matroid)
#     # get the number of actions 
#     (size_action,) = size(p.partition_matroid[1].sensors)
#     action_set = p.partition_matroid[1].sensors;
#     for i in size_robot
#         if i != 1
#             union(action_set, p.partition_matroid[i].sensors)
#         end
#     end

#     p = continuous_greedy(p.objective, action_set, size_robot, (size_robot*size_action)^5)
#     println("The size of x is $size_action")
#   end


# solve_continuous(problem)

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



