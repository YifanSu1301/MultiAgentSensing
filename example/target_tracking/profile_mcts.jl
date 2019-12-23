using SubmodularMaximization
using POMDPs
using MCTS

using Statistics
using ProfileView

default_steps = 20
grid_size = 10
horizon = 2
iterations = 100

function run_test(steps)
  println("Running test: ", steps, " steps")
  grid = Grid(grid_size, grid_size)
  sensor = RangingSensor(0.5^2, 0.1^2)

  robot_state = random_state(grid)
  target_state = random_state(grid)

  target_states = Array{State}(undef, steps)
  target_states[1] = target_state

  robot_states = Array{State}(undef, steps)
  robot_states[1] = robot_state

  histogram_filter = Filter(grid)

  solver = generate_solver(horizon, n_iterations = iterations)

  time = @elapsed for ii = 2:steps
    println("Step: ", ii)

    # Before the target moves and the robot receives a measurement, execute robot
    # dynamics
    mdp = SingleRobotTargetTrackingProblem(grid, sensor, horizon,
                                           [histogram_filter])
    policy = solve(solver, mdp)
    robot_state = action(policy, MDPState(robot_state))
    robot_states[ii] = robot_state

    # Then update the target and sample the observation
    target_state = target_dynamics(grid, target_states[ii-1])
    target_states[ii] = target_state

    range_observation = generate_observation(sensor, robot_state, target_state)

    # After updating states, update the filter (in place for now)
    process_update!(histogram_filter, transition_matrix(grid))
    measurement_update!(histogram_filter, robot_state, get_states(grid), sensor,
                        range_observation)
  end

  println("Test duration: ", time, " seconds")
  println("  ", time / (steps - 1), " seconds per step")
end

@profview run_test(2)
ProfileView.closeall()
@profview run_test(default_steps)

nothing