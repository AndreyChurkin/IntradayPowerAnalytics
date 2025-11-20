"""
Formulates and solves the backtesting optimisation problem for BESS trading in a single trading session (corresponding to one delivery hour).

Inputs: 
- DataFrame containing all bid and ask orders for a single trading session.
- BESS technical and economic parameters.

Outputs: 
- The objective value (maximum achievable trading profit).
- DataFrame with the optimal BESS trading actions.


Andrey Churkin
https://andreychurkin.ru/

"""



function optimise_BESS_in_1_session(session_df::DataFrame;
    E_max::Float64,
    P_ch_max::Float64,  P_disch_max::Float64,
    eta_ch::Float64,    eta_disch::Float64,
    SoC_init::Float64,  SoC_final::Float64,
    E_cost_0,
    fee
    )

    Δt = 1.0 # consider only 1-hour physical delivery products

    T = nrow(session_df)

    bid_prices = session_df.bid_price
    ask_prices = session_df.ask_price
    bid_volumes = session_df.bid_volume
    ask_volumes = session_df.ask_volume

    # # Define the model and the variables:
    Model_1_session = Model(Gurobi.Optimizer)
    # set_silent(Model_1_session)
    @variable(Model_1_session, 0 <= ch[1:T] <= P_ch_max)
    @variable(Model_1_session, 0 <= disch[1:T] <= P_disch_max)
    @variable(Model_1_session, 0 <= SoC[1:T] <= E_max)
    @variable(Model_1_session, x_ch[1:T], Bin)
    @variable(Model_1_session, x_disch[1:T], Bin)

    # # Linking charging actions with volumes:
    @constraint(Model_1_session, [t=1:T], ch[t] <= P_ch_max * x_ch[t])
    @constraint(Model_1_session, [t=1:T], disch[t] <= P_disch_max * x_disch[t])

    # # Limiting actions by the ask/bid volumes available in the orders:
    @constraint(Model_1_session, [t=1:T], ch[t] <= ask_volumes[t])
    @constraint(Model_1_session, [t=1:T], disch[t] <= bid_volumes[t])

    # # Prevent simultaneous charge & discharge:
    @constraint(Model_1_session, [t=1:T], x_ch[t] + x_disch[t] <= 1)

    # # SoC dynamics:
    @constraint(Model_1_session, SoC[1] == SoC_init + (eta_ch*ch[1] - disch[1]/eta_disch) * Δt)
    @constraint(Model_1_session, [t=2:T], SoC[t] == SoC[t-1] + (eta_ch*ch[t] - disch[t]/eta_disch) * Δt)
    @constraint(Model_1_session, SoC[T] == SoC_final)

    # # Objective:
    @objective(Model_1_session, Max, SoC_init*E_cost_0 
                + sum(((bid_prices[t] - fee)*disch[t] - (ask_prices[t] + fee)*ch[t]) * Δt for t in 1:T)
    )

    optimize!(Model_1_session)


    if termination_status(Model_1_session) == MOI.OPTIMAL
        printstyled("\n✅ Optimal solution found, objective value = ", objective_value(Model_1_session), color = :green)
    else
        printstyled("\n❌ WARNING: Solver did not return an optimal solution. Status = ", termination_status(Model_1_session), color = :red)
    end

    BESS_optimisation_results = DataFrame(
        ts = session_df.ts,
        hours_to_delivery = session_df.hours_to_delivery,
        ask_price = session_df.ask_price,
        bid_price = session_df.bid_price,
        action_ch = Int.(value.(ch) .> 1e-6),
        action_disch = Int.(value.(disch) .> 1e-6),
        volume_ch = value.(ch),
        volume_disch = value.(disch),
        SoC = value.(SoC),
        cum_revenue = cumsum((value.(disch) .* session_df.bid_price) - (value.(ch) .* session_df.ask_price))
    )

    return objective_value(Model_1_session), BESS_optimisation_results

end