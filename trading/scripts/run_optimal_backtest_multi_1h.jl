"""
Analyses multiple 1-hour intraday trading sessions in a loop, one by one.
For each session, computes the optimal BESS trading actions and the maximum achievable profit.


Andrey Churkin
https://andreychurkin.ru/

"""



cd(dirname(@__FILE__))
println(pwd())

using CSV
using DataFrames
using Dates
using JuMP, Gurobi
using Suppressor
using Plots, Plots.PlotMeasures
using Statistics



# # Select the intraday market data set to analyse:
ID_market_data_file_path = "C://Users//achurkin//Documents//MEGA//Imperial College London//Pierre Pinson//models//IDC_EPEX_DK1_BestBidAsk_clean_v1.csv"
ID_market_data_full = CSV.read(ID_market_data_file_path, DataFrame)
# ID_market_data_full = CSV.read(ID_market_data_file_path, DataFrame, limit = 2*10^6) # <--- read a limited data set to save time


# # Define the vector of trading sessions to optimise (delivery hours):
all_unique_delivery_hours = sort(unique(ID_market_data_full.delivery_start))
# delivery_hours_to_optimise = all_unique_delivery_hours[1:end]
delivery_hours_to_optimise = all_unique_delivery_hours[2019:end]



# # Read the day-ahead market data:
DA_market_data_file_path = "C://Users//achurkin//Documents//MEGA//Imperial College London//Pierre Pinson//ViPES2X//Roar Nicolaisen//DK1 Day-ahead Prices_202301010000-202401010000.csv"
DA_market_data = CSV.read(DA_market_data_file_path, DataFrame)
DA_delivery_start_times = split.(DA_market_data[!,"MTU (CET/CEST)"], " - ")

DA_missing_rows = findall(ismissing, DA_market_data[!, "Day-ahead Price [EUR/MWh]"])
if length(DA_missing_rows) != 0
    println()
    printstyled("\nWARNING: Day-ahead price data is missing for ",length(DA_missing_rows)," delivery hours:", color = :red)
    for missing_row in DA_missing_rows
        println()
        printstyled(DA_market_data[missing_row,:]["MTU (CET/CEST)"], color = :red)

        DA_market_data[missing_row,"Day-ahead Price [EUR/MWh]"] = 
            0.5*(DA_market_data[missing_row-1,"Day-ahead Price [EUR/MWh]"] + DA_market_data[missing_row+1,"Day-ahead Price [EUR/MWh]"])
    
    end
    printstyled("\nMissing DA prices are replaced with the average prices of neighbouring hours", color = :red)
end



# # Define battery energy storage system (BESS) parameters, in MW and MWh:
BESS_ch_power_max = 1.0 # MW
BESS_dischch_power_max = 1.0 # MW
""" 
Note that charging more energy than BESS_ch_power_max is physically impossible within one delivery hour. 
Such trading actions are possible, but would imply speculative trading.
"""
BESS_energy_max = 2.0 # MWh
BESS_energy_0 = 0.0 # initial energy (state of charge), MWh
BESS_energy_cost_0 = 20 # charging cost of the initial energy capacity, EUR/MW
BESS_eta_ch = 0.90 # battery charging efficiency factor
BESS_eta_disch = 0.90 # battery discharging efficiency factor
Trading_and_Clearing_fee = 0.124 # EUR/MWh (check Nord Pool or EPEX fee schedule)



""" Analysing trading sessions in a loop for each delivery hour """

bess_trading_summary_per_1h_session = DataFrame(
    delivery_start = String[],
    profit = Float64[],
    number_of_actions = Int[]
)

for session = 1:length(delivery_hours_to_optimise)
    delivery_hour_to_optimise = delivery_hours_to_optimise[session]
    print("\nOptimising session #",session,", delivery_start = ",delivery_hour_to_optimise)

    # # Remove orders in the records that are for sessions beyond the delivery hour:
    ID_market_data_session = filter(:delivery_start => ==(delivery_hour_to_optimise), ID_market_data_full)


    # # Remove duplicating rows (same prices/orders for the same millisecond timestamps):
    ID_market_data_session = unique(ID_market_data_session, [:ts, :delivery_start, :ask_price, :ask_volume, :bid_price, :bid_volume])


    # # Sort by timestamp to ensure correct chronological trades:
    ID_market_data_session = sort(ID_market_data_session, :ts)


    # # Convert timestamps of orders into hours till delivery time (for further plotting):
    hours_to_delivery = (Dates.value.(
                        DateTime.(ID_market_data_session.ts, "yyyy-mm-dd HH:MM:SS.sss") 
                        .- 
                        DateTime.(delivery_hour_to_optimise, "yyyy-mm-dd HH:MM:SS.sss")
                        ) / (60 * 60 * 1000)
    )

    ID_market_data_session.hours_to_delivery = hours_to_delivery

    # # Remove repeating orders from successive timestamps by nullifying their volumes (needed to make BESS backtesting more realistic)
    ID_market_data_session_copy_all_volumes = copy(ID_market_data_session)
    for i in 2:nrow(ID_market_data_session)
        if ID_market_data_session.ask_price[i] == ID_market_data_session_copy_all_volumes.ask_price[i-1] && ID_market_data_session.ask_volume[i] == ID_market_data_session_copy_all_volumes.ask_volume[i-1]
            ID_market_data_session.ask_volume[i] = 0.0
        end
        if ID_market_data_session.bid_price[i] == ID_market_data_session_copy_all_volumes.bid_price[i-1] && ID_market_data_session.bid_volume[i] == ID_market_data_session_copy_all_volumes.bid_volume[i-1]
            ID_market_data_session.bid_volume[i] = 0.0
        end
    end

    println()
    println("Total number of best bid/ask price changes = ",nrow(ID_market_data_session))
    println("The earliest order placed: ",round(minimum(hours_to_delivery), digits=2)," h to delivery")
    println("The latest order placed: ",round(maximum(hours_to_delivery), digits=2)," h to delivery")

    # # Day-ahead market data:
    # Parse the original string into a DateTime object
    delivery_hour_to_optimise_DateTime = DateTime(delivery_hour_to_optimise, "yyyy-mm-dd HH:MM:SS")

    # Convert the DateTime object into the desired format, find this delivery time in the day-ahead data, find the day-ahead price:
    delivery_hour_to_optimise_time_format2 = Dates.format(delivery_hour_to_optimise_DateTime, "dd.mm.yyyy HH:MM")
    DA_delivery_time_position = findall(x -> x == delivery_hour_to_optimise_time_format2, map(x -> x[1], DA_delivery_start_times))[1]
    DA_delivery_time_price = DA_market_data[!,"Day-ahead Price [EUR/MWh]"][DA_delivery_time_position][1]




    """ Building and solving the BESS trading optimisation model """
    @suppress begin

        include("../src/bess_optimisation_1h.jl")

        global BESS_objective_value, BESS_optimisation_results = optimise_BESS_in_1_session(ID_market_data_session;
            E_max = BESS_energy_max, 
            P_ch_max = BESS_ch_power_max, 
            P_disch_max = BESS_dischch_power_max, 
            eta_ch = BESS_eta_ch, 
            eta_disch = BESS_eta_disch, 
            SoC_init = BESS_energy_0, 
            SoC_final = BESS_energy_0, 
            E_cost_0 = BESS_energy_cost_0,
            fee = Trading_and_Clearing_fee
        )

    end # @suppress

    global BESS_optimisation_results_only_actions = filter(row -> (row.volume_ch != 0 || row.volume_disch != 0),
                BESS_optimisation_results
    )

    push!(bess_trading_summary_per_1h_session, (
        delivery_start = delivery_hour_to_optimise,
        profit = BESS_objective_value,
        number_of_actions = nrow(BESS_optimisation_results_only_actions)
    ))
    println("The optimal BESS strategy includes ",nrow(BESS_optimisation_results_only_actions)," actions")
    println("Maximum achievable profit = ",BESS_objective_value," EUR")


end


CSV.write("..//results//bess_trading_summary_per_1h_session.csv",
          bess_trading_summary_per_1h_session
)




""" This is a preliminary visualisation of multiple 1-hour intraday trading sessions. It will be improved later """

using StatsPlots

histogram(
    bess_trading_summary_per_1h_session.profit,
    bins = 100,
    xlabel = "Profit, EUR",
    ylabel = "Frequency",
    title = "Distribution of BESS trading profit (in 1h sessions)",
    legend = false,
    titlefont = 8,
    fontfamily = "Courier"
)

histogram(
    bess_trading_summary_per_1h_session.number_of_actions,
    bins = 100,
    xlabel = "Number of actions",
    ylabel = "Frequency",
    title = "Total number of BESS trading actions (in 1h sessions)",
    legend = false,
    titlefont = 8,
    fontfamily = "Courier",
    color = palette(:tab10)[5]
)

scatter(
    bess_trading_summary_per_1h_session.number_of_actions,
    bess_trading_summary_per_1h_session.profit,
    xlabel = "Total number of optimal BESS actions",
    ylabel = "Profit per session, EUR",
    size = (1100,1000), 
    # xlim = (-20, 1000),
    # ylim = (-200, 10000),
    xtickfontsize=fz, ytickfontsize=fz,
    fontfamily = "Courier", 
    titlefontsize = fz-3,
    xguidefontsize = fz,
    yguidefontsize = fz,
    legendfont = fz-3,
    legend = :topleft,
    framestyle = :box,
    # margin = 10mm,
    left_margin = 20mm,
    right_margin = 10mm,
    top_margin = 10mm,
    bottom_margin = 10mm,
    minorgrid = :true
)


action_profit_correlation = cor(
    bess_trading_summary_per_1h_session.number_of_actions,
    bess_trading_summary_per_1h_session.profit
)

println("Action-profit correlation: ", round(action_profit_correlation, digits=3))