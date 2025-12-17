"""
Computes bid/ask price volatility for 24 parallel trading sessions (for the entire delivery day).
Iterates over multiple 24-hour sessions to generate day-level volatility metrics and visualise them.

Andrey Churkin
https://andreychurkin.ru/

"""



cd(dirname(@__FILE__))
println(pwd())

using CSV
using DataFrames
using Dates
using Suppressor
using Plots, Plots.PlotMeasures
using Statistics
using StatsPlots


# # Select the intraday market data set to analyse:
ID_market_data_file_path = "C://Users//achurkin//Documents//MEGA//Imperial College London//Pierre Pinson//models//IDC_EPEX_DK1_BestBidAsk_clean_v1.csv"

# ID_market_data = CSV.read(ID_market_data_file_path, DataFrame)
ID_market_data = CSV.read(ID_market_data_file_path, DataFrame, limit = 2*10^6) # <--- read a limited data set to save time during testing

ID_market_data.delivery_start = DateTime.(ID_market_data.delivery_start, dateformat"yyyy-mm-dd HH:MM:SS")
ID_market_data.day = Date.(ID_market_data.delivery_start)


grouped_by_day_ID_sessions = groupby(ID_market_data, :day)
println("Number of unique delivery days: ", length(grouped_by_day_ID_sessions))


volatility_by_day = combine(grouped_by_day_ID_sessions) do df_day_session
    # println("delivery hour: ",df_day_session.day[1])

    mid_price = (df_day_session.ask_price .+ df_day_session.bid_price) ./ 2  # mid-prices between ask and bid

    μ_ask_price = mean(df_day_session.ask_price)
    σ_ask_price = std(df_day_session.ask_price)
    r_ask_price = maximum(df_day_session.ask_price) - minimum(df_day_session.ask_price)

    μ_bid_price = mean(df_day_session.bid_price)
    σ_bid_price = std(df_day_session.bid_price)
    r_bid_price = maximum(df_day_session.bid_price) - minimum(df_day_session.bid_price)

    μ_mid_price = mean(mid_price)
    σ_mid_price = std(mid_price)
    r_mid_price = maximum(mid_price) - minimum(mid_price)

    # NamedTuple becomes columns in the result
    return (; 
            mean_ask_price = μ_ask_price,
            std_ask_price  = σ_ask_price,
            range_ask_price = r_ask_price,
            mean_bid_price  = μ_bid_price,
            std_bid_price   = σ_bid_price,
            range_bid_price  = r_bid_price,
            mean_mid_price = μ_mid_price,
            std_mid_price  = σ_mid_price,
            range_mid_price = r_mid_price,
            n_points = length(mid_price))
end

describe_stats = describe(volatility_by_day[:, [:std_ask_price, :range_ask_price, :std_bid_price, :range_bid_price, :std_mid_price, :range_mid_price, :n_points]])

println("\nAggregated statistics on daily price volatility:")
println(describe_stats)


""" Plotting the aggregated volatility statistics """

# fz = 18
fz = 22

plot_kde_bandwidth = 5

ask_price_to_plot = volatility_by_day[!,"std_ask_price"]
bid_price_to_plot = volatility_by_day[!,"std_bid_price"]

# # Filter or trim the volatility distribution:
ask_price_to_plot = ask_price_to_plot[ask_price_to_plot .< 1000]
bid_price_to_plot = bid_price_to_plot[bid_price_to_plot .< 1000]

plt_density_24h = plot(
    title = "Distribution of price volatility across delivery days",
    size = (1800,1200),
    xtickfontsize = fz, ytickfontsize = fz,
    fontfamily = "Courier", 
    titlefontsize = fz,
    xguidefontsize = fz,
    yguidefontsize = fz,
    legendfontsize = fz-4,

    framestyle = :box,
    # margin = 10mm,

    left_margin=10mm, top_margin=10mm, right_margin=10mm, bottom_margin=10mm,

    xlim = (0.0-2*plot_kde_bandwidth,1000),
    ylim = (0.0,0.040)
)

density!(plt_density_24h,
    ask_price_to_plot,

    bandwidth = plot_kde_bandwidth, # kernel bandwidth in kernel density estimation (KDE)
 
    label = "Ask price\n",
    xlabel = "Standard deviation of prices, EUR/MWh",
    ylabel = "Density",
    color = palette(:bluesreds)[1],
    w = 6,
    alpha = 0.9,

    fillrange = 0,
    fill = (palette(:bluesreds)[1], 0.3),
)


density!(plt_density_24h,
    bid_price_to_plot,

    bandwidth = plot_kde_bandwidth, # kernel bandwidth in kernel density estimation (KDE)
 
    label = "Bid price\n",
    xlabel = "Standard deviation of prices, EUR/MWh",
    ylabel = "Density",
    color = palette(:bluesreds)[3],
    w = 6,
    alpha = 0.9,

    fillrange = 0,
    fill = (palette(:bluesreds)[3], 0.3),
)

display(plt_density_24h)

savefig("results/compute_volatility_24h_test1.png")
# savefig("results/compute_volatility_24h_test1.svg")
# savefig("results/compute_volatility_24h_test1.pdf")