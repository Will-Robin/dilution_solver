import pandas as pd
from dilution_solver import calculate_stock_volumes
from dilution_solver import validate_or_optimize


def run_from_files():
    # Provide proposed experiment design

    ## Stocks
    # Concentrations of four stock solutions, one for each component
    stock_df = pd.read_csv("data/stocks.csv")
    stock_c = stock_df.concentration.to_numpy()
    bounds = [(row.lower_bound, row.upper_bound) for _, row in stock_df.iterrows()]

    ## Targets
    target_df = pd.read_csv("data/targets.csv")
    targets_c = target_df.drop(columns=["sample_name", "volume"]).to_numpy()
    targets_v = target_df.volume.to_numpy()

    stock_concs = validate_or_optimize(stock_c, targets_c, targets_v, bounds)

    stock_volumes, solvent_volumes = calculate_stock_volumes(
        stock_concs, targets_c, targets_v
    )

    print("Suggested stock concentrations:\n", stock_concs)
    print("Suggested stock volumes:\n", stock_volumes)
    print("Minimum stock volumes:\n", stock_volumes.sum(axis=0))
    print("Suggested solvent additions:\n", solvent_volumes)

    # Create result output
    stock_design = pd.DataFrame()
    stock_design["stock_name"] = stock_df.stock_name
    stock_design["concentration"] = stock_concs
    stock_design["minimum_volume"] = stock_volumes.sum(axis=0)
    print(stock_design.head())

    target_design = pd.DataFrame(
        stock_volumes, columns=[f"{stock}_volume" for stock in stock_df.stock_name]
    )
    target_design["sample_name"] = target_df.sample_name
    target_design["solvent"] = solvent_volumes

    print(target_design.head())

    # Reorder columns for output
    cols = target_design.columns.to_list()
    cols.remove("sample_name")
    cols = ["sample_name"] + cols
    target_design = target_design[cols]
    stock_design.to_csv("data/stocks_design.csv", index=False)
    target_design.to_csv("data/sample_volumes.csv", index=False)


def main():
    run_from_files()


if __name__ == "__main__":
    main()
