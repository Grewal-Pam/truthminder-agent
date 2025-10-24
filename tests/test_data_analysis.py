import pandas as pd
from local_datasets.data_analysis import DataAnalysis


def test_real_dataset_analysis():
    # Load the real dataset
    dataset_path = "data/filtered_data.tsv"
    df = pd.read_csv(dataset_path, sep="\t")

    # Initialize DataAnalysis
    analysis = DataAnalysis(output_dir="real_analysis_results")

    # Check for missing values
    analysis.check_missing_values(df)

    # Check class imbalance for 2_way_label
    dist_2_way, imbalance_ratio_2_way = analysis.check_class_imbalance(
        df, "2_way_label"
    )
    print(f"2-Way Label Distribution:\n{dist_2_way}")
    print(f"2-Way Label Imbalance Ratio: {imbalance_ratio_2_way:.2f}")

    # Check class imbalance for 3_way_label
    dist_3_way, imbalance_ratio_3_way = analysis.check_class_imbalance(
        df, "3_way_label"
    )
    print(f"3-Way Label Distribution:\n{dist_3_way}")
    print(f"3-Way Label Imbalance Ratio: {imbalance_ratio_3_way:.2f}")

    # Normalize metadata columns
    metadata_columns = ["num_comments", "score", "upvote_ratio"]
    normalized_df = analysis.normalize_metadata(
        df, metadata_columns, scaler_path="real_scaler.pkl"
    )

    # Generate plots for 2_way_label
    analysis.plot_class_distribution(normalized_df, "2_way_label")

    # Generate plots for 3_way_label
    analysis.plot_class_distribution(normalized_df, "3_way_label")

    # Check outliers in metadata columns
    outliers = analysis.check_outliers(normalized_df, metadata_columns)
    print(f"Outliers detected in metadata columns:\n{outliers}")

    # Generate summary report
    analysis.generate_summary_report(normalized_df)
    print("Summary Report Generated.")

    print("Real dataset analysis completed. Check 'real_analysis_results' for outputs.")


# Run the test
if __name__ == "__main__":
    test_real_dataset_analysis()
