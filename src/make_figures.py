import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/insurance_clean.csv"
OUT_DIR = "reports/figures"

def main():
    df = pd.read_csv(DATA_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)

    plt.figure()
    df["charges"].hist(bins=30)
    plt.title("Distribution of Insurance Charges")
    plt.xlabel("charges")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/charges_distribution.png", dpi=200)
    plt.close()

    plt.figure()
    df.boxplot(column="charges", by="smoker")
    plt.title("Charges by Smoker")
    plt.suptitle("")
    plt.xlabel("smoker")
    plt.ylabel("charges")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/charges_by_smoker.png", dpi=200)
    plt.close()

    plt.figure()
    for s in df["smoker"].unique():
        subset = df[df["smoker"] == s]
        plt.scatter(subset["age"], subset["charges"], label=f"smoker={s}", alpha=0.6)
    plt.title("Charges vs Age (Smoker vs Non-smoker)")
    plt.xlabel("age")
    plt.ylabel("charges")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/charges_vs_age_smoker.png", dpi=200)
    plt.close()

    print(f"Saved figures to {OUT_DIR}/")

if __name__ == "__main__":
    main()