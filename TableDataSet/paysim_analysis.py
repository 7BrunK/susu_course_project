import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path):
    df = pd.read_csv(path)
    print("Размер датасета:", df.shape)
    return df


def statistical_analysis(df):
    stats = df.describe()
    stats.to_csv("tables/statistics.csv", sep=';')

    type_counts = df['type'].value_counts()
    type_counts.to_csv("tables/type_counts.csv", sep=';')

    type_ratio = df['type'].value_counts(normalize=True)
    type_ratio.to_csv("tables/type_ratio.csv", sep=';')

    print("\nСтатистика сохранена в tables/")


def plot_feature_distributions(df):
    numeric_cols = [
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest'
    ]

    plt.figure(figsize=(12, 8))

    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(3, 2, i)
        plt.hist(df[col], bins=50)
        plt.title(f'{col}')

    plt.tight_layout()
    plt.savefig("plots/figure_1_distributions.png")
    plt.show()


def plot_feature_relationships(df):
    # Bar chart
    plt.figure()
    df['type'].value_counts().plot(kind='bar')
    plt.title('Распределение типов транзакций')
    plt.xlabel('Тип')
    plt.ylabel('Количество')
    plt.savefig("plots/figure_2_types.png")
    plt.show()

    # Boxplot
    plt.figure()
    sns.boxplot(x='type', y='amount', data=df)
    plt.title('Сумма транзакции по типу')
    plt.savefig("plots/figure_2_boxplot.png")
    plt.show()


def analyze_missing_values(df):
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100

    missing_table = pd.DataFrame({
        'missing_count': missing,
        'missing_percent': missing_percent
    })

    missing_table.to_csv("tables/missing_values.csv", sep=';')

    print("\nТаблица пропусков сохранена")


def correlation_analysis(df):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()

    corr.to_csv("tables/correlation_matrix.csv", sep=';')

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.title('Матрица корреляций')
    plt.savefig("plots/figure_3_correlation.png")
    plt.show()


def remove_duplicates(df):
    duplicates = df.duplicated().sum()

    dup_df = pd.DataFrame({'duplicates': [duplicates]})
    dup_df.to_csv("tables/duplicates.csv", index=False)

    print(f"\nКоличество дубликатов: {duplicates}")

    return df.drop_duplicates()


def analyze_outliers(df):
    numeric_cols = [
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest'
    ]

    plt.figure(figsize=(12, 8))

    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(3, 2, i)
        plt.boxplot(df[col])
        plt.title(f'Выбросы {col}')

    plt.tight_layout()
    plt.savefig("plots/figure_4_outliers.png")
    plt.show()


def filter_data(df):
    df = df[df['type'].isin(['PAYMENT', 'TRANSFER', 'CASH_OUT'])]
    df = df.drop(['nameOrig', 'nameDest'], axis=1)

    df['type'].value_counts().to_csv("tables/filtered_type_counts.csv", sep=';')

    print("\nПосле фильтрации сохранено распределение классов")

    return df


def add_noise(df):
    noise = np.random.normal(0, 0.01, size=len(df))
    df['amount_noisy'] = df['amount'] * (1 + noise)

    df.head(100).to_csv("tables/sample_with_noise.csv", index=False, sep=';')

    return df


def main():
    path = 'PS_20174392719_1491204439457_log.csv'
    df = load_data(path)

    statistical_analysis(df)

    plot_feature_distributions(df)      # Рисунок 1
    plot_feature_relationships(df)      # Рисунок 2
    analyze_missing_values(df)          # Таблица 1
    correlation_analysis(df)            # Рисунок 3

    df = remove_duplicates(df)
    analyze_outliers(df)                # Рисунок 4

    df = filter_data(df)
    df = add_noise(df)
    df.to_csv("output.csv", index=False, sep=';')

if __name__ == "__main__":
    main()