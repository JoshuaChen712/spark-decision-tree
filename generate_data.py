import numpy as np
import pandas as pd


def generate_dataset(num_samples=1000):
    feature1_0 = np.random.randn(num_samples)
    feature2_0 = np.random.randn(num_samples)
    feature3_0 = np.random.randn(num_samples)

    feature1 = np.where(feature1_0 > 0, 1, 0)
    feature2 = np.where(feature2_0 > 0, 0, 1)
    feature3 = np.where(feature3_0 > 0, 1, 0)

    target = np.where((feature1_0 + feature2_0 + feature3_0) > 0, 1, 0)

    dataset = pd.DataFrame(
        {'feature1': feature1_0, 'feature2': feature2, 'feature3': feature3, 'target': target})

    return dataset


def save_dataset(dataset, filename):
    dataset.to_csv(filename, index=False)


if __name__ == "__main__":
    num_samples = 100000
    filename = "test_dataset_3.csv"

    dataset = generate_dataset(num_samples)
    save_dataset(dataset, filename)
    print(f"Test dataset with {num_samples} samples saved to {filename}.")
