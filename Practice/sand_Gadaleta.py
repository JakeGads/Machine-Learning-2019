from knn import knn

if __name__ == "__main__":
    knn("Data/glass.csv", "RI", max_k=15, max_perm=2)
