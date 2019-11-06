from knn import knn

if __name__ == "__main__":
    for i in range(10):
        knn("Data/glass.csv", i, max_k=15, max_perm=3, supressText=True)
