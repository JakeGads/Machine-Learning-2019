from classification import knn
from regression import poly_regression

if __name__ == "__main__":

    file = open('knn.csv', 'w+')
    file.write("X(s),y,score")
    for i in range(10):
        # knn(file, y, max_k=100, max_perm=0, supress_text=False):
        high = knn("Data/glass.csv", i, max_k=30, max_perm=3)
        file.write(high.writtable() + '\n')
    file.close()

    file = open('Regression.csv', 'w+')
    file.write("X,y,degree,score")
    for i in range(10):
        for h in range(10):
            if i == h:
                continue
            # def poly_regression(X_loc, y_loc, file, max_degree=10, supress_text=False):
            high = poly_regression(i, h, "Data/glass.csv")
            file.write(high.writtable() + '\n')

    file.close()
