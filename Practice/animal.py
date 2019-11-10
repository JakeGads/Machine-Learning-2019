"""
A classification problem requires that examples be classified into one of two or more classes.
A classification can have real-valued or discrete input variables.
A problem with two classes is often called a two-class or binary classification problem.
A problem with more than two classes is often called a multi-class classification problem.
A problem where an example is assigned multiple classes is called a multi-label classification problem.
"""

from classification import knn
from regression import poly_regression, multiple_regression, linears_regression

if __name__ == "__main__":
    # data headers
    # CLASS
        # Class_Number,Number_Of_Animal_Species_In_Class,Class_Type,Animal_Names
    # ZOO
        # animal_name,hair,feathers,eggs,milk,airborne,aquatic,predator,toothed,backbone,breathes,venomous,fins,legs,tail domestic,catsize,class_type

    # Does the class type depend on the byproducts (eggs,  milk) of an animal?

    # Does the class type depend on the physical features (hair, feathers, toothed, backbone, fins, legs, tail) of an animal? 

    # Is the class type defined by the predator and venomous nature of the animal?
