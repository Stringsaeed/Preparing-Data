import preprocessingdata
import pandas as pd
import model


def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    train = train.drop(columns=['Cabin'])
    test = test.drop(columns=['Cabin'])

    train.Name = preprocessingdata.get_surname(train)
    test.Name = preprocessingdata.get_surname(test)

    train_Obj = preprocessingdata.get_object(train)
    test_Obj = preprocessingdata.get_object(test)
    for i in range(len(train_Obj)):
        train[train_Obj[i]] = preprocessingdata.encode(train[train_Obj[i]])
    for i in range(len(test_Obj)):
        test[test_Obj[i]] = preprocessingdata.encode(test[test_Obj[i]])

    train_na = preprocessingdata.get_naData(train)
    test_na = preprocessingdata.get_naData(test)

    for i in range(len(train_na)):
        train[train_na[i]] = preprocessingdata.fill(train[train_na[i]])
    for i in range(len(test_na)):
        test[test_na[i]] = preprocessingdata.fill(test[test_na[i]])

    X = train.drop(columns=['Survived'])
    y = train.Survived
    X_std = model.scl(X)
    mdl = model.svc(X_std, y)
    prd = pd.DataFrame(mdl.prdeict(test))
    prd.index += 891
    prd.to_csv("Sub.csv")


if __name__ == '__main__':
    main()
