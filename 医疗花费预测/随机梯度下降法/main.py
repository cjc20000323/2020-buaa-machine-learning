import pandas
from people import People

df = pandas.read_csv('train.csv')
rf = pandas.read_csv('../pythonProject2/test_sample.csv')

list1 = []
list2 = []
predict1 = []
predict2 = []
answer = []
w = [0, 0, 0, 0, 0, 0, 0, 0, 0]


def f(person, w):
    return w[0] + w[1] * person.age + w[2] * person.sex + w[3] * person.bmi + w[4] * person.children + \
           w[5] * person.smoker + w[6] * person.region


def train(train_time, a):
    for i in range(0, train_time):
        print(i)
        for info in list1:
            for j in range(0, 7):
                w[j] = w[j] - a * (f(info, w) - info.charges) * info.get_x(j)


def add_answer():
    for person in list2:
        predict2.append(f(person, w))


def deal_data():
    for index, row in df.iterrows():
        new_person = People(row['age'], row['sex'], row['bmi'], row['children'], row['smoker'], row['region'],
                            row['charges'])
        new_person.change_sex()
        new_person.change_smoker()
        new_person.change_region()
        list1.append(new_person)
    for index, row in rf.iterrows():
        new_person = People(row['age'], row['sex'], row['bmi'], row['children'], row['smoker'], row['region'],
                            row['charges'])
        new_person.change_sex()
        new_person.change_smoker()
        new_person.change_region()
        list2.append(new_person)


def print_data():
    for i in range(0, len(predict2)):
        list2[i].back_sex()
        list2[i].back_smoker()
        list2[i].back_region()
        info = {"age": list2[i].age, "sex": list2[i].sex, "bmi": list2[i].bmi, "children": list2[i].children,
                "smoker": list2[i].smoker, "region": list2[i].region, "charges": predict2[i]}
        answer.append(info)
        print(info)
    csvfile = pandas.DataFrame(answer, columns=["age", "sex", "bmi", "children", "smoker", "region", "charges"])
    print(csvfile)
    csvfile.to_csv("out.csv", index=False)


def main():
    deal_data()
    train(1000000, 0.00001)
    add_answer()
    print_data()


if __name__ == '__main__':
    main()
