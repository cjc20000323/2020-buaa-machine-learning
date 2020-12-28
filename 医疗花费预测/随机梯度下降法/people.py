class People:
    def __init__(self, age, sex, bmi, children, smoker, region, charges):
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = region
        self.charges = charges

    def change_sex(self):
        if self.sex == "male":
            self.sex = 1
        else:
            self.sex = 2

    def back_sex(self):
        if self.sex == 1:
            self.sex = "male"
        else:
            self.sex = "female"

    def change_smoker(self):
        if self.smoker == "no":
            self.smoker = 1
        else:
            self.smoker = 2

    def back_smoker(self):
        if self.smoker == 1:
            self.smoker = "no"
        else:
            self.smoker = "yes"

    def change_region(self):
        if self.region == "northeast":
            self.region = 1
        elif self.region == "northwest":
            self.region = 2
        elif self.region == "southeast":
            self.region = 3
        elif self.region == "southwest":
            self.region = 4

    def back_region(self):
        if self.region == 1:
            self.region = "northeast"
        elif self.region == 2:
            self.region = "northwest"
        elif self.region == 3:
            self.region = "southeast"
        elif self.region == 4:
            self.region = "southwest"

    def get_x(self, j):
        if j == 0:
            return 1
        elif j == 1:
            return self.age
        elif j == 2:
            return self.sex
        elif j == 3:
            return self.bmi
        elif j == 4:
            return self.children
        elif j == 5:
            return self.smoker
        elif j == 6:
            return self.region
