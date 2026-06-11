# print ("Please input Japanese information")
# name = input("Enter your name: ")
# age = int(input("Enter your age: "))
# school1 = input("Enter your school name: ")
# print ("Your name is " + name)
# print ("Your age is " + str(age))
# print ("Your school name is " + school1)

# x = 5
# y = 10
# z = x+y
# print (z)

# n1 = "chan"
# n2 = "Ka"
# n3 = n1 + " " + n2
# print (n3)


# print ("The result of x*y is " + str(z))

# a = 5

# if (5 == 5):
#     print ("1 is less than or equal to 5")
# else
#     print ("1 is greater than 5")

# x = 13%3   

# num1 = float(input("1st number: "))
# num2 = int(input("2nd number: "))
# num3 = int(input("3rd number: "))
# if num1 >= num2 and num1 >= num3:
#  MaxNum = num1
# elif num2 >= num1 and num2 >= num3:
#  MaxNum = num2
# elif num3 >= num1 and num3 >= num2:
#  MaxNum = num3
# print(MaxNum)

# student = ["apple", "banana", "pear", "mango"]
# print(student.index("mango"))   


# for i in range( 6,11):
#     print(i)

# sum = 0
# for  i in range (3,8):
#     sum = sum + i
# print ("3+4+5+6+7 is " + str(sum))










total = 0
no_of_subjects = 0
mark = 0
while mark >= 0:
    mark = int(input("Please enter your mark: "))
    if (mark >0 ):
        total = total + mark
        no_of_subjects = no_of_subjects + 1
average = total / no_of_subjects
print ("Your average :" + str(average))









