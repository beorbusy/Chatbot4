# Simple Addition Calculator
def add_numbers(num1, num2):
    return num1 + num2
# Input from the user
number1 = float(input("Enter the first number: "))
number2 = float(input("Enter the second number: "))
# Perform addition
result = add_numbers(number1, number2)
# Display the result
print(f"The sum of {number1} and {number2} is {result}")
