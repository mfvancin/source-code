# Variables and Data Types
name = "Alice"        # String
age = 30              # Integer
height = 5.7          # Float
is_student = True     # Boolean

# Printing the variables
print(name)           # Outputs: Alice
print(age)            # Outputs: 30
print(height)         # Outputs: 5.7
print(is_student)     # Outputs: True

# Basic Operators
a = 10
b = 3

print(a + b)  # Addition: 13
print(a - b)  # Subtraction: 7
print(a * b)  # Multiplication: 30
print(a / b)  # Division: 3.333...
print(a // b) # Floor Division: 3
print(a % b)  # Modulus (remainder): 1
print(a ** b) # Exponentiation: 1000

print(a == b)  # Equals: False
print(a != b)  # Not Equals: True
print(a > b)   # Greater than: True
print(a < b)   # Less than: False
print(a >= b)  # Greater than or equal to: True
print(a <= b)  # Less than or equal to: False

a = 10
a += 3   # a = a + 3: a becomes 13
a -= 2   # a = a - 2: a becomes 11
a *= 2   # a = a * 2: a becomes 22
a /= 4   # a = a / 4: a becomes 5.5

# Input and Output
# Getting input from the user
name = input("What is your name? ")
print(f"Hello, {name}!")

# Getting numerical input
age = int(input("How old are you? "))
print(f"You will be {str(age + 1)} years old next year.")

# Conditional Statements
age = 18
if age >= 18:
    print("You are an adult.")
else:
    print("You are a minor.")

age = 20
if age < 18:
    print("You are a minor.")
elif age == 18:
    print("You just became an adult!")
else:
    print("You are an adult.")

# Loops
count = 1

for count in range(1, 6):
    print("Count is:", count)
    count += 1

# Using a for loop to iterate over a range of numbers
for i in range(1, 6):
    print("Number:", i)

# Looping through a list
fruits = ["apple", "banana", "cherry"]

for fruit in fruits:
    print(fruit)

# Break and Continue
for i in range(1, 10):
    if i == 5:
        break  # Exit the loop when i is 5
    if i % 2 == 0:
        continue  # Skip even numbers
    print(i)

# Defining Functions
def greet():
    print("Hello, world!")

greet()

# Function Arguments
def greet(name):
    print(f"Hello, {name}!")

def add(a, b):
    return a + b

result = add(5, 3)
print(result)  # Outputs: 8

# Default Arguments
def greet(name="stranger"):
    print(f"Hello, {name}!")
    
# Keyword Arguments
def describe_pet(animal_type, pet_name):
    print(f"I have a {animal_type} named {pet_name}.")
    
describe_pet(animal_type="dog", pet_name="Rex")
describe_pet(pet_name="Whiskers", animal_type="cat")

# Return Values
def square(x):
    return x * x

result = square(4)
print(result)  # Outputs: 16

# Scope and Lifetime of Variables
def my_function():
    x = 10  # Local variable
    print("Inside the function, x =", x)

x = 20  # Global variable
my_function()
print("Outside the function, x =", x)

# Lists #
# Creating a list
fruits = ["apple", "banana", "cherry"]

# Accessing elements
print(fruits[0])  # Outputs: apple

# Modifying elements
fruits[1] = "blueberry"
print(fruits)  # Outputs: ['apple', 'blueberry', 'cherry']

# Adding elements
fruits.append("orange")
print(fruits)  # Outputs: ['apple', 'blueberry', 'cherry', 'orange']

# Removing elements
fruits.remove("apple")
print(fruits)  # Outputs: ['blueberry', 'cherry', 'orange']

# Slicing a list
print(fruits[1:3])  # Outputs: ['cherry', 'orange']

# Tuples #
# Creating a tuple
coordinates = (10, 20)

# Accessing elements
print(coordinates[0])  # Outputs: 10

# Tuples are immutable, so you can't do this:
# coordinates[0] = 15  # This would raise an error

# Dictionaries #
# Creating a dictionary
person = {
    "name": "Alice",
    "age": 30,
    "city": "New York"
}

# Accessing values
print(person["name"])  # Outputs: Alice

# Modifying values
person["age"] = 31
print(person)  # Outputs: {'name': 'Alice', 'age': 31, 'city': 'New York'}

# Adding new key-value pairs
person["email"] = "alice@example.com"
print(person)  # Outputs: {'name': 'Alice', 'age': 31, 'city': 'New York', 'email': 'alice@example.com'}

# Removing key-value pairs
del person["city"]
print(person)  # Outputs: {'name': 'Alice', 'age': 31, 'email': 'alice@example.com'}

# Sets #
# Creating a set
fruits = {"apple", "banana", "cherry"}

# Adding elements
fruits.add("orange")
print(fruits)  # Outputs: {'apple', 'orange', 'cherry', 'banana'}

# Removing elements
fruits.remove("banana")
print(fruits)  # Outputs: {'apple', 'orange', 'cherry'}

# Checking membership
print("apple" in fruits)  # Outputs: True

# Set operations
set1 = {1, 2, 3}
set2 = {3, 4, 5}

print(set1 | set2)  # Union: {1, 2, 3, 4, 5}
print(set1 & set2)  # Intersection: {3}
print(set1 - set2)  # Difference: {1, 2}

# String Basics #
# Single and double quotes
greeting = "Hello"
response = 'Hi there'

# Triple quotes for multi-line strings
multi_line = """This is
a multi-line
string."""

# String Operations #
# Concatenation
full_name = "John" + " " + "Doe"
print(full_name)  # Outputs: John Doe

# Repetition
laugh = "Ha" * 3
print(laugh)  # Outputs: HaHaHa

# Slicing
text = "Hello, World!"
print(text[:5])  # Outputs: Hello
print(text[-6:])  # Outputs: World!

# String Methods #
# Converting to upper and lower case
text = "Hello, World!"
print(text.upper())  # Outputs: HELLO, WORLD!
print(text.lower())  # Outputs: hello, world!

# Replacing parts of a string
print(text.replace("World", "Python"))  # Outputs: Hello, Python!

# Splitting a string into a list
words = text.split(", ")
print(words)  # Outputs: ['Hello', 'World!']

# Joining a list into a string
new_text = " ".join(words)
print(new_text)  # Outputs: Hello World!

# String Formatting #
name = "Alice"
age = 30
text = f"My name is {name} and I am {age} years old."
print(text)  # Outputs: My name is Alice and I am 30 years old.

text = f"My name is {name} and I am {age} years old."
print(text)  # Outputs: My name is Alice and I am 30 years old.

text = f"My name is {name} and I am {age} years old."
print(text)  # Outputs: My name is Alice and I am 30 years old.

# Escape Characters #
# Including quotes in a string
text = "He said, \"Hello!\""
print(text)  # Outputs: He said, "Hello!"

# Newline and tab characters
text = "First Line\nSecond Line\tIndented"
print(text)
# Outputs:
# First Line
# Second Line    Indented

# Opening and Closing Files
file = open("filename.txt", "mode")
# Do something with the file
file.close()

with open("example.txt", "r") as file:
    content = file.read()
print(content)
file.close()

# Reading Files
# Reading the entire file
with open("example.txt", "r") as file:
    content = file.read()
    print(content)

# Reading line by line
with open("example.txt", "r") as file:
    for line in file:
        print(line.strip())

# Reading lines into a list
with open("example.txt", "r") as file:
    lines = file.readlines()
    print(lines)

# Writing to Files
# Writing to a file (overwrites existing content)
with open("example.txt", "w") as file:
    file.write("Hello, World!\n")
    file.write("This is a new line.")

# Appending to a file
with open("example.txt", "a") as file:
    file.write("\nThis line is appended.")

# Working with File Paths
# Absolute path (specific to your system)
file = open("/path/to/your/file/example.txt", "r")

# Relative path (relative to the current working directory)
file = open("data/example.txt", "r")

# Handling Exceptions
try:
    file = open("non_existent_file.txt", "r")
except FileNotFoundError:
    print("The file does not exist.")

# Working with CSV Files
import csv
with open("data.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)

import csv
with open("output.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Age", "City"])
    writer.writerow(["Alice", 30, "New York"])
    writer.writerow(["Bob", 25, "Los Angeles"])

# Understanding Exceptions
# ValueError: Raised when a function receives an argument of the correct type but an inappropriate value.
# TypeError: Raised when an operation is applied to an object of inappropriate type.
# IndexError: Raised when a list is indexed with an out-of-range index.
# KeyError: Raised when a dictionary is accessed with a key that doesn’t exist.
# FileNotFoundError: Raised when trying to open a file that doesn’t exist.

#The try-except Block
try:
    number = int(input("Enter a number: "))
    print("The number you entered is:", number)
except ValueError:
    print("That's not a valid number!")

# Catching Multiple Exceptions
try:
    x = int(input("Enter a number: "))
    y = int(input("Enter another number: "))
    result = x / y
    print("Result:", result)
except ValueError:
    print("Invalid input! Please enter a number.")
except ZeroDivisionError:
    print("You can't divide by zero!")

#Using else and finally
try:
    file = open("example.txt", "r")
    content = file.read()
except FileNotFoundError:
    print("File not found.")
else:
    print("File content:")
    print(content)
finally:
    if 'file' in locals():
        file.close()
        print("File closed.")

# Raising Exceptions
def check_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative!")
    else:
        print("Your age is:", age)

try:
    check_age(-1)
except ValueError as e:
    print(e)

# Custom Exceptions
class CustomError(Exception):
    pass

def check_value(x):
    if x > 100:
        raise CustomError("Value cannot be greater than 100!")
    else:
        print("Value is acceptable.")

try:
    check_value(150)
except CustomError as e:
    print(e)

# Basic Concepts of OOP
#Class: A blueprint for creating objects. It defines a set of attributes (data) and methods (functions) that the objects created from the class will have.
#Object: An instance of a class. When a class is defined, no memory is allocated until an object of that class is created.
#Attribute: A variable that belongs to a class or an object.
#Method: A function that is defined within a class and can be called on objects of that class.
#Inheritance: A mechanism where a new class (child class) inherits attributes and methods from an existing class (parent class).
#Encapsulation: The practice of keeping an object's data (attributes) private and providing controlled access through methods.
#Polymorphism: The ability to use a method or operation in different ways for different data types or classes.

# Creating a Class and Object
class Dog:
    # Class attribute
    species = "Canis familiaris"

    # Initializer (constructor)
    def __init__(self, name, age):
        self.name = name  # Instance attribute
        self.age = age    # Instance attribute

    # Method to describe the dog
    def describe(self):
        return f"{self.name} is {self.age} years old."

    # Method to make the dog bark
    def bark(self, sound):
        return f"{self.name} says {sound}"

# Creating objects
dog1 = Dog("Buddy", 3)
dog2 = Dog("Lucy", 5)

# Accessing attributes and methods
print(dog1.describe())  # Outputs: Buddy is 3 years old.
print(dog2.bark("Woof"))  # Outputs: Lucy says Woof

# Inheritance
# Parent class
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f"{self.name} makes a sound."

# Child class (inherits from Animal)
class Dog(Animal):
    def speak(self):
        return f"{self.name} barks."

# Creating objects
animal = Animal("Generic Animal")
dog = Dog("Buddy")

# Accessing methods
print(animal.speak())  # Outputs: Generic Animal makes a sound.
print(dog.speak())  # Outputs: Buddy barks.

# Encapsulation
class Account:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.__balance = balance  # Private attribute

    def deposit(self, amount):
        self.__balance += amount
        return f"Added {amount}. New balance is {self.__balance}"

    def withdraw(self, amount):
        if amount > self.__balance:
            return "Insufficient funds."
        self.__balance -= amount
        return f"Withdrew {amount}. New balance is {self.__balance}"

    def get_balance(self):
        return self.__balance

# Creating an object
account = Account("John", 100)

# Accessing methods
print(account.deposit(50))  # Outputs: Added 50. New balance is 150
print(account.withdraw(30))  # Outputs: Withdrew 30. New balance is 120
print(account.get_balance())  # Outputs: 120

# Polymorphism
class Cat:
    def speak(self):
        return "Meow"

class Dog:
    def speak(self):
        return "Woof"

# Function that takes any animal and calls its speak method
def make_animal_speak(animal):
    print(animal.speak())

# Creating objects
cat = Cat()
dog = Dog()

# Using polymorphism
make_animal_speak(cat)  # Outputs: Meow
make_animal_speak(dog)  # Outputs: Woof

# Modules
# For example, if you have a file named mymodule.py, you can import and use it in another Python script.
# mymodule.py
def greet(name):
    return f"Hello, {name}!"

def add(a, b):
    return a + b

# main.py
import mymodule

print(mymodule.greet("Alice"))  # Outputs: Hello, Alice!
print(mymodule.add(5, 3))  # Outputs: 8

# You can also import specific functions from a module:
from mymodule import greet

print(greet("Bob"))  # Outputs: Hello, Bob!

# Packages
#mypackage/
#    __init__.py
#    module1.py
#    module2.py

# mypackage/module1.py
def foo():
    return "foo from module1"
# mypackage/module2.py
def bar():
    return "bar from module2"

from mypackage import module1, module2
print(module1.foo())  # Outputs: foo from module1
print(module2.bar())  # Outputs: bar from module2

# Standard Library
import math
print(math.sqrt(16))  # Outputs: 4.0

import datetime
now = datetime.datetime.now()
print(now)

import os
print(os.getcwd())  # Outputs the current working directory

# Third-Party Libraries
# Installing: pip install requests

import requests
response = requests.get("https://api.github.com")
print(response.status_code)  # Outputs: 200

# Libraries:
#NumPy: For numerical computations.
#Pandas: For data manipulation and analysis.
#Requests: For making HTTP requests.
#Flask: For building web applications.
#Django: For building more complex web applications.
#BeautifulSoup: For web scraping.

