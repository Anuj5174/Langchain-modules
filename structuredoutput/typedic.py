from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    city: str

new_person : Person ={'name':'John', 'age':30, 'city':'New York'}   
#person = Person(name='John', age=30, city='New York')
#print(person)       
print(new_person)   