from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):

    name: str = 'Anuj'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=8.72, description='A decimal value representing the cgpa of the student')


new_student = {'age':'32', 'email':'abc@gmail.com','cgpq':8.72}

student = Student(**new_student)
print(student)
student_dict = dict(student)

print(student_dict['age'])

student_json = student.model_dump_json()