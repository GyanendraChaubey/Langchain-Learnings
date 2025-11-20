from pydantic import BaseModel, Field
from typing import Optional


class Student(BaseModel):
    name: str
    age: int
    # email: Optional[EmailStr] = None
    cgpa: Optional[float] = Field(None, ge=0.0, le=10.0, description="Cumulative Grade Point Average on a scale of 0 to 10")

student = Student(name="Gyanendra", age='20', email="gyanendra@example.com", cgpa=8.5)

print(student)
