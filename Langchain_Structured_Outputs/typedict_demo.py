from typing import TypedDict

class RecommendedTherapy(TypedDict):
    TherapyCategory: str
    TherapySubcategory: str
    Rationale: str

therapy: RecommendedTherapy = { "TherapyCategory": "Yoga_and_Pranayama", 
                               "TherapySubcategory": "10 Minutes Yog Nidra", 
                               "Rationale": "Yog Nidra is a powerful relaxation technique that helps reduce stress and anxiety, promoting mental clarity and emotional balance." }


print(therapy)
