from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict, Literal, Annotated
from dotenv import load_dotenv

load_dotenv()

# Define the structured output schema using TypedDict
class TherapyRecommendationDict(TypedDict):
    """
    TypedDict schema for therapy recommendations.
    
    Unlike Pydantic, TypedDict provides type hints without runtime validation.
    It's lighter weight but offers less control over data validation.
    """
    TherapyCategory: Literal["Yoga_and_Pranayam", "Mindfulness", "Kirtan_and_Mantras", "Qawwali", "Storytelling", "Meditative_Music"]
    TherapySubcategory: Annotated[str, """Select exactly one option from the therapy category you chose:
    
    Kirtan_and_Mantras:
    - Relaxing mantra chants for inner peace~18
    - Soothing Kirtan music for meditation~19
    - Powerful Kirtan music for devotion and peace~20
    - Chanting Mala Hare krishna~21
    - CCL Mantra Meditation~58
    - OM Mantra Chanting (RRU)~64
    - Kirtan-1 (Chanting in a Helpless Mood)~70
    - Kirtan-2 (Chanting in a Helpless Mood)~71
    - Hare Krishna Mahamantra - Darbari Raga~72
    - Hare Krishna Mahamantra - Charukesi Raga~73
    
    Qawwali:
    - Assalaame Hazrate Makhdoom Sabir~31
    - Mujhe Sabir Se Hai Nisbat~32
    - Rab Ne Sabir Jisko Kiya Hai~33
    - Atte Hi Rahate Hum To~34
    
    Meditative_Music:
    - Relaxing music for stress relief ~15
    - Piano instrumental for calmness and sleep~16
    - Bollywood instrumental for relaxation~17
    - Shah e Madina Naat Sharif Flute Instrumental~22
    - Qasid Burda Sharif: Relaxing Flute Instrumental~23
    - Islamic Music Background: Sufi Meditation~24
    - Amjad Sabri-Naat Sharif - Flute Instrumental~25
    - Mera Dil Badal De - Peaceful & Soothing~26
    - Wohi Khuda Hai | Flute Instrumental~27
    - Allahu (Heart Touching Nasheed)~28
    - Azaan In The Heaven~29
    - Sufi Meditation Music Allah Madad~30
    - Emotional Background sounds | Islamic music~35
    - Relaxing Islamic Music, Traditional Arabic Music~36
    - ALONE WITH HOLY SPIRIT | Soaking Worship Music~37
    - REST IN JESUS - Soaking worship instrumental~38
    - Christian Meditation Music~39
    - PRAY: Deep Prayer Music~40
    - OCEANS | Soaking Worship Music~41
    - HOLY SPIRIT COME - Instrumental Soaking worship Music~42
    - LIVING HOPE - Soaking worship instrumental~43
    - MOMENT WITH GOD - Soothing Worship Instrumental~44
    - Peace, Be Still! I Speak Calm~45
    - CALM - Soaking Worship Instrumental~46
    - Listen to the rain on the forest path [Instrumental]~47
    - Sea Animals for Relaxation & Calming Music [Instrumental]~48
    - Relaxing Piano Music with Soft Rain Sounds [Instrumental]~49
    - Calm Music, Relax, Meditation, Stress Relief, Spa, Study, Sleep [Instrumental]~50
    - Relaxing Music with Ocean Waves [Instrumental]~51
    - Masterpieces for Relaxation and the Soul [Instrumental]~52
    - Relaxing Sleep Music for Meditation [Instrumental]~53
    - Relaxing Piano [Instrumental]~54
    - Himalayan Flute Music [Instrumental]~55
    - Sound Healing For Deep Relaxation [Instrumental]~56
    - CCL Classical Music Therapy~59
    
    Yoga_and_Pranayam:
    - Kapalbhati for energizing and detoxifying~3
    - Anulom Vilom for relaxation and mental clarity~8
    - Brahamari Pranayam for calming the mind~1
    - Sheetali pranayam for cooling the body and mind~2
    - CCL Anulom Vilom (Simple Alternate-Nostril Breathing)~60
    - CCL Sheetali (Cooling Breath)~62
    - 10 Minutes YOG NIDRA~65
    
    Mindfulness:
    - Guided mindfulness meditation for stress relief~9
    - Vipassana meditation for relaxation~10
    - Guided meditation for focus and stress relief~11
    - Guided meditation to release stress and anxiety~12
    - Various guided meditations for energy and relaxation~13
    - Guided Meditation [Instrumental]~57
    - CCL Guided Meditation~61
    - Walking Meditation for Peace of Mind~66
    
    Storytelling:
    - Empty Your Mind - Motivational Story~74
    - The Wrong Door That Opened the Right Future~75
    - The Untold Story of Kurma Avatar~76
    """]
    Rationale: Annotated[str, "Provide a detailed rationale for the chosen therapy and subcategory, explaining why it's suitable for the patient's profile"]

def validate_response_manually(response: dict) -> bool:
    """
    Manual validation function for TypedDict responses.
    
    Since TypedDict doesn't provide runtime validation like Pydantic,
    we need to implement our own validation logic if strict enforcement is needed.
    """
    
    # Define valid subcategories (same as in Pydantic example)
    VALID_SUBCATEGORIES = {
        "Kirtan_and_Mantras": [
            "Relaxing mantra chants for inner peace~18",
            "Soothing Kirtan music for meditation~19",
            "Powerful Kirtan music for devotion and peace~20",
            "Chanting Mala Hare krishna~21",
            "CCL Mantra Meditation~58",
            "OM Mantra Chanting (RRU)~64",
            "Kirtan-1 (Chanting in a Helpless Mood)~70",
            "Kirtan-2 (Chanting in a Helpless Mood)~71",
            "Hare Krishna Mahamantra - Darbari Raga~72",
            "Hare Krishna Mahamantra - Charukesi Raga~73"
        ],
        "Qawwali": [
            "Assalaame Hazrate Makhdoom Sabir~31",
            "Mujhe Sabir Se Hai Nisbat~32",
            "Rab Ne Sabir Jisko Kiya Hai~33",
            "Atte Hi Rahate Hum To~34"
        ],
        "Meditative_Music": [
            "Relaxing music for stress relief ~15",
            "Piano instrumental for calmness and sleep~16",
            "Bollywood instrumental for relaxation~17",
            "Shah e Madina Naat Sharif Flute Instrumental~22",
            "Qasid Burda Sharif: Relaxing Flute Instrumental~23",
            "Islamic Music Background: Sufi Meditation~24",
            "Amjad Sabri-Naat Sharif - Flute Instrumental~25",
            "Mera Dil Badal De - Peaceful & Soothing~26",
            "Wohi Khuda Hai | Flute Instrumental~27",
            "Allahu (Heart Touching Nasheed)~28",
            "Azaan In The Heaven~29",
            "Sufi Meditation Music Allah Madad~30",
            "Emotional Background sounds | Islamic music~35",
            "Relaxing Islamic Music, Traditional Arabic Music~36",
            "ALONE WITH HOLY SPIRIT | Soaking Worship Music~37",
            "REST IN JESUS - Soaking worship instrumental~38",
            "Christian Meditation Music~39",
            "PRAY: Deep Prayer Music~40",
            "OCEANS | Soaking Worship Music~41",
            "HOLY SPIRIT COME - Instrumental Soaking worship Music~42",
            "LIVING HOPE - Soaking worship instrumental~43",
            "MOMENT WITH GOD - Soothing Worship Instrumental~44",
            "Peace, Be Still! I Speak Calm~45",
            "CALM - Soaking Worship Instrumental~46",
            "Listen to the rain on the forest path [Instrumental]~47",
            "Sea Animals for Relaxation & Calming Music [Instrumental]~48",
            "Relaxing Piano Music with Soft Rain Sounds [Instrumental]~49",
            "Calm Music, Relax, Meditation, Stress Relief, Spa, Study, Sleep [Instrumental]~50",
            "Relaxing Music with Ocean Waves [Instrumental]~51",
            "Masterpieces for Relaxation and the Soul [Instrumental]~52",
            "Relaxing Sleep Music for Meditation [Instrumental]~53",
            "Relaxing Piano [Instrumental]~54",
            "Himalayan Flute Music [Instrumental]~55",
            "Sound Healing For Deep Relaxation [Instrumental]~56",
            "CCL Classical Music Therapy~59"
        ],
        "Yoga_and_Pranayam": [
            "Kapalbhati for energizing and detoxifying~3",
            "Anulom Vilom for relaxation and mental clarity~8",
            "Brahamari Pranayam for calming the mind~1",
            "Sheetali pranayam for cooling the body and mind~2",
            "CCL Anulom Vilom (Simple Alternate-Nostril Breathing)~60",
            "CCL Sheetali (Cooling Breath)~62",
            "10 Minutes YOG NIDRA~65"
        ],
        "Mindfulness": [
            "Guided mindfulness meditation for stress relief~9",
            "Vipassana meditation for relaxation~10",
            "Guided meditation for focus and stress relief~11",
            "Guided meditation to release stress and anxiety~12",
            "Various guided meditations for energy and relaxation~13",
            "Guided Meditation [Instrumental]~57",
            "CCL Guided Meditation~61",
            "Walking Meditation for Peace of Mind~66"
        ],
        "Storytelling": [
            "Empty Your Mind - Motivational Story~74",
            "The Wrong Door That Opened the Right Future~75",
            "The Untold Story of Kurma Avatar~76"
        ]
    }
    
    # Check if all required fields are present
    required_fields = ['TherapyCategory', 'TherapySubcategory', 'Rationale']
    for field in required_fields:
        if field not in response:
            print(f"Missing required field: {field}")
            return False
    
    # Validate category
    category = response['TherapyCategory']
    valid_categories = ["Yoga_and_Pranayam", "Mindfulness", "Kirtan_and_Mantras", "Qawwali", "Storytelling", "Meditative_Music"]
    if category not in valid_categories:
        print(f"Invalid category: {category}")
        return False
    
    # Validate subcategory
    subcategory = response['TherapySubcategory']
    if category in VALID_SUBCATEGORIES:
        valid_options = VALID_SUBCATEGORIES[category]
        if subcategory not in valid_options:
            print(f"Invalid subcategory '{subcategory}' for category '{category}'")
            print(f"Valid options: {valid_options}")
            return False
    
    print("All validations passed!")
    return True

# Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# For TypedDict, we need to convert it to a Pydantic model or use a different approach
# Let's create a Pydantic wrapper for the TypedDict
from pydantic import BaseModel

# Convert TypedDict to Pydantic BaseModel for LangChain compatibility
class TherapyRecommendationPydantic(BaseModel):
    """Pydantic wrapper for TypedDict to work with LangChain's structured output."""
    TherapyCategory: Literal["Yoga_and_Pranayam", "Mindfulness", "Kirtan_and_Mantras", "Qawwali", "Storytelling", "Meditative_Music"]
    TherapySubcategory: Annotated[str, """Select exactly one option from the therapy category you chose:
    
    [Same options as in TypedDict - abbreviated for brevity]
    """]
    Rationale: Annotated[str, "Provide a detailed rationale for the chosen therapy and subcategory"]
    
    def to_dict(self) -> TherapyRecommendationDict:
        """Convert Pydantic model to TypedDict format."""
        return {
            'TherapyCategory': self.TherapyCategory,
            'TherapySubcategory': self.TherapySubcategory,
            'Rationale': self.Rationale
        }

# Create structured model (using Pydantic under the hood)
structured_model = model.with_structured_output(TherapyRecommendationPydantic)

# Example patient profiles for testing
patient_profiles = [
    {
        "name": "Hindu Female Patient",
        "profile": """
Gender: Female
Age: 42
Preferred Language: Hindi
Religion: Hindu
Stress Level: 4
Anxiety Level: 4
Relaxation Level: 0
Sleep Quality: 0
Focus Difficulty: 5
Emotional Flatness: 0
"""
    },
    {
        "name": "Muslim Male Patient", 
        "profile": """
Gender: Male
Age: 35
Preferred Language: Urdu
Religion: Muslim
Stress Level: 3
Anxiety Level: 5
Relaxation Level: 1
Sleep Quality: 2
Focus Difficulty: 4
Emotional Flatness: 2
"""
    }
]

def run_therapy_recommendation(patient_name: str, patient_profile: str):
    """Run therapy recommendation for a patient and validate the response."""
    
    print(f"\n{'='*60}")
    print(f"🩺 THERAPY RECOMMENDATION FOR: {patient_name}")
    print(f"{'='*60}")
    
    prompt = f"""You are a therapy assistant. Given the following patient profile:
{patient_profile}

IMPORTANT: You must select your TherapySubcategory from the exact list provided in the schema. Do not create new subcategory names. Use the exact text including the ID number.

Suggest an alternate therapy approach that takes into account their religion, language preference, and current symptom severity. Provide your full recommendation in English. Briefly explain why this approach may help them, focusing on cultural relevance, language comfort, and their reported symptoms.
If relevant, mention evidence-based therapy modalities known to help people with similar profiles.
"""

    try:
        # Get structured response (returns Pydantic model)
        pydantic_response = structured_model.invoke(prompt)
        
        # Convert to TypedDict format for demonstration
        response = pydantic_response.to_dict()
        
        print(f"Category: {response['TherapyCategory']}")
        print(f"Subcategory: {response['TherapySubcategory']}")
        print(f"Rationale: {response['Rationale']}")
        
        # Show the type information
        print(f"\nTYPE INFORMATION:")
        print(f"   Pydantic response type: {type(pydantic_response)}")
        print(f"   TypedDict response type: {type(response)}")
        print(f"   TypedDict matches schema: {isinstance(response, dict)}")
        
        # Manual validation (demonstrating TypedDict validation approach)
        print(f"\nVALIDATION RESULTS:")
        is_valid = validate_response_manually(response)
        
        if is_valid:
            print("Response is valid and follows constraints!")
        else:
            print("Response failed validation!")
            
    except Exception as e:
        print(f"Error generating recommendation: {e}")

if __name__ == "__main__":
    print("LANGCHAIN STRUCTURED OUTPUTS WITH TYPEDDICT DEMO")
    print("=" * 60)
    print("This demo shows how to use TypedDict for structured outputs.")
    print("TypedDict provides type hints but requires manual validation")
    print("for strict constraint enforcement, unlike Pydantic.")
    
    # Run recommendations for different patient profiles
    for patient in patient_profiles:
        run_therapy_recommendation(patient["name"], patient["profile"])
    
    print(f"\n{'='*60}")
    print("COMPARISON: TypedDict vs Pydantic")
    print("=" * 60)
    print("TypedDict Pros:")
    print(" Lighter weight (no extra dependencies)")
    print(" Simple type hints")
    print(" Good IDE support")
    print(" Compatible with standard library")

    print("\nTypedDict Cons:")
    print(" No automatic runtime validation")
    print(" No field validators or custom logic")
    print(" Manual validation code required")
    print(" Less structured error handling")

    print("\nPydantic Pros:")
    print(" Automatic runtime validation")
    print(" Custom validators and field logic")
    print(" Detailed error messages")
    print(" Data serialization/deserialization")
    
    print("\nPydantic Cons:")
    print(" Additional dependency")
    print(" Slightly more complex")
    print(" Potential performance overhead")

    print("\nRECOMMENDATION:")
    print("Use Pydantic for complex validation needs,")
    print("TypedDict for simple type hints without validation.")