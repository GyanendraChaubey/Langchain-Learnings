from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, field_validator
from typing import Annotated, Literal
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

# Define valid subcategories for each category
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

# Flatten all subcategories for the field description
ALL_SUBCATEGORIES = []
for category, subcats in VALID_SUBCATEGORIES.items():
    ALL_SUBCATEGORIES.extend(subcats)

# schema
class TherapyRecommendation(BaseModel):
    TherapyCategory: Literal["Yoga_and_Pranayam", "Mindfulness", "Kirtan_and_Mantras", "Qawwali", "Storytelling", "Meditative_Music"]
    
    TherapySubcategory: Annotated[str, Field(
        description=f"""MANDATORY: You MUST select exactly one of these predefined options. No custom names allowed:
        
        {', '.join(ALL_SUBCATEGORIES)}
        
        Choose based on your selected TherapyCategory."""
    )]
    Rationale: Annotated[str, "Provide a rationale for the chosen therapy and subcategory"]
    
    @field_validator('TherapySubcategory')
    @classmethod
    def validate_subcategory(cls, v, info):
        """Ensure the subcategory is valid for the selected category."""
        # Get the category from the data being validated
        category = info.data.get('TherapyCategory')
        
        if category and category in VALID_SUBCATEGORIES:
            valid_options = VALID_SUBCATEGORIES[category]
            if v not in valid_options:
                # Provide helpful error message with valid options
                valid_list = '\n- '.join(valid_options)
                raise ValueError(
                    f"Invalid subcategory '{v}' for category '{category}'. "
                    f"Valid options are:\n- {valid_list}"
                )
        elif v not in ALL_SUBCATEGORIES:
            raise ValueError(f"'{v}' is not a valid subcategory. Must be one of the predefined options.")
        
        return v

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

structured_model = model.with_structured_output(TherapyRecommendation)

struct_model_response = structured_model.invoke(""" You are a therapy assistant. Given the following patient profile:
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

IMPORTANT: You must select your TherapySubcategory from the exact list provided in the schema. Do not create new subcategory names. Use the exact text including the ID number (e.g., "Assalaame Hazrate Makhdoom Sabir~31").

Suggest an alternate therapy approach that takes into account her religion, language preference, and current symptom severity. Provide your full recommendation in English. Briefly explain why this approach may help her, focusing on cultural relevance, language comfort, and her reported high stress, anxiety, very poor sleep and relaxation, and severe focus issues.
If relevant, mention evidence-based therapy modalities known to help people with similar profiles.
""")

# print(struct_model_response)
print("Category: ", struct_model_response.TherapyCategory)
print("\nSubcategory: ", struct_model_response.TherapySubcategory)
print("\nRationale: ", struct_model_response.Rationale)
