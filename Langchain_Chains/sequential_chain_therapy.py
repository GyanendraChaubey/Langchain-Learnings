from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1,
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

# Define available categories and subcategories
CATEGORIES = ["Yoga_and_Pranayam", "Mindfulness", "Kirtan_and_Mantras", "Qawwali", "Storytelling", "Meditative_Music"]

SUBCATEGORIES = {
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

# Step 1: Select therapy category based on patient profile
template1 = PromptTemplate(
    template="""You are a therapy recommendation expert. Based on the patient profile, select ONE therapy category from the available options.

Patient Profile:
{patient_profile}

Available Categories: {categories}

Instructions:
- Analyze the patient's stress level (4), anxiety level (4), sleep quality (0), and focus difficulty (5)
- Select ONLY ONE category name from the available categories
- Respond with ONLY the category name, nothing else (e.g., "Mindfulness" or "Yoga_and_Pranayam")

Selected Category:""",
    input_variables=["patient_profile", "categories"]
)

# Step 2: Select subcategory based on category and patient profile
template2 = PromptTemplate(
    template="""Based on the patient profile and the selected therapy category, recommend ONE specific subcategory.

Patient Profile:
{patient_profile}

Selected Category: {therapy_category}

Available Subcategories for {therapy_category}:
{subcategories}

Instructions:
- Choose the most appropriate subcategory from the list above
- Respond with ONLY the complete subcategory text (including the ~number), nothing else

Selected Subcategory:""",
    input_variables=["patient_profile", "therapy_category", "subcategories"]
)

# Step 3: Provide rationale based on all three
template3 = PromptTemplate(
    template="""Provide a detailed rationale for recommending this therapy to the patient.

Patient Profile:
{patient_profile}

Recommended Category: {therapy_category}

Recommended Subcategory: {therapy_subcategory}

Instructions:
- Explain why this therapy is suitable for this patient
- Reference the patient's stress level (4), anxiety level (4), sleep quality (0), and focus difficulty (5)
- Keep the rationale concise but informative (2-3 sentences)

Rationale:""",
    input_variables=["patient_profile", "therapy_category", "therapy_subcategory"]
)

# Define patient profile
patient_profile = """
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

print("\n=== Sequential Chain Execution ===\n")

# Step 1: Get therapy category
print("Step 1: Selecting therapy category...")
chain1 = template1 | model | parser
therapy_category = chain1.invoke({
    "patient_profile": patient_profile,
    "categories": ", ".join(CATEGORIES)
}).strip()
print(f"Selected Category: {therapy_category}\n")

# Step 2: Get subcategory
print("Step 2: Selecting subcategory...")
# Get subcategories for the selected category
category_clean = therapy_category.strip()
available_subcategories = SUBCATEGORIES.get(category_clean, [])

if not available_subcategories:
    print(f"Warning: No subcategories found for '{category_clean}'. Trying to match...")
    # Try to find a matching category
    for cat in CATEGORIES:
        if cat.lower() in category_clean.lower() or category_clean.lower() in cat.lower():
            available_subcategories = SUBCATEGORIES.get(cat, [])
            therapy_category = cat
            break

chain2 = template2 | model | parser
therapy_subcategory = chain2.invoke({
    "patient_profile": patient_profile,
    "therapy_category": therapy_category,
    "subcategories": "\n".join(available_subcategories)
}).strip()
print(f"Selected Subcategory: {therapy_subcategory}\n")

# Step 3: Get rationale
print("Step 3: Generating rationale...")
chain3 = template3 | model | parser
rationale = chain3.invoke({
    "patient_profile": patient_profile,
    "therapy_category": therapy_category,
    "therapy_subcategory": therapy_subcategory
}).strip()

print("\n=== Final Therapy Recommendation ===")
print(f"Category: {therapy_category}")
print(f"Subcategory: {therapy_subcategory}")
print(f"\nRationale: {rationale}")