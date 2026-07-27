"""
suggestions.py
Rule-based health tips shown when a disease prediction is positive.
Every message set includes the mandatory disclaimer.
"""

DISCLAIMER = "This is general information, please consult a doctor."

SUGGESTIONS = {
    "diabetes": [
        "Monitor blood glucose levels regularly.",
        "Reduce intake of refined sugar and processed carbohydrates.",
        "Aim for at least 30 minutes of moderate exercise most days.",
        "Maintain a healthy body weight.",
        "Schedule an HbA1c test with your doctor.",
    ],
    "heart": [
        "Reduce salt and saturated fat intake.",
        "Engage in regular cardiovascular exercise (as advised by a doctor).",
        "Avoid smoking and limit alcohol.",
        "Monitor blood pressure and cholesterol regularly.",
        "Manage stress through relaxation techniques.",
    ],
    "parkinsons": [
        "Consult a neurologist for a detailed clinical evaluation.",
        "Physical therapy and regular movement can help manage symptoms.",
        "Maintain a balanced diet rich in antioxidants.",
        "Track tremors, stiffness, or balance changes over time.",
    ],
    "liver": [
        "Avoid alcohol consumption entirely.",
        "Reduce fatty and fried food intake.",
        "Stay hydrated and maintain a fiber-rich diet.",
        "Get liver function tests (LFT) done periodically.",
    ],
    "kidney": [
        "Monitor blood pressure and blood sugar closely.",
        "Reduce salt and protein intake as advised by a doctor.",
        "Stay well hydrated unless a doctor advises fluid restriction.",
        "Avoid overuse of painkillers (NSAIDs).",
    ],
    "breast_cancer": [
        "Consult an oncologist for further diagnostic imaging and biopsy.",
        "Regular self-examinations and screening mammograms are important.",
        "Maintain a healthy weight and stay physically active.",
    ],
    "stroke": [
        "Control blood pressure, cholesterol, and blood sugar levels.",
        "Avoid smoking and limit alcohol intake.",
        "Engage in regular physical activity as advised by a doctor.",
        "Learn to recognize stroke warning signs (F.A.S.T.).",
    ],
    "hepatitis": [
        "Avoid alcohol and hepatotoxic medications.",
        "Get vaccinated for Hepatitis A/B if not already done.",
        "Maintain good hygiene and safe practices to prevent transmission.",
        "Follow up with a hepatologist for liver function monitoring.",
    ],
    "thyroid": [
        "Get TSH, T3, and T4 levels tested regularly.",
        "Maintain a balanced diet with adequate iodine.",
        "Track symptoms like fatigue, weight changes, or mood changes.",
    ],
    "lung_cancer": [
        "Consult a pulmonologist or oncologist immediately for further tests.",
        "Avoid smoking and exposure to secondhand smoke or pollutants.",
        "Get a chest CT scan or biopsy if recommended by a doctor.",
    ],
}


def get_suggestions(disease, prediction):
    """
    disease: str, one of the 10 disease keys
    prediction: int, 0 (negative/low risk) or 1 (positive/high risk)
    Returns: dict with "tips" (list[str]) and "disclaimer" (str)
    """
    if prediction == 0:
        return {
            "tips": [
                "Your results indicate low risk. Continue regular health check-ups.",
                "Maintain a balanced diet and stay physically active.",
            ],
            "disclaimer": DISCLAIMER,
        }

    tips = SUGGESTIONS.get(disease, ["Please consult a doctor for further evaluation."])
    return {
        "tips": tips,
        "disclaimer": DISCLAIMER,
    }
