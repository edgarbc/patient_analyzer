import random
import matplotlib.pyplot as plt


# Define sample medical conditions, medications, and procedures
conditions = ['hypertension', 'diabetes', 'asthma', 'arthritis', 'heart disease', 'cancer']
medications = ['metformin', 'lisinopril', 'albuterol', 'ibuprofen', 'atorvastatin', 'aspirin']
procedures = ['angioplasty', 'colonoscopy', 'MRI', 'CT scan', 'blood test']

def generate_patient():
    return {
        'age': random.randint(20, 80),
        'gender': random.choice(['male', 'female']),
        'conditions': random.sample(conditions, random.randint(1, 3)),
        'medications': random.sample(medications, random.randint(1, 3)),
        'procedures': random.sample(procedures, random.randint(1, 2))
    }

def generate_patients(num_patients):
    return [generate_patient() for _ in range(num_patients)]

def visualize_patients(patients):
    """
    visualize the patients as a matrix where each row is a patient and each column is a condition, medication, or procedure
    
    Args:
        patients (list): a list of patients
    Returns:
        None
    """
    # use matplotlib to create a scatter plot of the patients
    # use seaborn to create a heatmap of the patients
    # use plotly to create an interactive scatter plot of the patients

    # create a matrix where each row is a patient and each column is a condition, medication, or procedure
    # use matplotlib to create a scatter plot of the patients
    # use seaborn to create a heatmap of the patients
    # use plotly to create an interactive scatter plot of the patients

    patient_matrix = []
    for patient in patients:
        patient_matrix.append(patient['conditions'] + patient['medications'] + patient['procedures'])

    # use matplotlib to create a scatter plot of the patients
    plt.scatter(patient_matrix[:, 0], patient_matrix[:, 1])
    plt.show()

def patient_to_text(patient):
    return f"Patient is a {patient['age']} year old {patient['gender']} with conditions: {', '.join(patient['conditions'])}, " \
           f"taking medications: {', '.join(patient['medications'])}, and has undergone procedures: {', '.join(patient['procedures'])}."




# code to test the function
if __name__ == "__main__":
    # Generate a population of patients
    num_patients = 10  # You can increase this number to create a larger population
    patients = generate_patients(num_patients)
    print(patients)
    visualize_patients(patients)

    # Convert all patients to text descriptions
    patient_texts = [patient_to_text(patient) for patient in patients]
    print(patient_texts)
