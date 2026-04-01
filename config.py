# src/config.py

DATA_PATH = "/Users/ayamkattel/Desktop/MISC_LATEST/Nepali-Speech-Recognition/data/raw"
PROCESSED_DATA_PATH = "/Users/ayamkattel/Desktop/MISC_LATEST/Nepali-Speech-Recognition/data/processed"
MODEL_SAVE_PATH = "/Users/ayamkattel/Desktop/MISC_LATEST/Nepali-Speech-Recognition/models/nepali_asr_model.h5"

# 8 Nepali sentence categories
SENTENCES = [
    'Ma_Khusi_Xu',
    'Malai_Mero_Desh_Pyaro_Lagxa',
    'Namaste',
    'Ram_Le_Vaat_Khanxa',
    'Tapaiko_Ghar_Kaha_Xa',
    'TimiLai_Kasto_Chha',
    'Uh_Mero_Mitra_Ho',
    'Yo_Hamro_AI_Ko_Project_ho'
] 
NUM_CLASSES = len(SENTENCES)

# Audio processing parameters
SAMPLE_RATE = 22050
DURATION = 4.224 
N_MFCC = 13