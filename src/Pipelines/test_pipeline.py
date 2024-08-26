import pandas as pd
import sys
from src.exception import CustomException
import os
from src.utils import load_object
class Predict_pipeline:
      def __init__(self):
            pass
      def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','Processor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
            
            
class CustomData:
    def __init__ (self, 
                Age: int,
                Number_of_sexual_partners: int,
                First_sexual_intercourse: int,
                Num_of_pregnancies: int,
                Smokes: int,
                Smokes_years: float,
                Smokes_packs_per_year: float,
                Hormonal_Contraceptives: int,
                Hormonal_Contraceptives_years: float,
                IUD: int,
                IUD_years: float,
                STDs: int,
                STDs_number: int,
                STDs_condylomatosis: int,
                STDs_cervical_condylomatosis: int,
                STDs_vaginal_condylomatosis: int,
                STDs_vulvo_perineal_condylomatosis: int,
                STDs_syphilis: int,
                STDs_pelvic_inflammatory_disease: int,
                STDs_genital_herpes: int,
                STDs_molluscum_contagiosum: int,
                STDs_AIDS: int,
                STDs_HIV: int,
                STDs_Hepatitis_B: int,
                STDs_HPV: int,
                STDs_Number_of_diagnosis: int,
                STDs_Time_since_first_diagnosis: float,
                STDs_Time_since_last_diagnosis: float,
                Dx_CIN: int,
                Dx_HPV: int,
                Dx: int,
                Hinselmann: int,
                Schiller: int,
                Citology: int,
                Biopsy: int):
                self.Age = Age
                self.Number_of_sexual_partners = Number_of_sexual_partners
                self.First_sexual_intercourse = First_sexual_intercourse
                self.Num_of_pregnancies = Num_of_pregnancies
                self.Smokes = Smokes
                self.Smokes_years = Smokes_years
                self.Smokes_packs_per_year = Smokes_packs_per_year
                self.Hormonal_Contraceptives = Hormonal_Contraceptives
                self.Hormonal_Contraceptives_years = Hormonal_Contraceptives_years
                self.IUD = IUD
                self.IUD_years = IUD_years
                self.STDs = STDs
                self.STDs_number = STDs_number
                self.STDs_condylomatosis = STDs_condylomatosis
                self.STDs_cervical_condylomatosis = STDs_cervical_condylomatosis
                self.STDs_vaginal_condylomatosis = STDs_vaginal_condylomatosis
                self.STDs_vulvo_perineal_condylomatosis = STDs_vulvo_perineal_condylomatosis
                self.STDs_syphilis = STDs_syphilis
                self.STDs_pelvic_inflammatory_disease = STDs_pelvic_inflammatory_disease
                self.STDs_genital_herpes = STDs_genital_herpes
                self.STDs_molluscum_contagiosum = STDs_molluscum_contagiosum
                self.STDs_AIDS = STDs_AIDS
                self.STDs_HIV = STDs_HIV
                self.STDs_Hepatitis_B = STDs_Hepatitis_B
                self.STDs_HPV = STDs_HPV
                self.STDs_Number_of_diagnosis = STDs_Number_of_diagnosis
                self.STDs_Time_since_first_diagnosis = STDs_Time_since_first_diagnosis
                self.STDs_Time_since_last_diagnosis = STDs_Time_since_last_diagnosis
                self.Dx_CIN = Dx_CIN
                self.Dx_HPV = Dx_HPV
                self.Dx = Dx
                self.Hinselmann = Hinselmann
                self.Schiller = Schiller
                self.Citology = Citology
                self.Biopsy = Biopsy
    def get_data_as_data_frame(self)-> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "Number_of_sexual_partners": [self.Number_of_sexual_partners],
                "First_sexual_intercourse": [self.First_sexual_intercourse],
                "Num_of_pregnancies": [self.Num_of_pregnancies],
                "Smokes": [self.Smokes],
                "Smokes_years": [self.Smokes_years],
                "Smokes_packs_per_year": [self.Smokes_packs_per_year],
                "Hormonal_Contraceptives": [self.Hormonal_Contraceptives],
                "Hormonal_Contraceptives_years": [self.Hormonal_Contraceptives_years],
                "IUD": [self.IUD],
                "IUD_years": [self.IUD_years],
                "STDs": [self.STDs],
                "STDs_number": [self.STDs_number],
                "STDs_condylomatosis": [self.STDs_condylomatosis],
                "STDs_cervical_condylomatosis": [self.STDs_cervical_condylomatosis],
                "STDs_vaginal_condylomatosis": [self.STDs_vaginal_condylomatosis],
                "STDs_vulvo_perineal_condylomatosis": [self.STDs_vulvo_perineal_condylomatosis],
                "STDs_syphilis": [self.STDs_syphilis],
                "STDs_pelvic_inflammatory_disease": [self.STDs_pelvic_inflammatory_disease],
                "STDs_genital_herpes": [self.STDs_genital_herpes],
                "STDs_molluscum_contagiosum": [self.STDs_molluscum_contagiosum],
                "STDs_AIDS": [self.STDs_AIDS],
                "STDs_HIV": [self.STDs_HIV],
                "STDs_Hepatitis_B": [self.STDs_Hepatitis_B],
                "STDs_HPV": [self.STDs_HPV],
                "STDs_Number_of_diagnosis": [self.STDs_Number_of_diagnosis],
                "STDs_Time_since_first_diagnosis": [self.STDs_Time_since_first_diagnosis],
                "STDs_Time_since_last_diagnosis": [self.STDs_Time_since_last_diagnosis],
                "Dx_CIN": [self.Dx_CIN],
                "Dx_HPV": [self.Dx_HPV],
                "Dx": [self.Dx],
                "Hinselmann": [self.Hinselmann],
                "Schiller": [self.Schiller],
                "Citology": [self.Citology],
                "Biopsy": [self.Biopsy]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
    