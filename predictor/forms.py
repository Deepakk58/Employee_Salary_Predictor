from django import forms
import pandas as pd
import os

# Load and preprocess the dataset
csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'adult_3.csv')
df = pd.read_csv(csv_path)

# Clean dataset: remove '?' and drop nulls
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)

# Helper to generate sorted unique choices for a column
def get_choices(column):
    return sorted([(val, val) for val in df[column].unique()])

class PredictionForm(forms.Form):
    age = forms.IntegerField(min_value=18, max_value=100)
    workclass = forms.ChoiceField(choices=get_choices("workclass"))
    education = forms.ChoiceField(choices=get_choices("education"))
    marital_status = forms.ChoiceField(choices=get_choices("marital_status"))
    occupation = forms.ChoiceField(choices=get_choices("occupation"))
    relationship = forms.ChoiceField(choices=get_choices("relationship"))
    race = forms.ChoiceField(choices=get_choices("race"))
    gender = forms.ChoiceField(choices=get_choices("gender"))  # 'gender' in form, 'sex' in CSV
    capital_gain = forms.IntegerField(min_value=0, max_value=100000)
    capital_loss = forms.IntegerField(min_value=0, max_value=5000)
    hours_per_week = forms.IntegerField(min_value=1, max_value=100)
    native_country = forms.ChoiceField(choices=get_choices("native_country"))
