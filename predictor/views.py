from django.shortcuts import render
from .forms import PredictionForm
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from django.conf import settings

# Load model and encoders
model = joblib.load("predictor/ml/model.pkl")
encoders = joblib.load("predictor/ml/encoders.pkl")

def predict_income(request):
    result = None

    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            input_df = pd.DataFrame([data])

            # Apply encoders to categorical columns
            for col in input_df.columns:
                if col in encoders:
                    le = encoders[col]
                    input_df[col] = le.transform(input_df[col])

            prediction = model.predict(input_df)[0]
            result = ">50K" if prediction == 1 else "<=50K"
    else:
        form = PredictionForm()

    return render(request, "predictor/form.html", {'form': form, 'result': result})


def home(request):
    # Load data
    df = pd.read_csv('adult_3.csv')

    # Clean data
    df = df.replace(' ?', pd.NA).dropna()
    df['income'] = df['income'].str.strip()

    # Graphs folder
    img_dir = os.path.join(settings.MEDIA_ROOT, 'graphs')
    os.makedirs(img_dir, exist_ok=True)

    plots = []

    # Define which plots to generate
    columns_to_plot = ['education', 'gender', 'occupation', 'race', 'marital_status']
    for col in columns_to_plot:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=col, hue='income')
        plt.xticks(rotation=45)
        plt.tight_layout()

        img_name = f"{col}_vs_income.png"
        img_path = os.path.join(img_dir, img_name)
        plt.savefig(img_path)
        plt.close()

        plots.append(f"media/graphs/{img_name}")

    return render(request, "predictor/home.html", {"plots": plots})
