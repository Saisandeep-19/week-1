import pandas as pd
import streamlit as st
import joblib

# ------------------------------------------------------------------ load model
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

POLLUTANTS = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

# ------------------------------------------------------------------ quality standards
# Indicative limits (assumed to be in mg/L). Adjust them to the standard you want to follow
# (e.g. WHO / BIS IS 10500). 'ideal' is the value of pure water (0, except dissolved oxygen).
STANDARDS = {
    'O2':  {'name': 'Dissolved oxygen', 'limit': 5.0,   'ideal': 14.6, 'rule': 'minimum'},
    'NO3': {'name': 'Nitrate',          'limit': 45.0,  'ideal': 0.0,  'rule': 'maximum'},
    'NO2': {'name': 'Nitrite',          'limit': 3.0,   'ideal': 0.0,  'rule': 'maximum'},
    'SO4': {'name': 'Sulphate',         'limit': 200.0, 'ideal': 0.0,  'rule': 'maximum'},
    'PO4': {'name': 'Phosphate',        'limit': 0.5,   'ideal': 0.0,  'rule': 'maximum'},
    'CL':  {'name': 'Chloride',         'limit': 250.0, 'ideal': 0.0,  'rule': 'maximum'},
}

# Water Quality Index (WQI) scale
WQI_SCALE = [
    (50,  'Excellent',            'success'),
    (100, 'Good',                 'success'),
    (200, 'Poor',                 'warning'),
    (300, 'Very poor',            'error'),
    (float('inf'), 'Unsuitable',  'error'),
]
WQI_RANGES = ['0 - 50', '51 - 100', '101 - 200', '201 - 300', 'Above 300']


def sub_index(p, value):
    s = STANDARDS[p]
    q = 100 * (value - s['ideal']) / (s['limit'] - s['ideal'])
    return max(q, 0.0)


def calculate_wqi(values):
    """Weighted arithmetic WQI: sum(Qi * Wi) / sum(Wi), with Wi = 1 / limit."""
    num, den = 0.0, 0.0
    for p, v in values.items():
        w = 1 / STANDARDS[p]['limit']
        num += w * sub_index(p, v)
        den += w
    return num / den


def wqi_category(score):
    for upper, label, level in WQI_SCALE:
        if score <= upper:
            return label, level


def pollutant_status(p, v):
    lim = STANDARDS[p]['limit']
    if STANDARDS[p]['rule'] == 'minimum':
        return 'Safe' if v >= lim else ('Caution' if v >= 0.6 * lim else 'Unsafe')
    return 'Safe' if v <= lim else ('Caution' if v <= 2 * lim else 'Unsafe')


ICON = {'Safe': '🟢 Safe', 'Caution': '🟡 Caution', 'Unsafe': '🔴 Unsafe'}

# ------------------------------------------------------------------ user interface
st.title("Water Pollutants Predictor")
st.write("Predict the water pollutants and the overall water quality based on year and station ID.")

year_ip = st.number_input("Enter year", min_value=2000, max_value=2100, value=2022)
station_id = st.text_input("Enter Station ID", value='1')

if st.button('Predict'):
    if not station_id:
        st.write("Enter the station ID")
    else:
        input_df = pd.DataFrame({'year': [year_ip], 'id': [station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        predicted = model.predict(input_encoded)[0]
        values = {p: max(float(v), 0.0) for p, v in zip(POLLUTANTS, predicted)}

        st.subheader(f"Predicted pollutant levels for station '{station_id}' in {year_ip}")
        table = pd.DataFrame({
            'Pollutant': [f"{p} ({STANDARDS[p]['name']})" for p in POLLUTANTS],
            'Predicted value': [round(values[p], 2) for p in POLLUTANTS],
            'Reference limit': [
                f"{'at least' if STANDARDS[p]['rule'] == 'minimum' else 'up to'} {STANDARDS[p]['limit']}"
                for p in POLLUTANTS],
            'Status': [ICON[pollutant_status(p, values[p])] for p in POLLUTANTS],
        })
        st.table(table.set_index('Pollutant'))

        score = calculate_wqi(values)
        label, level = wqi_category(score)
        st.subheader("Overall water quality")
        message = f"Water Quality Index (WQI): {score:.1f}  \u2192  {label}"
        getattr(st, level)(message)

        st.markdown("**WQI scale**")
        st.table(pd.DataFrame({
            'WQI': WQI_RANGES,
            'Quality': [x[1] for x in WQI_SCALE],
        }).set_index('WQI'))

        st.caption(
            "Values are model predictions, not lab measurements. Reference limits are indicative and "
            "assumed to be in mg/L; the model is less reliable for O2 and NO2.")