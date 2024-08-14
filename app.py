import streamlit as st
import pickle
import numpy as np

pickle_in = open('classifier1', 'rb')
pkl_in = pickle.load(pickle_in)

lr_model = pkl_in[0] 
knn = pkl_in[1]
means = pkl_in[2]
std = pkl_in[3]


@st.cache()


# Define the function which will make the prediction using data
# inputs from users
def prediction(input_list, prediction_probability, model_type ='LR'):
    # Make predictions
    if model_type =='LR':
        prediction = (lr_model.predict_proba([input_list])[::,1]>= prediction_probability ).astype(int)
    elif model_type =='KNN':
        prediction = (knn.predict_proba([input_list])[::,1]>= prediction_probability ).astype(int)
    
    if prediction == 0:
        pred = 'Flight Is On-time'
    else:
        pred = ' Wait! Flight is delayed'
    return pred

def set_carrier(carrier):
    carrier_delta=0
    carrier_us=0
    carrier_envoy=0
    carrier_continental=0
    carrier_discovery=0
    carrier_other=0
    
    if carrier =='delta':
        carrier_delta =1
    elif carrier =='us':
        carrier_us =1
    elif carrier =='envoy':
        carrier_envoy =1
    elif carrier =='continental':
        carrier_continental =1
    elif carrier =='discovery':
        carrier_discovery =1
    else: carrier_other = 1
    
    return carrier_delta,carrier_us,carrier_envoy,carrier_continental,carrier_discovery,carrier_other

def set_origin(origin):
    origin_dca = 0
    origin_iad = 0
    origin_bwi = 0
    
    if origin =='dca':
        origin_dca = 1
    elif origin =='iad':
        origin_iad = 1
    else: origin_bwi = 1
    
    return origin_dca,origin_iad,origin_bwi

def set_dest(dest):
    dest_jfk = 0
    dest_ewr = 0
    dest_lga = 0
    
    if dest =='jfk':
        dest_jfk = 1
    elif dest =='ewr':
        dest_ewr = 1
    else: dest_lga = 1
    
    return dest_jfk,dest_ewr,dest_lga

def select_day_of_week(day_of_week):
    Monday = 0
    Tuesday = 0
    Wednesday = 0
    Thursday = 0
    Friday = 0
    Saturday = 0
    Sunday = 0
    
    if day_of_week =='Monday':
        Monday =1
    elif day_of_week =='Tuesday':
        Tuesday =1
    elif day_of_week =='Wednesday':
        Wednesday =1
    elif day_of_week =='Thursday':
        Thursday =1
    elif day_of_week =='Friday':
        Friday =1
    elif day_of_week =='Saturday':
        Saturday = 1
    elif day_of_week =='Sunday':
        Sunday = 1
        
    return Monday,Tuesday,Wednesday,Thursday,Friday,Saturday,Sunday
        
    

def main():
    
    # Create input fields
    st.markdown("<h1 style='text-align: center; color: red;'>Flight Status Predictor</h1>", unsafe_allow_html=True)
    
    sch_dep_time = st.number_input("Departure Time(in 24 hour format)",
                                  min_value=0,
                                  max_value=20,
                                  value=2,
                                  step=1,
                                 )
    carrier = st.selectbox("Select Airline Carrier",['delta','us','envoy','continental','discovery','other'])
    carrier_delta,carrier_us,carrier_envoy,carrier_continental,carrier_discovery,carrier_other = set_carrier(carrier)
    
    origin = st.selectbox("Select origin",['dca', 'iad', 'bwi'])
    origin_dca,origin_iad,origin_bwi = set_origin(origin)
    
    dest = st.selectbox("Select Destination",['jfk', 'ewr', 'lga'])
    dest_jfk,dest_ewr,dest_lga = set_dest(dest)
                                              
    
    day_of_week = st.selectbox("Select Day of the week",['Monday','Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    Monday,Tuesday,Wednesday,Thursday,Friday,Saturday,Sunday = select_day_of_week(day_of_week)
    
                                              
    bad_weather = st.number_input("Select if weather is bad or not (0-1)",
                              min_value=0,
                              max_value=1,
                              value=1,
                              step=1
                             )
    
    distance = st.number_input("Distance in Miles",
                          min_value=0,
                              max_value=100000,
                              value=229,
                              step=100
                         )
                                              
    
    model_type = st.sidebar.selectbox("Model type", ['KNN','LR'])
    prediction_probability = [st.sidebar.slider("Probablity Threshold",0.0, 1.0, 0.3, 0.01)]
    
    #scaling data with test stats used for training
    
    
    input_data = [sch_dep_time,carrier_delta,carrier_us,
       carrier_envoy,carrier_continental,carrier_discovery,
       carrier_other,dest_jfk,dest_ewr,dest_lga,distance,
       origin_dca,origin_iad,origin_bwi,bad_weather,Monday,
       Tuesday,Wednesday,Thursday,Friday,Saturday,Sunday]
    
    input_list1 = np.divide(np.subtract(input_data,means),std).tolist()
    
    result = ""
#     with st.sidebar:
#         result = prediction(sch_dep_time,carrier_delta,carrier_us,
#        carrier_envoy,carrier_continental,carrier_discovery,
#        carrier_other,dest_jfk,dest_ewr,dest_lga,distance,
#        origin_dca,origin_iad,origin_bwi,bad_weather,Monday,
#        Tuesday,Wednesday,Thursday,Friday,Saturday,Sunday, prediction_probability, model_type)
#         st.success(result)
        

    
    # When 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        result = prediction(input_list1, prediction_probability, model_type)
        if result =='Flight Is On-time':
            st.balloons()
        st.success(result)
        
    if st.button('Show ROC curve'):
        if model_type =='KNN':
            st.image('KNN_ROC.png',caption='KNN ROC image')
        elif model_type =='LR':
            st.image('LR_ROC.png',caption='LR ROC image')
       
if __name__=='__main__':
    main()
