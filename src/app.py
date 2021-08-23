import streamlit as st
import joblib
import config

model = joblib.load(config.TRAINED_MODEL)

@st.cache()

def prediction(Popularity, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness,
liveness, valence, tempo, duration, time_signature, new4, new5, new6, new7, new8, new9, new10):

    prediction = model.predict([[Popularity, danceability, energy, key, loudness, mode, speechiness, acousticness,
     instrumentalness,liveness, valence, tempo, duration, time_signature, new4, new5, new6, new7, 
     new8, new9, new10]])

    if prediction == 0:
        pred='Acoustic/Folk_0'
    else:
        pred='Other music'
    
    return pred

def main():
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Music Genre Prediction ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 

    Popularity = st.number_input("Popularity")  
    danceability = st.number_input("danceability")  
    energy = st.number_input("energy")  
    key = st.number_input("key") 
    loudness = st.number_input("loudness")  
    mode = st.number_input("mode")  
    speechiness = st.number_input("speechiness")  
    acousticness = st.number_input("acousticness")  
    instrumentalness = st.number_input("instrumentalness") 
    liveness = st.number_input("liveness")
    valence = st.number_input("valence") 
    tempo = st.number_input("tempo")  
    duration = st.number_input("duration") 
    time_signature = st.number_input("time signature")  
    new4 = st.number_input("new4")  
    new5 = st.number_input("new5")  
    new6 = st.number_input("new6")  
    new7 = st.number_input("new7")  
    new8 = st.number_input("new8")  
    new9 = st.number_input("new9")  
    new10 = st.number_input("new10") 
    result =""

    if st.button("Predict"): 
        result = prediction(Popularity, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness,
        liveness, valence, tempo, duration, time_signature, new4, new5, new6, new7, new8, new9, new10) 
        st.success('Your music genre is {}'.format(result))

if __name__=='__main__': 
    main()
