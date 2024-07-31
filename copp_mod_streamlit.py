import streamlit as st
import pickle
import numpy as np
import sklearn
from streamlit_option_menu import option_menu


#streamlit 
st.set_page_config(page_title= "Industrial Copper Modeling",
                   page_icon="🧊",
                   layout="wide",
                   initial_sidebar_state="expanded",
                   menu_items = None)

st.markdown(f'<h1 style="text-align: center;">INDUSTRIAL COPPER MODELING</h1>',unsafe_allow_html=True)


# Functions
def predict_status(ctry,itmtp,aplcn,wth,prdrf,qtlg,cstlg,tknslg,slgplg,itmdt,itmmn,itmyr,deldtdy,deldtmn,deldtyr):

    #change the datatypes "string" to "int"
    itdd= int(itmdt)
    itdm= int(itmmn)
    itdy= int(itmyr)

    dydd= int(deldtdy)
    dydm= int(deldtmn)
    dydy= int(deldtyr)
    #modelfile of the classification
    with open("B:\Project\.venv\classification_model.pkl","rb") as f:
        model_class=pickle.load(f)

    user_data= np.array([[ctry,itmtp,aplcn,wth,prdrf,qtlg,cstlg,tknslg,
                       slgplg,itdd,itdm,itdy,dydd,dydm,dydy]])
    
    y_pred= model_class.predict(user_data)

    if y_pred == 1:
            return 1
    else:
            return 0


def predict_selling_price(ctry,sts,itmtp,aplcn,wth,prdrf,qtlg,cstlg,
                   tknslg,itmdt,itmmn,itmyr,deldtdy,deldtmn,deldtyr):

    #change the datatypes "string" to "int"
    itdd= int(itmdt)
    itdm= int(itmmn)
    itdy= int(itmyr)

    dydd= int(deldtdy)
    dydm= int(deldtmn)
    dydy= int(deldtyr)
    #modelfile of the classification
    with open("B:\Project\.venv\regression_model.pkl","rb") as f:
        model_regg=pickle.load(f)

    user_data= np.array([[ctry,sts,itmtp,aplcn,wth,prdrf,qtlg,cstlg,tknslg,
                       itdd,itdm,itdy,dydd,dydm,dydy]])
    
    y_pred= model_regg.predict(user_data)

    ac_y_pred= np.exp(y_pred[0])

    return ac_y_pred



with st.sidebar:
    option = option_menu('Santhi', options=["PREDICT SELLING PRICE", "PREDICT STATUS"])
try:
        
    if option == "PREDICT STATUS":

        st.header("PREDICT STATUS (Won / Lose)")
        st.write(" ")

        col1,col2= st.columns(2)

        with col1:
                ccountry= st.number_input(label="**Enter the Value for COUNTRY**/ Min:25.0, Max:113.0")
                citem_type= st.number_input(label="**Enter the Value for ITEM TYPE**/ Min:0.0, Max:6.0")
                capplication= st.number_input(label="**Enter the Value for APPLICATION**/ Min:2.0, Max:87.5")
                cwidth= st.number_input(label="**Enter the Value for WIDTH**/ Min:700.0, Max:1980.0")
                product_ref= st.number_input(label="**Enter the Value for PRODUCT_REF**/ Min:611728, Max:1722207579")
                cquantity_tons= st.number_input(label="**Enter the Value for QUANTITY_TONS**/ Min: 0.00001 and Max: 1000000000.0",format="%0.15f")
                ccustomer= st.number_input(label="**Enter the Value for CUSTOMER ID**")
                cthickness= st.number_input(label="**Enter the Value for THICKNESS**/ Min: 0.18 and Max: 400.0",format="%0.15f")
            
        with col2:
                cselling_price= st.number_input(label="**Enter the Value for SELLING PRICE**/ Min:0.1, Max:100001015.0",format="%0.15f")
                item_date_day= st.selectbox("**Select the Day for ITEM DATE**",("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"))
                item_date_month= st.selectbox("**Select the Month for ITEM DATE**",("1","2","3","4","5","6","7","8","9","10","11","12"))
                item_date_year= st.selectbox("**Select the Year for ITEM DATE**",("2020","2021"))
                delivery_date_day= st.selectbox("**Select the Day for DELIVERY DATE**",("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"))
                delivery_date_month= st.selectbox("**Select the Month for DELIVERY DATE**",("1","2","3","4","5","6","7","8","9","10","11","12"))
                delivery_date_year= st.selectbox("**Select the Year for DELIVERY DATE**",("2020","2021","2022"))
                

        cbutton= st.button(":violet[***PREDICT THE STATUS***]",use_container_width=True)

        if cbutton is not None:
                        if cquantity_tons and cselling_price:
                            with open(r"cmodel.pkl", 'rb') as file:
                                cloaded_model = pickle.load(file)
                            with open(r'cscaler.pkl', 'rb') as f:
                                cscaler_loaded = pickle.load(f)
                            with open(r"ct.pkl", 'rb') as f:
                                ct_loaded = pickle.load(f)
        
        new_sample = np.array(
                                [[np.log(float(cquantity_tons)), np.log(float(cselling_price)), capplication,
                                  np.log(float(cthickness)), float(cwidth), ccountry, int(ccustomer),
                                  int(product_ref), citem_type]])
        new_sample_ohe = ct_loaded.transform(new_sample[:, [8]]).toarray()
        new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6, 7]], new_sample_ohe),
                                    axis=1)
        new_sample = cscaler_loaded.transform(new_sample[:, :12])
        new_pred = cloaded_model.predict(new_sample)
        if new_pred == 1:
            st.write('## :green[The Status is Won] ')
        else:
            st.write('## :red[The status is Lost] ')

except Exception as e:
        st.write()

try: 
    if option == "PREDICT SELLING PRICE":

        st.header("**PREDICT SELLING PRICE**")
        st.write(" ")

        col1,col2= st.columns(2)

    with col1:
            country= st.number_input(label="**Enter the Value for COUNTRY**/ Min:25.0, Max:113.0")
            status= st.number_input(label="**Enter the Value for STATUS**/ Min:0.0, Max:8.0")
            item_type= st.number_input(label="**Enter the Value for ITEM TYPE**/ Min:0.0, Max:6.0")
            application= st.number_input(label="**Enter the Value for APPLICATION**/ Min:2.0, Max:87.5")
            width= st.number_input(label="**Enter the Value for WIDTH**/ Min:700.0, Max:1980.0")
            product_ref= st.number_input(label="**Enter the Value for PRODUCT_REF**/ Min:611728, Max:1722207579")
            quantity_tons= st.number_input(label="**Enter the Value for QUANTITY_TONS**/  Min: 0.00001 and Max: 1000000000.0",format="%0.15f")
            customer= st.number_input(label="**Enter the Value for CUSTOMER ID**")
            
        
    with col2:
            thickness= st.number_input(label="**Enter the Value for THICKNESS (Log Value)**/ Min:-1.7147984280919266, Max:3.281543137578373",format="%0.15f")
            item_date_day= st.selectbox("**Select the Day for ITEM DATE**",("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"))
            item_date_month= st.selectbox("**Select the Month for ITEM DATE**",("1","2","3","4","5","6","7","8","9","10","11","12"))
            item_date_year= st.selectbox("**Select the Year for ITEM DATE**",("2020","2021"))
            delivery_date_day= st.selectbox("**Select the Day for DELIVERY DATE**",("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"))
            delivery_date_month= st.selectbox("**Select the Month for DELIVERY DATE**",("1","2","3","4","5","6","7","8","9","10","11","12"))
            delivery_date_year= st.selectbox("**Select the Year for DELIVERY DATE**",("2020","2021","2022"))
            

    button= st.button(":violet[***PREDICT THE SELLING PRICE***]",use_container_width=True)

    if button is not None:
                    with open(r"model.pkl", 'rb') as file:
                        loaded_model = pickle.load(file)
                    with open(r'scaler.pkl', 'rb') as f:
                        scaler_loaded = pickle.load(f)
                    with open(r"t.pkl", 'rb') as f:
                        t_loaded = pickle.load(f)
                    with open(r"s.pkl", 'rb') as f:
                        s_loaded = pickle.load(f)

    new_sample = np.array([[np.log(float(quantity_tons)), application, np.log(float(thickness)),
                                                    float(width), country, float(customer), int(product_ref), item_type,
                                                    status]])
    new_sample_ohe = t_loaded.transform(new_sample[:, [7]]).toarray()
    new_sample_be = s_loaded.transform(new_sample[:, [8]]).toarray()
    new_sample = np.concatenate(
        (new_sample[:, [0, 1, 2, 3, 4, 5, 6]], new_sample_ohe, new_sample_be),
        axis=1)
    new_sample1 = scaler_loaded.transform(new_sample)
    new_pred = loaded_model.predict(new_sample1)[0]

    # Used np.log earlier to handle data discrepancies, so to get the real output using np.exp
    st.write('## :green[Predicted selling price:] ', np.exp(new_pred))

except Exception as e:
            st.write("Error")