import streamlit as st
from streamlit_option_menu import option_menu
#improting our project pages
#from streamlit_extras.app_logo import add_logo

import About,gender,Areacontrols,Adv,TakingROI,tes,queq, stock,Count,takevid#streamlit run /Users/khawlahd/Desktop/cv/my-venv/Home.py,stock



st.set_page_config (
    page_title="Hyper-Eye",
    )

class MultiApp:
    
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):

        self.apps.append({
            "title": title,
            "function": func
        })
    


    def run():
        # app = st.sidebar(#'Wheelchair user',,'Taking Roi'
        with st.sidebar:        
            app = option_menu(
                menu_title="Hyper-Eye",
                menu_icon="d.jpeg",
                options=['About',    'Count',"Count by Gender",'Self-checkouts', 'Stock out detection','Area controls','Advertising',    'Chat',         'Garbage Notify','VIP Notify','Queuing Notify'],
                icons=['house-fill','people','gender-ambiguous','cash-coin'      ,   'bag-x',           'bullseye'  ,'play-btn-fill',  'chat-dots', 'trash2' ,'suit-heart-fill', 'input-cursor'   ],
                #menu_icon='chat-text-fill',
                default_index=1,
                styles={
                    "container": {"padding": "5!important","background-color":'white'},
        "icon": {"color": "black", "font-size": "23px"}, 
        "nav-link": {"color":"black","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#d0e0e3"},
        "nav-link-selected": {"background-color": "#45818e"},},
               
        
                )

        
        #if app == "Home":
            #Home.app()
        if app == "About": 
            About.app()    
        if app == "Count": 
            takevid.app()  
        if app == "Count by Gender": 
            gender.app()      
        if app == 'Self-checkouts': #Hardware limitation 
            About.app()
        if app == 'Stock out detection':
            stock.app()
        if app == 'Area controls': #
            Areacontrols.app()
        if app == 'Advertising':
            Adv.app()
        
        if app == 'Chat':# Still not integrated 
            tes.app()
        if app == 'Garbage Notify':# Still not integrated 
            About.app()
        if app == 'VIP Notify':# Still not integrated 
            About.app()
        if app == 'Queuing Nofity': 
            About.app()
        #if app== 'Taking Roi':
        #    TakingROI.app()
    run()      