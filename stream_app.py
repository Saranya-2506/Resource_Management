import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import streamlit as st
import pickle 
import re
from nltk.corpus import stopwords
import seaborn as sns
import plotly.express as px
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as pt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go

with st.sidebar:
    selected = option_menu("Main Menu", ["Visualization", "Clustering"], 
        icons=['bar-chart-line', 'share'], menu_icon="cast", default_index=1)
    st.write(selected)

if selected == "Visualization":

        list1=["NIC Name VS Total Workers","TotalWorkers in Agra State","Class Vs Female Workers","Class Vs Male Workers","coorelation ","State Vs Workers","Male Vs Female Workers","Top Ten States Vs Workers"]
        
        options=st.selectbox("select any option",list1)
        
        df=pd.read_csv("/Users/saranya/Documents/Projects/Human_Resorce_Management/resource.csv")

        category_df=df.groupby("NIC Name")['Total_Workers'].sum()

        category_df=category_df.reset_index()

        category_df.columns=['NIC Name','Total_Workers']
    
        topcategory_df=category_df.head(10)
        #st.write(topcategory_df)
        columns=['Total_Workers','Male Workers','Female workers']
        class_df=df.groupby(['Class'])[columns].sum()
        class_df=class_df.head(10)
        class_df=class_df.reset_index()
        class_df['Female workers']=np.log(class_df['Female workers'])
        class_df['Male Workers']=np.log(class_df['Male Workers'])
        class_df['Total_Workers']=np.log(class_df['Total_Workers'])

        if options == "NIC Name VS Total Workers" :
            fig = px.bar(
            topcategory_df,
            x="NIC Name",
            y="Total_Workers",
            color="NIC Name",
            barmode="group",
            title="workers by different department")
            fig.update_layout(
            height=600,   
            width=1000    
            )
            
            
            st.plotly_chart(fig,use_container_width=True)

        elif options == "TotalWorkers in Agra State":

            st.title("Distribution of state Agra")
        
            sum=df.groupby(['India/States','NIC Name'])['Total_Workers'].sum()
        
            sum.head(10)
        
            sum.columns=['State','NIC Name','Total_Workers']

            fig=px.histogram(
            sum,
            x="Total_Workers",
            nbins=10,
            title="Worker distribution"
    ) 
            st.plotly_chart(fig,use_container_width=True)

        elif options == "Class Vs Female Workers":

            x = class_df['Class']
            y = class_df['Female workers']

            fig=px.bar(
                class_df,
                x="Class",
                y="Female workers",
                title="Number of Female workers in each class")
            st.plotly_chart(fig,use_container_width=True)

        elif options == "Class Vs Male Workers":

            df=class_df['Male Workers']
            fig=px.histogram(
            df,
            x="Male Workers",
            nbins=10,
            title="Histogram for Male workers classwise")
            st.plotly_chart(fig,use_container_width=True)
        

        elif options == "coorelation ":
            cols=['Class','Total_Workers','Male Workers','Female workers']
            df= class_df[cols]
            corr_matrix = df.corr(numeric_only=True)

    # Create the heatmap using Plotly Graph Objects
            fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.values,
            y=corr_matrix.index.values,
            colorscale='Viridis',
            colorbar=dict(title='Correlation Coefficient'),
        # Add text labels to the cells
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            hoverinfo="z+x+y"
    ))

            st.plotly_chart(fig,use_container_width=True)

        elif options == "State Vs Workers":
            
            x=df['India/States'].head()
            y=df['Marginal Workers'].head()
            
            fig=px.line(df,x="India/States",
                        y="Marginal Workers",
                        title="State wise workers")
            st.plotly_chart(fig,use_container_width=True)
            
            

        elif options == "Male Vs Female Workers":

            x=df['India/States'].head()
            y=df['Male Workers'].head()
          
           

            fig=px.scatter(df,
                            x="India/States",
                            y="Male Workers",
                            title="Male and female workers",
                            color="Male Workers",
                            size="Male Workers",
                            hover_data="Male Workers")
            st.plotly_chart(fig,use_container_width=True)
            

            
        elif options == "Top Ten States Vs Workers":

            state_worker=df.groupby('India/States')['Total_Workers'].sum()
            state_df = state_worker.reset_index()
            state_df.columns = ['State', 'Total_Workers']
            state_df = state_df[state_df['State'].str.contains('STATE')]
            top10_states = state_df.sort_values(by='Total_Workers', ascending=False).head(10)

            fig=px.pie(top10_states,
                       values="Total_Workers",
                       names="State",
                       hover_data="Total_Workers")
            st.plotly_chart(fig,use_container_width=True)

else:
    stop_words=set(stopwords.words("english"))

def clean_text(text):
    text=text.lower()
    text=re.sub(r"[^a-z\s]"," ",text)
    words=text.split()
    words=[w for w in words if w not in stop_words]
    return " ".join(words)


#Load model

tfidf=pickle.load(open("/Users/saranya/Documents/Projects/Human_Resorce_Management/tfidf.pkl","rb"))
kmeans=pickle.load(open("/Users/saranya/Documents/Projects/Human_Resorce_Management/kmeans.pkl","rb"))

st.title("Text Clustering App")
st.write("Welcome to home page!")
user_input=st.text_area("Enter the Text")

button=st.button("predict")

if button:
    clean=clean_text(user_input)

    vector=tfidf.transform([clean])

    cluster=kmeans.predict(vector)
    st.markdown(f"###predicted Cluster: **{cluster[0]}**")


    if st.button("Go to page 2"):
        st.switch_page("/Users/saranya/Documents/Projects/Human_Resorce_Management/pages/second.py")



