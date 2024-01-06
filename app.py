from datetime import date
import numpy as np
import pandas as pd
import pickle
import psycopg2
import streamlit as st
import plotly.express as px
from streamlit_option_menu import option_menu
from streamlit_extras.add_vertical_space import add_vertical_space
import warnings
warnings.filterwarnings('ignore')



def streamlit_config():

    # page configuration
    st.set_page_config(page_title='Rendal Property', layout="wide")

    # page header transparent color
    page_background_color = """
    <style>

    [data-testid="stHeader"] 
    {
    background: rgba(0,0,0,0);
    }

    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)

    # title and position
    st.markdown(f'<h1 style="text-align: center;">Rental Property Price Prediction</h1>',
                unsafe_allow_html=True)
    add_vertical_space(1)


# custom style for submit button - color and width

def style_submit_button():

    st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                                                        background-color: #367F89;
                                                        color: white;
                                                        width: 70%}
                    </style>
                """, unsafe_allow_html=True)


# custom style for prediction result text - color and position

def style_prediction():

    st.markdown(
        """
            <style>
            .center-text {
                text-align: center;
                color: #20CA0C
            }
            </style>
            """,
        unsafe_allow_html=True
    )



class plotly:

    def pie_chart(df, x, y, title, title_x=0.20):

        fig = px.pie(df, names=x, values=y, hole=0.5, title=title)

        fig.update_layout(title_x=title_x, title_font_size=22)

        fig.update_traces(text=df[y], textinfo='percent+value',
                          textposition='outside',
                          textfont=dict(color='white'),
                          outsidetextfont=dict(size=14))

        st.plotly_chart(fig, use_container_width=True)


    def vertical_bar_chart(df, x, y, text, color, title, title_x=0.35):

        fig = px.bar(df, x=x, y=y, labels={x: '', y: ''}, title=title)

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        fig.update_layout(title_x=title_x, title_font_size=22)

        df[y] = df[y].astype(float)
        text_position = ['inside' if val >= max(
            df[y]) * 0.90 else 'outside' for val in df[y]]

        fig.update_traces(marker_color=color,
                          text=df[text],
                          textposition=text_position,
                          texttemplate='%{y}',
                          textfont=dict(size=14),
                          insidetextfont=dict(color='white'),
                          textangle=0,
                          hovertemplate='%{x}<br>%{y}')

        st.plotly_chart(fig, use_container_width=True, height=100)


    def scatter_chart(df, x, y, size, title):

        fig = px.scatter(data_frame=df, x=x, y=y, size=size, color=y, 
                         labels={x: '', y: ''}, title=title)
        
        fig.update_layout(title_x=0.4, title_font_size=22)
        
        fig.update_traces(hovertemplate=f"{x} = %{{x}}<br>{y} = %{{y}}")
        
        st.plotly_chart(fig, use_container_width=True, height=100)


    def line_chart(df, x, y, text, textposition, color, title, title_x=0.25):

        fig = px.line(df, x=x, y=y, labels={
                      x: '', y: ''}, title=title, text=df[text])

        fig.update_layout(title_x=title_x, title_font_size=22)

        fig.update_traces(line=dict(color=color, width=3.5),
                          marker=dict(symbol='diamond', size=10),
                          texttemplate='%{text}',
                          textfont=dict(size=13.5),
                          textposition=textposition,
                          hovertemplate='%{x}<br>%{y}')

        st.plotly_chart(fig, use_container_width=True, height=100)


    
class sql:

    def create_table():

        try:

            gopi = psycopg2.connect(host='localhost',
                                    user='postgres',
                                    password='root',
                                    database='rental_property')
            cursor = gopi.cursor()

            cursor.execute(f'''create table if not exists rent(
                                        activation_day      int,
                                        activation_month    int,
                                        activation_year     int,
                                        locality            varchar(255),
                                        latitude            float,
                                        longitude           float,
                                        type                varchar(255),
                                        lease_type          varchar(255),
                                        property_size       int,
                                        property_age        float,
                                        furnishing          varchar(255),
                                        facing              varchar(255),
                                        floor               float,
                                        total_floor         float,
                                        building_type       varchar(255),
                                        water_supply        varchar(255),
                                        negotiable          int,
                                        cup_board           float,
                                        balconies           float,
                                        parking             varchar(255),
                                        bathroom            float,
                                        gym                 int,
                                        lift                int,
                                        swimming_pool       int,
                                        internet            int,
                                        ac                  int,
                                        club                float,
                                        intercom            int,
                                        cpa                 float,
                                        fs                  int,
                                        servant             float,
                                        security            int,
                                        sc                  int,
                                        gp                  float,
                                        park                int,
                                        rwh                 float,
                                        stp                 float,
                                        hk                  int,
                                        pb                  int,
                                        vp                  float,
                                        no_of_amenities     float,
                                        rent                float
                            );''')

            gopi.commit()
            cursor.close()
            gopi.close()
        
        except:
            
            st.warning("There is no database named 'rent_prediction'. Please create the database.")


    def drop_table():

        try:

            gopi = psycopg2.connect(host='localhost',
                                    user='postgres',
                                    password='root',
                                    database='rental_property')
            cursor = gopi.cursor()

            cursor.execute(f'''drop table if exists rent;''')

            gopi.commit()
            cursor.close()
            gopi.close()

        except:
            pass
    

    def data_migration():
        
        try:
            df = pd.read_csv(r'Dataset\df_before_encode.csv')
            # df = pd.DataFrame(f)

            gopi = psycopg2.connect(host='localhost',
                                    user='postgres',
                                    password='root',
                                    database='rental_property')
            cursor = gopi.cursor()

            cursor.executemany(f'''insert into rent(activation_day, activation_month, activation_year, locality,
                                                    latitude, longitude, type, lease_type, property_size,
                                                    property_age, furnishing, facing, floor, total_floor,
                                                    building_type, water_supply, negotiable, cup_board, balconies,
                                                    parking, bathroom, gym, lift, swimming_pool, internet, ac,
                                                    club, intercom, cpa, fs, servant, security, sc, gp,
                                                    park, rwh, stp, hk, pb, vp, no_of_amenities, rent) 
                                    values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                                           %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);''', df.values.tolist())
            gopi.commit()
            cursor.close()
            gopi.close()
            st.success('Successfully Data Migrated to SQL Database')
            st.balloons()

        except Exception as e:
            st.warning(e)



class analysis:

    def type():

        gopi = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='root',
                                database='rental_property')
        cursor = gopi.cursor()

        cursor.execute(f'''select distinct type, avg(rent) as average 
                           from rent
                           group by type
                           order by average desc;''')

        s = cursor.fetchall()

        i = [i for i in range(1, len(s)+1)]
        df = pd.DataFrame(s, columns=['type', 'average'], index=i)
        df['average'] = df['average'].apply(lambda x: int(x))

        gopi.commit()
        cursor.close()
        gopi.close()

        return df


    def lease_type():

        gopi = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='root',
                                database='rental_property')
        cursor = gopi.cursor()

        cursor.execute(f'''select distinct lease_type, avg(rent) as average 
                           from rent
                           group by lease_type
                           order by average desc;''')

        s = cursor.fetchall()

        i = [i for i in range(1, len(s)+1)]
        df = pd.DataFrame(s, columns=['lease_type', 'average'], index=i)
        df['average'] = df['average'].apply(lambda x: int(x))

        gopi.commit()
        cursor.close()
        gopi.close()

        return df


    def property_size():

        gopi = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='root',
                                database='rental_property')
        cursor = gopi.cursor()

        cursor.execute(f'''select distinct property_size, avg(rent) as average
                           from rent 
                           group by property_size
                           order by average desc;''')

        s = cursor.fetchall()

        i = [i for i in range(1, len(s)+1)]
        df = pd.DataFrame(s, columns=['property_size', 'average'], index=i)
        df['average'] = df['average'].apply(lambda x: int(x))

        gopi.commit()
        cursor.close()
        gopi.close()

        return df


    # create 10 bins based on range values [like 20-40, 40-60, etc.,]

    def bins(df, feature, bins=50):
        
        # filter 2 columns ---> like temperature and weekly_sales
        df1 = df[['average',feature]]

        # Calculate bin edges
        bin_edges = pd.cut(df1[feature], bins=bins, labels=False, retbins=True)[1]
        bin_edges[0] = 0

        # Create labels for the bins
        bin_labels = [f"{f'{bin_edges[i]:.0f}'} to <br>{f'{bin_edges[i+1]:.0f}'}" for i in range(0, len(bin_edges)-1)]

        # Create a new column by splitting into 10 bins
        df1['part'] = pd.cut(df1[feature], bins=bin_edges, labels=bin_labels, include_lowest=True)

        # group unique values and sum weekly_sales
        df2 = df1.groupby('part')['average'].mean().reset_index()

        # only select weekly sales greater than zero (less than zero bins automatically removed and it can't show barchart)
        df2 = df2[df2['average']>0]

        df2['average'] = df2['average'].apply(lambda x: int(x))

        return df2
            

    def property_age():

        gopi = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='root',
                                database='rental_property')
        cursor = gopi.cursor()

        cursor.execute(f'''select distinct property_age, avg(rent) as average
                           from rent 
                           group by property_age
                           order by average desc;''')

        s = cursor.fetchall()

        i = [i for i in range(1, len(s)+1)]
        df = pd.DataFrame(s, columns=['property_age', 'average'], index=i)
        df['average'] = df['average'].apply(lambda x: int(x))

        gopi.commit()
        cursor.close()
        gopi.close()

        return df


    def furnishing():

        gopi = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='root',
                                database='rental_property')
        cursor = gopi.cursor()

        cursor.execute(f'''select distinct furnishing, avg(rent) as average
                           from rent 
                           group by furnishing
                           order by average desc;''')

        s = cursor.fetchall()

        i = [i for i in range(1, len(s)+1)]
        df = pd.DataFrame(s, columns=['furnishing', 'average'], index=i)
        df['average'] = df['average'].apply(lambda x: int(x))

        gopi.commit()
        cursor.close()
        gopi.close()

        return df


    def facing():

        gopi = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='root',
                                database='rental_property')
        cursor = gopi.cursor()

        cursor.execute(f'''select distinct facing, avg(rent) as average
                           from rent 
                           group by facing
                           order by average desc;''')

        s = cursor.fetchall()

        i = [i for i in range(1, len(s)+1)]
        df = pd.DataFrame(s, columns=['facing', 'average'], index=i)
        df['average'] = df['average'].apply(lambda x: int(x))

        gopi.commit()
        cursor.close()
        gopi.close()

        return df


    def floor():

        gopi = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='root',
                                database='rental_property')
        cursor = gopi.cursor()

        cursor.execute(f'''select distinct floor, avg(rent) as average
                           from rent 
                           group by floor
                           order by average desc;''')

        s = cursor.fetchall()

        i = [i for i in range(1, len(s)+1)]
        df = pd.DataFrame(s, columns=['floor', 'average'], index=i)
        df['average'] = df['average'].apply(lambda x: int(x))

        gopi.commit()
        cursor.close()
        gopi.close()

        return df


    def total_floor():

        gopi = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='root',
                                database='rental_property')
        cursor = gopi.cursor()

        cursor.execute(f'''select distinct total_floor, avg(rent) as average
                           from rent 
                           group by total_floor
                           order by average desc;''')

        s = cursor.fetchall()

        i = [i for i in range(1, len(s)+1)]
        df = pd.DataFrame(s, columns=['total_floor', 'average'], index=i)
        df['average'] = df['average'].apply(lambda x: int(x))

        gopi.commit()
        cursor.close()
        gopi.close()

        return df


    def building_type():

        gopi = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='root',
                                database='rental_property')
        cursor = gopi.cursor()

        cursor.execute(f'''select distinct building_type, avg(rent) as average
                           from rent 
                           group by building_type
                           order by average desc;''')

        s = cursor.fetchall()

        i = [i for i in range(1, len(s)+1)]
        df = pd.DataFrame(s, columns=['building_type', 'average'], index=i)
        df['average'] = df['average'].apply(lambda x: int(x))

        gopi.commit()
        cursor.close()
        gopi.close()

        return df


    def water_supply():

        gopi = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='root',
                                database='rental_property')
        cursor = gopi.cursor()

        cursor.execute(f'''select distinct water_supply, avg(rent) as average
                           from rent 
                           group by water_supply
                           order by average desc;''')

        s = cursor.fetchall()

        i = [i for i in range(1, len(s)+1)]
        df = pd.DataFrame(s, columns=['water_supply', 'average'], index=i)
        df['average'] = df['average'].apply(lambda x: int(x))

        gopi.commit()
        cursor.close()
        gopi.close()

        return df


    def negotiable():

        gopi = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='root',
                                database='rental_property')
        cursor = gopi.cursor()

        cursor.execute(f'''select distinct negotiable, avg(rent) as average
                           from rent 
                           group by negotiable
                           order by average desc;''')

        s = cursor.fetchall()

        i = [i for i in range(1, len(s)+1)]
        df = pd.DataFrame(s, columns=['negotiable', 'average'], index=i)
        df['average'] = df['average'].apply(lambda x: int(x))
        df['negotiable'] = df['negotiable'].apply(lambda x: 'Yes' if x==1 else 'No')

        gopi.commit()
        cursor.close()
        gopi.close()

        return df


    def cup_board():

        gopi = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='root',
                                database='rental_property')
        cursor = gopi.cursor()

        cursor.execute(f'''select distinct cup_board, avg(rent) as average
                           from rent 
                           group by cup_board
                           order by average desc;''')

        s = cursor.fetchall()

        i = [i for i in range(1, len(s)+1)]
        df = pd.DataFrame(s, columns=['cup_board', 'average'], index=i)
        df['average'] = df['average'].apply(lambda x: int(x))
        df['cup_board'] = df['cup_board'].apply(lambda x: f'{int(x)}`')

        gopi.commit()
        cursor.close()
        gopi.close()

        return df


    def balconies():

        gopi = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='root',
                                database='rental_property')
        cursor = gopi.cursor()

        cursor.execute(f'''select distinct balconies, avg(rent) as average
                           from rent 
                           group by balconies
                           order by average desc;''')

        s = cursor.fetchall()

        i = [i for i in range(1, len(s)+1)]
        df = pd.DataFrame(s, columns=['balconies', 'average'], index=i)
        df['average'] = df['average'].apply(lambda x: int(x))
        df['balconies'] = df['balconies'].apply(lambda x: f'{int(x)}`')

        gopi.commit()
        cursor.close()
        gopi.close()

        return df


    def parking():

        gopi = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='root',
                                database='rental_property')
        cursor = gopi.cursor()

        cursor.execute(f'''select distinct parking, avg(rent) as average
                           from rent 
                           group by parking
                           order by average desc;''')

        s = cursor.fetchall()

        i = [i for i in range(1, len(s)+1)]
        df = pd.DataFrame(s, columns=['parking', 'average'], index=i)
        df['average'] = df['average'].apply(lambda x: int(x))

        gopi.commit()
        cursor.close()
        gopi.close()

        return df


    def bathroom():

        gopi = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='root',
                                database='rental_property')
        cursor = gopi.cursor()

        cursor.execute(f'''select distinct bathroom, avg(rent) as average
                           from rent 
                           group by bathroom
                           order by average desc;''')

        s = cursor.fetchall()

        i = [i for i in range(1, len(s)+1)]
        df = pd.DataFrame(s, columns=['bathroom', 'average'], index=i)
        df['average'] = df['average'].apply(lambda x: int(x))
        df['bathroom'] = df['bathroom'].apply(lambda x: f'{int(x)}`')

        gopi.commit()
        cursor.close()
        gopi.close()

        return df


    def no_of_amenities():

        gopi = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='root',
                                database='rental_property')
        cursor = gopi.cursor()

        cursor.execute(f'''select distinct no_of_amenities, avg(rent) as average
                           from rent 
                           group by no_of_amenities
                           order by average desc;''')

        s = cursor.fetchall()

        i = [i for i in range(1, len(s)+1)]
        df = pd.DataFrame(s, columns=['no_of_amenities', 'average'], index=i)
        df['average'] = df['average'].apply(lambda x: int(x))
        df['no_of_amenities'] = df['no_of_amenities'].apply(lambda x: f'{int(x)}`')

        gopi.commit()
        cursor.close()
        gopi.close()

        return df


    def amenities(amenity):

        gopi = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='root',
                                database='rental_property')
        cursor = gopi.cursor()

        cursor.execute(f'''select distinct {amenity}, avg(rent) as average
                           from rent 
                           group by {amenity}
                           order by average desc;''')

        s = cursor.fetchall()

        i = [i for i in range(1, len(s)+1)]
        df = pd.DataFrame(s, columns=['amenity', 'average'], index=i)
        df['average'] = df['average'].apply(lambda x: int(x))
        df['amenity'] = df['amenity'].apply(lambda x: 'Yes' if x==1 else 'No')

        gopi.commit()
        cursor.close()
        gopi.close()

        return df



class prediction:

    type_dict          = {'RK1':0, 'BHK1':1, 'BHK2':2, 'BHK3':3, 'BHK4':4, 'BHK4PLUS':5}
    lease_type_dict    = {'BACHELOR':1, 'FAMILY':2, 'COMPANY':3, 'ANYONE':4}
    facing_dict        = {'N':1, 'E':2, 'W':3, 'S':4, 'NE':5, 'NW':6, 'SE':7, 'SW':8}
    furnishing_dict    = {'NOT_FURNISHED':0, 'SEMI_FURNISHED':1, 'FULLY_FURNISHED':2}
    parking_dict       = {'NONE':0, 'TWO_WHEELER':1, 'FOUR_WHEELER':2, 'BOTH':3}
    water_supply_dict  = {'CORPORATION':1, 'CORP_BORE':2, 'BOREWELL':3,}
    building_type_dict = {'AP':1, 'IH':2, 'IF':3, 'GC':4}
    binary_dict        = {'Yes':1, 'No':0}
    amenities_dit      = {'gym':'Gym', 'lift':'Lift', 'swimming_pool':'Swimming Pool', 
                        'internet':'Internet', 'ac':'AC', 'club':'Club', 'intercom':'Intercom', 
                        'cpa':'CPA', 'fs':'FS', 'servant':'Servant', 'security':'Security',
                        'sc':'SC', 'gp':'GP', 'park':'Park', 'rwh':'RWH', 'stp':'STP', 
                        'hk':'HK', 'pb':'PB', 'vp':'VP'}

    type_list          = ['RK1', 'BHK1', 'BHK2', 'BHK3', 'BHK4', 'BHK4PLUS']
    facing_list        = ['N', 'E', 'W', 'S', 'NE', 'NW', 'SE', 'SW']
    lease_type_list    = ['BACHELOR', 'FAMILY', 'COMPANY', 'ANYONE']
    furnishing_list    = ['NOT_FURNISHED', 'SEMI_FURNISHED', 'FULLY_FURNISHED']
    parking_list       = ['NONE', 'TWO_WHEELER', 'FOUR_WHEELER', 'BOTH']
    water_supply_list  = ['CORPORATION', 'CORP_BORE', 'BOREWELL']
    building_type_list = ['AP', 'IH', 'IF', 'GC']  # AP-Apartment, IH-Independent House, IF-Inherited House, GC-Guesthouse/Condo
    amenities_list     = ['gym', 'lift', 'swimming_pool', 'internet', 'ac', 
                          'club', 'intercom', 'cpa', 'fs', 'servant', 'security', 
                          'sc', 'gp', 'park', 'rwh', 'stp', 'hk', 'pb', 'vp']
    


    def feature_list(feature):

            gopi = psycopg2.connect(host='localhost',
                                    user='postgres',
                                    password='root',
                                    database='rental_property')
            cursor = gopi.cursor()

            cursor.execute(f'''select distinct {feature} 
                               from rent;''')

            s = cursor.fetchall()

            data = [int(i[0]) for i in s]
            data.sort(reverse=False)

            cursor.close()
            gopi.close()

            return data


    def predict_rent():

        with st.form('prediction'):

            col1,col2,col3 = st.columns([0.45, 0.1, 0.45])

            with col1:

                activation_date = st.date_input(label='Activation Date', min_value=date(2017,1,1),
                                                max_value=date(2018,12,31), value=date(2017,1,1))

                latitute = st.number_input(label='Latitude', min_value=12.90, max_value=12.99, value=12.90)

                longitude = st.number_input(label='Latitude', min_value=77.50, max_value=80.27, value=77.50)

                type = st.selectbox(label='Property Type', options=prediction.type_list)

                lease_type = st.selectbox(label='Lease Type', options=prediction.lease_type_list)

                property_size = st.number_input(label='Property Size', min_value=1, max_value=50000, value=1000)

                property_age = st.number_input(label='Property Age', min_value=0.0, max_value=400.0, value=0.0)

                furnishing = st.selectbox(label='Furnishing', options=prediction.furnishing_list)

                facing = st.selectbox(label='Facing', options=prediction.facing_list)

                floor = st.selectbox(label='Floor', options=prediction.feature_list('floor'))


            with col3:

                total_floor = st.selectbox(label='Total Floor', options=prediction.feature_list('total_floor'))

                building_type = st.selectbox(label='Building Type', options=prediction.building_type_list)

                water_supply = st.selectbox(label='Water Supply', options=prediction.water_supply_list)

                negotiable = st.selectbox(label='Negotiable', options=['Yes', 'No'])

                cup_board = st.selectbox(label='Cup Board', options=prediction.feature_list('cup_board'))

                balconies = st.selectbox(label='Balconies', options=prediction.feature_list('balconies'))

                parking = st.selectbox(label='Parking', options=prediction.parking_list)

                bathroom = st.selectbox(label='Bathroom', options=prediction.feature_list('bathroom'))

                amenities = st.multiselect(label='Amenities', options=prediction.amenities_list)

                add_vertical_space(2)
                button = st.form_submit_button(label='SUBMIT')
                style_submit_button()


            if button:

                with st.spinner(text='Processing...'):

                    # load the regression pickle model
                    with open(r'model\regression_model.pkl', 'rb') as f:
                        model = pickle.load(f)
                    
                    # amenity encoding based on user input
                    amenity_value = []
                    for i in ['gym', 'lift', 'swimming_pool', 'internet', 'ac', 
                              'club', 'intercom', 'cpa', 'fs', 'servant', 'security', 
                              'sc', 'gp', 'park', 'rwh', 'stp', 'hk', 'pb', 'vp']:

                        if i in amenities:
                            amenity_value.append(1)
                        else:
                            amenity_value.append(0)
                    
                    amenity_value.append(sum(amenity_value))


                    # get user input and combine single list

                    data = [activation_date.day, activation_date.month,
                            activation_date.year, latitute, longitude,
                            prediction.type_dict[type], 
                            prediction.lease_type_dict[lease_type], 
                            property_size, property_age, 
                            prediction.furnishing_dict[furnishing], 
                            prediction.facing_dict[facing], floor, total_floor, 
                            prediction.building_type_dict[building_type], 
                            prediction.water_supply_dict[water_supply], 
                            prediction.binary_dict[negotiable], cup_board, 
                            balconies, prediction.parking_dict[parking], bathroom]
                    
                    data.extend(amenity_value)

                    # make array for all user input values in required order for model prediction
                    user_data = np.array([data])
                    
                    # model predict the selling price based on user input
                    y_pred = model.predict(user_data)[0]
                    
                    # round the value with 2 decimal point (Eg: 1.35678 to 1.36)
                    rent_price = f"{y_pred:.2f}"

                    return rent_price



streamlit_config()


with st.sidebar:

    add_vertical_space(2)
    option = option_menu(menu_title='', options=['Migrating to SQL', 'Data Analysis', 'Prediction', 'Exit'],
                         icons=['database-fill', 'bar-chart-line', 'slash-square', 'sign-turn-right-fill'])
    
    col1, col2, col3 = st.columns([0.35, 0.60, 0.2])

    with col2:
        button = st.button(label='Submit')



if button and option == 'Migrating to SQL':

    col1, col2, col3 = st.columns([0.3, 0.4, 0.3])

    with col2:

        add_vertical_space(2)

        with st.spinner('Dropping the Existing Table...'):
            sql.drop_table()
        
        with st.spinner('Creating Sales Table...'):
            sql.create_table()
        
        with st.spinner('Migrating Data to SQL Database...'):
            sql.data_migration()

        

elif option == 'Data Analysis':

    tab1,tab2,tab3,tab4 = st.tabs(['Type', 'Lease Type', 'Property Size','Property Age'])

    with tab1:
        df = analysis.type()
        plotly.vertical_bar_chart(df=df, x='type', y='average', text='average', color='#5D9A96', 
                                  title='Property Type wise Average Rent')
    
    with tab2:
        df1 = analysis.lease_type()
        plotly.vertical_bar_chart(df=df1, x='lease_type', y='average', text='average', 
                                  color='#5cb85c', title='Lease Type wise Average Rent')

    with tab3:
        df2 = analysis.property_size()
        df3 = analysis.bins(df=df2, feature='property_size')
        plotly.vertical_bar_chart(df=df3, x='part', y='average', text='part', color='#5D9A96',
                                  title='Property Size wise Average Rent')

    with tab4:
        df4 = analysis.property_age()
        df5 = analysis.bins(df=df4, feature='property_age')
        plotly.vertical_bar_chart(df=df5, x='part', y='average', text='part', color='#5cb85c',
                                  title='Property Age wise Average Rent')
    



    tab5,tab6,tab7,tab8,tab9,tab10,tab11 = st.tabs(['Furnishing','Building Type',
                                                    'Water Supply', 'Parking','Negotiable',
                                                    'Amenities','Amenities Types'])

    with tab5:
        df6 = analysis.furnishing()
        plotly.pie_chart(df=df6, x='furnishing', y='average', title_x=0.25, 
                         title='Furnishing wise Average Rent')

    with tab6:
        df12 = analysis.building_type()
        plotly.pie_chart(df=df12, x='building_type', y='average', title_x=0.28,
                         title='Building Type wise Average Rent')

    with tab7:
        df13 = analysis.water_supply()
        plotly.pie_chart(df=df13, x='water_supply', y='average', title_x=0.25,
                         title='Water Supply wise Average Rent')

    with tab8:
        df17 = analysis.parking()
        plotly.pie_chart(df=df17, x='parking', y='average', title_x=0.26,
                         title='Parking wise Average Rent')

    with tab9:
        df14 = analysis.negotiable()
        plotly.pie_chart(df=df14, x='negotiable', y='average', title_x=0.28,
                         title='Negotiable wise Average Rent')

    with tab10:
        df20 = analysis.no_of_amenities()
        df20['no_of_amenities'] = df20['no_of_amenities'].apply(
                                        lambda x: 'Yes' if x!='0`' else 'No')
        df20 = df20.groupby('no_of_amenities').mean().reset_index()
        df20['average'] = df20['average'].apply(lambda x: round(x,0))
        plotly.pie_chart(df=df20, x='no_of_amenities', y='average', title_x=0.28,
                         title='Amenities based Average Rent')
    
    with tab11:   
        options = ['gym','lift','swimming_pool','internet','ac','club',
                   'intercom','cpa','fs','servant','security','sc',
                   'gp','park','rwh','stp','hk','pb','vp']
        
        options_dict = {'gym':'Gym', 'lift':'Lift', 'swimming_pool':'Swimming Pool', 
                        'internet':'Internet', 'ac':'AC', 'club':'Club', 
                        'intercom':'Intercom', 'cpa':'CPA', 'fs':'FS', 
                        'servant':'Servant', 'security':'Security', 'sc':'SC', 
                        'gp':'GP', 'park':'Park', 'rwh':'RWH', 'stp':'STP', 
                        'hk':'HK', 'pb':'PB', 'vp':'VP'}
        
        col1,col2,col3 = st.columns([0.33,0.33,0.33])

        with col1:
            amenities = st.selectbox(label='', options=options)
        df21 = analysis.amenities(amenity=amenities)
        plotly.pie_chart(df=df21, x='amenity', y='average', title_x=0.32,
                         title=f'{options_dict[amenities]} wise Average Rent')
        
         


    tab12,tab13,tab14 = st.tabs(['Amenity Count', 'Cup Board','Bathroom'])

    with tab12:
        df19 = analysis.no_of_amenities()
        plotly.line_chart(df=df19, x='no_of_amenities', y='average', 
                          text='average', textposition='top right', 
                          color='#5cb85c', title_x=0.30,
                          title='Amenity Count wise Average Rent')


    with tab13:
        df15 = analysis.cup_board()
        plotly.line_chart(df=df15, x='cup_board', y='average', text='average', 
                          textposition=['middle right'] + ['top right']*(len(df15['cup_board'])-1), 
                          color='#5D9A96', title_x=0.30,
                          title='Cup Board wise Average Rent')


    with tab14:
        df18 = analysis.bathroom()
        plotly.line_chart(df=df18, x='bathroom', y='average', text='average', 
                          textposition='top right', color='#5cb85c', 
                          title_x=0.30, title='Bathroom wise Average Rent')   

    

    
    tab15,tab16,tab17,tab18 = st.tabs(['Facing', 'Floor','Total Floor','Balconies'])

    with tab15:
        df7 = analysis.facing()
        plotly.vertical_bar_chart(df=df7, x='facing', y='average', text='facing', 
                                  color='#5D9A96', title='Facing wise Average Rent')

    with tab16:
        df8 = analysis.floor()
        df9 = analysis.bins(df=df8, feature='floor', bins=10)
        plotly.vertical_bar_chart(df=df9, x='part', y='average', text='average',
                                  color='#5cb85c', title='Floor wise Average Rent')
    
    with tab17:
        df10 = analysis.total_floor()
        df11 = analysis.bins(df=df10, feature='total_floor', bins=10)
        plotly.vertical_bar_chart(df=df11, x='part', y='average', text='average',
                                  color='#5D9A96', title='Total Floor wise Average Rent')
    
    with tab18:
        df16 = analysis.balconies()
        plotly.vertical_bar_chart(df=df16, x='balconies', y='average', text='average',
                                  color='#5cb85c', title='Balconies wise Average Rent')
        


elif option == 'Prediction':

    rent = prediction.predict_rent()

    if rent:

        # apply custom css style for prediction text
        style_prediction()

        st.markdown(f'### <div class="center-text">Predicted Rent = {rent}</div>', 
                    unsafe_allow_html=True)

        st.balloons()



elif option == 'Exit':
    
    add_vertical_space(2)

    col1,col2,col3 = st.columns([0.20,0.60,0.20])

    with col2:

        st.success('#### Thank you for your time. Exiting the application')
        st.balloons()

