import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm

st.set_page_config(
    page_title=":green[Analisis] Data Penyewaan Sepeda Harian :bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded")

# Membaca data yang sudah dibersihkan
url = 'https://raw.githubusercontent.com/feverlash/Analisis-Data/1b784470df03a8e53000ea6a7393bf2fbafc7919/Bike-sharing-dataset/cleaned_day.csv'
data = pd.read_csv(url)

#Fungsi 1
def visualize_season_counts(data):
    data['season'] = data['season'].map({
        'Spring': 'Spring',
        'Summer': 'Summer',
        'Fall': 'Fall',
        'Winter': 'Winter'
    })

    season_counts = data.groupby('season')['total_rental'].sum().reset_index()

    color_map = {'Spring': '#4CAF50', 
                 'Summer': '#FFC107',  
                 'Fall': '#FF5722', 
                 'Winter': '#2196F3'}

    fig = px.bar(season_counts, x='season', y='total_rental', 
                 title='Total Jumlah Penyewaan Sepeda Berdasarkan Musim',
                 labels={'season': 'Musim', 'total_rental': 'Total Penyewaan Sepeda'},
                 color='season', color_discrete_map=color_map)

    fig.update_xaxes(title_text='Musim')
    fig.update_yaxes(title_text='Total Penyewaan Sepeda')
    st.plotly_chart(fig)

#Fungsi 2
def visualize_year_tren(data):
    data['month'] = pd.Categorical(data['month'], categories=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], ordered=True)
    
    monthly_counts = data.groupby(by=["month","year"]).agg({"total_rental": "sum"}).reset_index()

    fig = px.line(title="Jumlah sepeda yang disewakan berdasarkan Bulan dan Tahun",
                  labels={"month": "Bulan", "total_rental": "Total Penyewaan Sepeda", "year": "Tahun"},
                  markers=False)

    fig.add_scatter(x=monthly_counts[monthly_counts['year']==0]['month'], 
                        y=monthly_counts[monthly_counts['year']==0]['total_rental'],
                        mode='lines', name='2011', marker=dict(color='blue'))
    
    fig.add_scatter(x=monthly_counts[monthly_counts['year']==1]['month'], 
                        y=monthly_counts[monthly_counts['year']==1]['total_rental'],
                        mode='lines', name='2012', marker=dict(color='green'))

    fig.update_layout(xaxis_title=None, yaxis_title=None, legend_title="Tahun",
                      width=800, height=500)
    
    st.plotly_chart(fig)



#Fungsi 3
def visualize_weather_counts(data):
    # Menghitung jumlah penyewaan sepeda berdasarkan workingday
    bike_rentals_per_weather = data.groupby('weather_cond')['total_rental'].sum().reset_index()
    bike_rentals_per_weather.columns = ['Weather Situation', 'Total Rentals']
    
    fig = px.bar(bike_rentals_per_weather, x='Weather Situation', y='Total Rentals', 
                 title='Jumlah Penyewaan Sepeda Berdasarkan Kondisi Cuaca', 
                 labels={'Weather Situation': 'Kondisi Cuaca', 'Jumlah Sewa Harian': 'Jumlah Penyewaan Sepeda'},
                 color='Weather Situation',
                 color_discrete_sequence=px.colors.qualitative.Plotly)  # Specify discrete colors
    
    fig.update_xaxes(title_text='Kondisi Cuaca')
    fig.update_yaxes(title_text='Jumlah Penyewaan Sepeda')
    fig.update_xaxes(tickvals=[1, 2, 3], ticktext=['1', '2', '3'])
    st.plotly_chart(fig)

#Fungsi 4
def visualize_workingday_counts(data):
    workingday_counts = data.groupby('workingday')['total_rental'].sum().reset_index()
    workingday_counts.columns = ['Working Day', 'Counts']
    workingday_counts['Working Day'] = workingday_counts['Working Day'].map({0: 'Hari Libur', 1: 'Hari Kerja'})
    
    fig = px.bar(workingday_counts, x='Working Day', y='Counts', 
                 title='Jumlah Penyewaan Sepeda Berdasarkan Hari Kerja dan Hari Libur', 
                 labels={'Working Day': 'Hari', 'Counts': 'Jumlah Penyewaan Sepeda'},
                 color='Working Day', color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_xaxes(title_text='Hari Kerja / Hari Libur')
    fig.update_yaxes(title_text='Jumlah Penyewaan Sepeda')
    st.plotly_chart(fig)

#Fungsi 5

def visualize_timeseries(data):
    # Set 'dateday' as the index of the DataFrame
    data['dateday'] = pd.to_datetime(data['dateday'])
    data.set_index('dateday', inplace=True)

    y = data['total_rental']

    # Decompose the time series
    decomposition = sm.tsa.seasonal_decompose(y, model='additive')
    
    # Display the plots using Streamlit based on checkbox selection
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    if not (show_original or show_trend or show_seasonal or show_residual):
        st.write(":heavy_exclamation_mark: Silahkan pilih salah satu checkbox untuk menampilkan grafik :heavy_exclamation_mark:")
    else:
        if show_original:
            original_trace = px.line(x=y.index, y=y, title='Original', labels={'x': 'Date', 'y': 'Total Rental'}, color_discrete_sequence=[color_palette[0]])
            st.plotly_chart(original_trace)

        if show_trend:
            trend_trace = px.line(x=y.index, y=decomposition.trend, title='Trend', labels={'x': 'Date', 'y': 'Trend'}, color_discrete_sequence=[color_palette[1]])
            st.plotly_chart(trend_trace)

        if show_seasonal:
            seasonal_trace = px.line(x=y.index, y=decomposition.seasonal, title='Seasonal', labels={'x': 'Date', 'y': 'Seasonal'}, color_discrete_sequence=[color_palette[2]])
            st.plotly_chart(seasonal_trace)

        if show_residual:
            residual_trace = px.line(x=y.index, y=decomposition.resid, title='Residual', labels={'x': 'Date', 'y': 'Residual'}, color_discrete_sequence=[color_palette[3]])
            st.plotly_chart(residual_trace)

# Sidebar with options
st.sidebar.title(':computer: :thinking_face: :question:')
st.sidebar.title('**Pilihan Pertanyaan Bisnis**')
st.sidebar.markdown("**Pada proyek ini saya menganalisis dataset sewa sepeda harian untuk menjawab beberapa pertanyaan di bawah ini**")
question = st.sidebar.selectbox('Pilih Pertanyaan:', ['Pertanyaan 1', 'Pertanyaan 2', 'Pertanyaan 3', 'Pertanyaan 4', 'Analisis Lanjutan'])

# Display the selected question and visualization based on selected question
st.title(':green[Analisis] Data Penyewaan Sepeda Harian :bar_chart:')

if question == 'Pertanyaan 1':
    st.write("### Bagaimana perbedaan pola penyewaan sepeda pada setiap musim?")
    visualize_season_counts(data)
elif question == 'Pertanyaan 2':
    st.write("### Bagaimana tren jumlah total penyewaan sepeda dari bulan ke bulan selama periode 2011 dan 2012?")
    visualize_year_tren(data)
elif question == 'Pertanyaan 3':
    st.write("### Bagaimana perbedaan pola penyewaan sepeda antara setiap cuaca?")
    visualize_weather_counts(data)
elif question == 'Pertanyaan 4':
    st.write("### Bagaimana perbedaan pola penyewaan sepeda antara hari kerja dan hari libur?")
    visualize_workingday_counts(data)
elif question == 'Analisis Lanjutan':
    st.write("### Analisis lanjutan time series analysis")
    show_original = st.sidebar.checkbox('Show Original', value=True)
    show_trend = st.sidebar.checkbox('Show Trend', value=True)
    show_seasonal = st.sidebar.checkbox('Show Seasonal', value=True)
    show_residual = st.sidebar.checkbox('Show Residual', value=True)
    visualize_timeseries(data)
