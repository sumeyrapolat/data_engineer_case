# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 22:45:25 2024

@author: sumey
"""

import requests
from pydantic import BaseModel, ValidationError, condecimal
from datetime import datetime
import numpy as np
import time
from influxdb_client import InfluxDBClient, Point, WritePrecision
import pytz
import pandas as pd
from sklearn.impute import KNNImputer

# KNN Imputer'ı tanımla
imputer = KNNImputer(n_neighbors=5)

api_key = "8c34616e78bcffbe4ce33f1bee817b61"
# InfluxDB bağlantısı için ayarlar
token = "akG6YxrQ0Ja2Rgj4LDVl5A_ggiYYe2o3reFpp0-udvS6oKNYzvTnUvfAfDaEbjJIluSWIc2FrIsep0NHGhFM3g=="
org = "test"
bucket = "weather_data"  # Daha önce oluşturduğunuz bucket adı

client = InfluxDBClient(url="http://localhost:8086", token=token)
write_api = client.write_api()

# Define Pydantic models
class WeatherMain(BaseModel):
    temp: condecimal(gt=-273.15, lt=1e5)
    humidity: condecimal(ge=0, le=100)

class WeatherWind(BaseModel):
    speed: condecimal(ge=0, le=200)
    deg: condecimal(ge=0, le=360)

class WeatherData(BaseModel):
    name: str
    main: WeatherMain
    wind: WeatherWind

def fetch_weather_data(lat, lon, api_key):
    url = f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}'
    response = requests.get(url)
    data = response.json()
    
    # Add timestamp
    data['timestamp'] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    
    print("API Response:", data)
    return data

def detect_outliers_zscore(data, key):
    values = np.array([item[key] for item in data])
    mean = np.mean(values)
    std_dev = np.std(values)

    if std_dev == 0:
        print(f"No variation in {key}, outlier detection not applicable.")
        return

    threshold = 3  # Z-Score threshold
    outliers = np.where(np.abs((values - mean) / std_dev) > threshold)
    
    if outliers[0].size > 0:
        print(f"Outliers detected in {key} (Z-Score): {values[outliers]}")

def detect_outliers_iqr(data, key):
    values = np.array([item[key] for item in data])
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = values[(values < lower_bound) | (values > upper_bound)]

    if outliers.size > 0:
        print(f"Outliers detected in {key} (IQR): {outliers}")

def impute_missing_values_knn(df):
    # KNN ile eksik verileri doldurma
    df_imputed = imputer.fit_transform(df)
    return pd.DataFrame(df_imputed, columns=df.columns)

# Eksik veriyi doldurmak için kullanılacak fonksiyon
def impute_missing_value_with_knn(data, previous_data):
    # Verileri pandas DataFrame'e dönüştür
    df = pd.DataFrame(previous_data)

    # Yeni veriyi ekle
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)   

    # KNN ile eksik verileri doldur
    df_filled = impute_missing_values_knn(df)
    
    # Son doldurulmuş veri
    return df_filled.iloc[-1].to_dict()

def validate_and_process_data(data, previous_values=[]):
    try:
        if 'main' not in data or 'wind' not in data:
            raise ValueError("Missing expected data in API response")
        
        # Verileri eksik veri kontrolü ve KNN ile tamamlama
        data_to_impute = {
            'temp': data['main'].get('temp', np.nan),
            'humidity': data['main'].get('humidity', np.nan),
            'wind_speed': data['wind'].get('speed', np.nan),
            'wind_deg': data['wind'].get('deg', np.nan)
        }

        # Eksik verileri KNN ile doldur
        filled_data = impute_missing_value_with_knn(data_to_impute, previous_values)
        data['main']['temp'] = filled_data['temp']
        data['main']['humidity'] = int(filled_data['humidity'])  # Humidity'yi integer olarak kaydediyoruz
        data['wind']['speed'] = filled_data['wind_speed']
        data['wind']['deg'] = filled_data['wind_deg']

        # Veriyi önceki değerlere ekle
        previous_values.append(filled_data)

        weather_data = WeatherData(
            name=data.get('name', 'Unknown location'),
            main=data['main'],
            wind=data['wind']
        )
        
        # Sıcaklığı Celsius'a çevir
        temp_c = kelvin_to_celsius(weather_data.main.temp)
        
        # Rüzgar hızını km/sa cinsine çevir
        wind_speed_kmh = float(weather_data.wind.speed) * 3.6
        
        # Rüzgar verilerini daha kullanıcı dostu bir formata dönüştür
        wind_data = f"{wind_speed_kmh:.1f} km/h {wind_direction_to_text(weather_data.wind.deg)} direction"
        
        # "Hissedilen" sıcaklığı hesapla
        feels_like_temp = calculate_feels_like(temp_c, float(weather_data.main.humidity), wind_speed_kmh)
        
        # Aykırı değer tespiti (Z-Score)
        detect_outliers_zscore([{'temp_c': temp_c}], 'temp_c')
        detect_outliers_zscore([{'humidity': float(weather_data.main.humidity)}], 'humidity')
        detect_outliers_zscore([{'wind_speed': wind_speed_kmh}], 'wind_speed')
        
        # Aykırı değer tespiti (IQR)
        detect_outliers_iqr([{'temp_c': temp_c}], 'temp_c')
        detect_outliers_iqr([{'humidity': float(weather_data.main.humidity)}], 'humidity')
        detect_outliers_iqr([{'wind_speed': wind_speed_kmh}], 'wind_speed')
        
        # Sonuçları yazdır
        print(f"Timestamp: {data['timestamp']}")
        print(f"Location: {weather_data.name}")
        print(f"Temperature: {temp_c:.2f}°C")
        print(f"Feels Like: {feels_like_temp:.2f}°C")
        print(f"Humidity: {weather_data.main.humidity}%")
        print(f"Wind: {wind_data}")
    
    except ValueError as e:
        print(f"Data Error: {e}")
    except ValidationError as e:
        print(f"Validation Error: {e}")
    except KeyError as e:
        print(f"Missing key in API response: {e}")

def kelvin_to_celsius(temp_k):
    return float(temp_k) - 273.15

def wind_direction_to_text(deg):
    if (deg >= 0 and deg < 45) or (deg >= 315 and deg <= 360):
        return "north"
    elif deg >= 45 and deg < 135:
        return "east"
    elif deg >= 135 and deg < 225:
        return "south"
    elif deg >= 225 and deg < 315:
        return "west"
    else:
        return "unknown direction"

def calculate_feels_like(temp_c, humidity, wind_speed):
    return temp_c - ((100 - humidity) / 10) + (wind_speed / 10)

def write_to_influxdb(data):
    local_time = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(pytz.timezone('Europe/Istanbul'))
    
    point = Point("weather_data") \
        .tag("location", data['name']) \
        .field("temperature", kelvin_to_celsius(data['main']['temp'])) \
        .field("feels_like", kelvin_to_celsius(data['main']['feels_like'])) \
        .field("humidity", int(data['main']['humidity'])) \
        .field("wind_speed", data['wind']['speed'] * 3.6) \
        .field("wind_direction", int(data['wind']['deg'])) \
        .time(local_time, WritePrecision.NS)

    write_api.write(bucket=bucket, org=org, record=point)
    print("Data written to InfluxDB")
    
# Veritabanından verileri çekip CSV'ye yazma fonksiyonu
def query_and_save_to_csv():
    query_api = client.query_api()

    # Sorguyu oluştur
    query = '''
    from(bucket: "weather_data")
    |> range(start: -1d)  // Son 1 günün verilerini çekmek için
    |> filter(fn: (r) => r["_measurement"] == "weather_data")
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''

    # Verileri çek
    tables = query_api.query(query, org=org)

    # Sonuçları pandas DataFrame'e çevir
    records = []
    for table in tables:
        for record in table.records:
            records.append(record.values)
    
    df = pd.DataFrame.from_records(records)

    # DataFrame'in sütunlarını doğru şekilde düzenleyin
    df = df[["_time", "location", "temperature", "feels_like", "humidity", "wind_speed", "wind_direction"]]

    # Zaman damgasını UTC'den Europe/Istanbul'a dönüştür
    df['_time'] = pd.to_datetime(df['_time']).dt.tz_convert('Europe/Istanbul')

    # CSV dosyasına yaz
    df.to_csv(r'C:\Users\sumey\Masaüstü\case_study\weather_data.csv', index=False)
    print("Veriler CSV dosyasına kaydedildi.")

# Kullanıcıdan enlem ve boylam bilgilerini al
lat = input("Please enter latitude: ")
lon = input("Please enter longitude: ")

# Önceki değerler listesi
previous_values = []

# Her 5 dakikada bir veri çek ve işle
while True:
    data = fetch_weather_data(lat, lon, api_key)
    validate_and_process_data(data, previous_values)
    write_to_influxdb(data)  # InfluxDB'ye veri yazma
    query_and_save_to_csv()  # Veritabanındaki verileri CSV'ye kaydet
    time.sleep(300)  # 300 saniye (5 dakika) bekle



    #39.924997660618786, 32.837078596522346