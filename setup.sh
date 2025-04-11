#!/bin/bash

# Gerekli kütüphaneleri yükle
pip install -r requirements.txt

# Streamlit yapılandırması
mkdir -p ~/.streamlit
echo "[server]" > ~/.streamlit/config.toml
echo "headless = true" >> ~/.streamlit/config.toml
echo "port = 8501" >> ~/.streamlit/config.toml
echo "enableCORS = false" >> ~/.streamlit/config.toml
