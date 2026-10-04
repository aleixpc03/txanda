FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY txanda/ txanda/
COPY app.py .

# Los precios de OMIE se descargan al usarlos. Los futuros de OMIP no se incluyen en la
# imagen (sus condiciones no permiten redistribuirlos): se montan con `-v`.
VOLUME ["/app/datos", "/app/salidas"]
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8501", "--server.address", "0.0.0.0", "--browser.gatherUsageStats", "false"]
