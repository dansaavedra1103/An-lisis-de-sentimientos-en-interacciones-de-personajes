import requests
from pathlib import Path
# Descargar el PDF
## Obtención de datos desde la URL##
url = "https://web.seducoahuila.gob.mx/biblioweb/upload/13%20Cuentos%20Comp%20Kafka.pdf"

response = requests.get(url, timeout=30)
response.raise_for_status()  

## Escritura del archivo en formato PDF##
Path("resources/kafka_cuentos.pdf").write_bytes(response.content)
print("PDF descargado correctamente")