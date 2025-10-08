from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Cargar el modelo entrenado
modelo = joblib.load("modelo_costos_rf.pkl")

# Inicializar la aplicación FastAPI
app = FastAPI()

# Habilitar CORS para permitir solicitudes desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # tu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Definir el modelo de entrada
class DatosEntrada(BaseModel):
    seccion: str
    sub_categoria: str
    tipo_solicitud: str
    proceso: str
    tiempo_promedio: float    

# Endpoint de predicción
@app.post("/predecir")
def predecir(data: DatosEntrada):
    try:
        # Combinar los campos en el mismo formato que el entrenamiento
        entrada_texto = f"{data.seccion} {data.sub_categoria} {data.tipo_solicitud} {data.proceso} {data.tiempo_promedio}"

        # Realizar la predicción
        prediccion = modelo.predict([entrada_texto])
        print(modelo)
        return {"costo_estimado": round(prediccion[0], 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
