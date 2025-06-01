from fastapi import FastAPI # type: ignore
import uvicorn # type: ignore
from pydantic import BaseModel # type: ignore
import joblib # type: ignore
import pandas as pd

#criar instancia do fastapi
app = FastAPI()

# criar uma classe com os dados de entrada que virão no request body com os tipos esperados
class request_body(BaseModel):
    tempo_na_empresa: int
    nivel_na_empresa: int
    
#Carregar model para realizar a predição
modelo_poly = joblib.load('./modelo_salario.pkl')

@app.post('/predict')
def predict(data : request_body):
        
    input_features = {
        'tempo_na_empresa': data.tempo_na_empresa,
        'nivel_na_empresa': data.nivel_na_empresa
    }

    pred_df = pd.DataFrame(input_features, index=[1])

    y_pred = modelo_poly.predict(pred_df)[0].astype(float)
    
    return {'salario_em_reais': y_pred.tolist()}