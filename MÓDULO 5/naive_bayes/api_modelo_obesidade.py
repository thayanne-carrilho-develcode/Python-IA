from flask import Flask, request, jsonify
from pydantic import BaseModel
from flask_pydantic import validate
import joblib
import pandas as pd 

#Criar a instancia do flask
app = Flask(__name__)

#Classe que recebera os dados dos formulários
class request_body(BaseModel):
    Genero_Masculino: int 
    Idade: int                
    Historico_Familiar_Sobrepeso: int 
    Consumo_Alta_Caloria_Com_Frequencia: int  
    Consumo_Vegetais_Com_Frequencia: int
    Refeicoes_Dia: int                  
    Consumo_Alimentos_entre_Refeico: int
    Fumante: int                        
    Consumo_Agua: int                   
    Monitora_Calorias_Ingeridas: int    
    Nivel_Atividade_Fisica: int         
    Nivel_Uso_Tela: int                 
    Consumo_Alcool: int                 
    Transporte_Automovel: int           
    Transporte_Bicicleta: int           
    Transporte_Motocicleta: int         
    Transporte_Publico: int             
    Transporte_Caminhada: int           


#carregar modelo
modelo_obesidade = joblib.load('./modelo_obesidade.pkl') 

@app.route('/predict', methods=['POST'])
@validate()
def predict(body: request_body):

    #tranformar o body em um dataframe
    predict_df = pd.DataFrame(body.model_dump(), index=[1])

    #bucketing 
    bins = [10, 20, 30, 40, 50, 60, 70]
    bins_ordinal = [0, 1, 2, 3, 4, 5]
    labels_faixa_etaria = ['10-20', '20-30', '30-40', '40-50', '50-60', '60-70']

    predict_df['Faixa_Etaria_String'] = pd.cut(
        x=predict_df['Idade'], bins=bins, labels=labels_faixa_etaria, include_lowest=True
    )

    #Deixar as k melhores features
    predict_df = predict_df[['Historico_Familiar_Sobrepeso',
                              'Consumo_Alta_Caloria_Com_Frequencia',
                              'Consumo_Alimentos_entre_Refeicoes',
                              'Monitora_Calorias_Ingeridas',
                              'Nivel_Atividade_Fisica',
                              'Nivel_Uso_Tela',
                              'Transporte_Caminhada',
                              'Faixa_Etaria'
                               ]]
    #predizer a classificação
    y_pred = modelo_obesidade.predict(predict_df)[0].astype(int)

    #retornar valores na API
    return jsonify({'obesidade': y_pred.tolist()})


#Executar a aplicação
if __name__ == '__main__':
    app.run(port=5000, debug=True)
    









