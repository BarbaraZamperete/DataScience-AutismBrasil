from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def pergunta_1(df_ia):
    """
    Pergunta 1: Quantas consultas médicas um paciente realizará no ano?
    """
    X_p1 = df_ia.drop(columns=['Numero_de_Consultas'])
    y_p1 = df_ia['Numero_de_Consultas']

    # Tratamento de valores ausentes
    df_ia.fillna(df_ia.mean(), inplace=True)  # Exemplo de preenchimento com a média

    # Normalização
    scaler = StandardScaler()
    X_p1 = scaler.fit_transform(X_p1)

    """
    Dividir os dados em conjuntos de treinamento e teste
    """
    X_train_p1, X_test_p1, y_train_p1, y_test_p1 = train_test_split(X_p1, y_p1, test_size=0.2, random_state=42)

    """
    Treinar o modelo 1
    """
    model_p1 = LinearRegression()
    model_p1.fit(X_train_p1, y_train_p1)

    """
    Treinar o modelo 2
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5)
    grid_search.fit(X_train_p1, y_train_p1)
    model_p2 = grid_search.best_estimator_

    """
    Fazer previsões
    """
    y_pred_p1 = model_p1.predict(X_test_p1)
    y_pred_p2 = model_p2.predict(X_test_p1)

    """
    Calcular o erro quadrático médio e R²
    """
    mse_p1 = mean_squared_error(y_test_p1, y_pred_p1)
    r2_p1 = r2_score(y_test_p1, y_pred_p1)

    mse_p2 = mean_squared_error(y_test_p1, y_pred_p2)
    r2_p2 = r2_score(y_test_p1, y_pred_p2)

    # Exibir uma explicação dos resultados do treinamento
    print("Modelo 1 - Linear Regression:")
    print(f"MSE: {mse_p1:.2f}, R²: {r2_p1:.2f}")
    
    print("Modelo 2 - Random Forest Regressor:")
    print(f"MSE: {mse_p2:.2f}, R²: {r2_p2:.2f}")

    return mse_p1, r2_p1, mse_p2, r2_p2


def corelacao(df_ia):

    # Gráfico de Correlação
    correlation_matrix = df_ia.corr()  # Calcula a matriz de correlação
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, square=True, cbar_kws={"shrink": .8})
    ax.set_title("Matriz de Correlação", fontsize=16)
    plt.show()


if __name__ == "__main__":
    df_ia = pd.read_csv("data/base_ajustada_realista_pre_processada.csv", sep=';')
    # print(pergunta_1(df_ia))
    corelacao(df_ia)