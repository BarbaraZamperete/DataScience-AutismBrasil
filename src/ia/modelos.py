from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import joblib
import os
import shutil
import traceback

class ModeloAutismo:
    def __init__(self):
        # Scalers separados para cada modelo
        self.scaler_consultas = StandardScaler()
        self.scaler_medicamentos = StandardScaler()
        self.label_encoders = {}
        
        # Modelos para número de consultas
        self.modelo_consultas_rf = None
        self.modelo_consultas_lr = None
        
        # Modelos para medicamentos
        self.modelo_medicamentos_rf = None
        self.modelo_medicamentos_lr = None
        
        # Definir ordem das colunas para cada modelo
        self.colunas_basicas = ['Idade', 'Ano_Diagnostico', 'Regiao', 'Renda_Familiar', 
                              'Acesso_a_Servicos', 'Zona', 'Tratamento']
        
        # Usar as mesmas colunas para ambos os modelos
        self.colunas_consultas = self.colunas_basicas.copy()
        self.colunas_medicamentos = self.colunas_basicas.copy()

    def treinar_modelos(self, df):
        """
        Treina todos os modelos necessários para as predições
        """
        try:
            print("\n=== Iniciando treinamento dos modelos ===")
            
            # Remover modelos antigos
            self.remover_modelos_salvos()
            self.limpar_modelos()
            
            print(f"Colunas disponíveis no DataFrame: {df.columns.tolist()}")
            print(f"Colunas básicas que serão usadas: {self.colunas_basicas}")
            
            # Preparar dados
            print("\nPreparando dados para treinamento...")
            df = self._preparar_dados(df)
            
            # 1. Treinamento dos modelos para número de consultas
            print("\n1. Treinando modelos para previsão de número de consultas...")
            print(f"Usando colunas: {self.colunas_basicas}")
            X_consultas = df[self.colunas_basicas]
            y_consultas = df['Numero_de_Consultas']
            
            # Resetar o scaler e treinar com as novas colunas
            print("Normalizando dados para consultas...")
            self.scaler_consultas = StandardScaler()
            X_consultas_norm = self.scaler_consultas.fit_transform(X_consultas)
            
            # Treinar Random Forest
            print("\nTreinando Random Forest para consultas...")
            self.modelo_consultas_rf = self._otimizar_modelo_consultas(X_consultas_norm, y_consultas)
            pred_rf = self.modelo_consultas_rf.predict(X_consultas_norm)
            mse_rf = mean_squared_error(y_consultas, pred_rf)
            r2_rf = r2_score(y_consultas, pred_rf)
            print(f"Random Forest - MSE: {mse_rf:.2f}, R²: {r2_rf:.2f}")
            
            # Treinar Regressão Linear
            print("\nTreinando Regressão Linear para consultas...")
            self.modelo_consultas_lr = LinearRegression()
            self.modelo_consultas_lr.fit(X_consultas_norm, y_consultas)
            pred_lr = self.modelo_consultas_lr.predict(X_consultas_norm)
            mse_lr = mean_squared_error(y_consultas, pred_lr)
            r2_lr = r2_score(y_consultas, pred_lr)
            print(f"Regressão Linear - MSE: {mse_lr:.2f}, R²: {r2_lr:.2f}")
            
            # 2. Treinamento dos modelos para medicamentos
            print("\n2. Treinando modelos para previsão de necessidade de medicamentos...")
            print(f"Usando colunas: {self.colunas_basicas}")
            X_medicamentos = df[self.colunas_basicas]
            y_medicamentos = df['Tratamento'].apply(lambda x: 1 if x in ['Medicamentos', 'Ambos'] else 0)
            
            # Resetar o scaler e treinar com as novas colunas
            print("Normalizando dados para medicamentos...")
            self.scaler_medicamentos = StandardScaler()
            X_medicamentos_norm = self.scaler_medicamentos.fit_transform(X_medicamentos)
            
            # Verificar se há pelo menos duas classes
            classes_unicas = np.unique(y_medicamentos)
            if len(classes_unicas) < 2:
                print(f"AVISO: Apenas uma classe encontrada nos dados: {classes_unicas}")
                # Usar DummyClassifier como fallback
                self.modelo_medicamentos_rf = DummyClassifier(strategy='constant', constant=classes_unicas[0])
                self.modelo_medicamentos_lr = DummyClassifier(strategy='constant', constant=classes_unicas[0])
                self.modelo_medicamentos_rf.fit(X_medicamentos_norm, y_medicamentos)
                self.modelo_medicamentos_lr.fit(X_medicamentos_norm, y_medicamentos)
            else:
                # Treinar Random Forest
                print("\nTreinando Random Forest para medicamentos...")
                self.modelo_medicamentos_rf = self._otimizar_modelo_medicamentos(X_medicamentos_norm, y_medicamentos)
                pred_rf = self.modelo_medicamentos_rf.predict(X_medicamentos_norm)
                acc_rf = accuracy_score(y_medicamentos, pred_rf)
                print(f"Random Forest - Acurácia: {acc_rf:.2f}")
                
                # Treinar Regressão Logística
                print("\nTreinando Regressão Logística para medicamentos...")
                self.modelo_medicamentos_lr = LogisticRegression(random_state=42)
                self.modelo_medicamentos_lr.fit(X_medicamentos_norm, y_medicamentos)
                pred_lr = self.modelo_medicamentos_lr.predict(X_medicamentos_norm)
                acc_lr = accuracy_score(y_medicamentos, pred_lr)
                print(f"Regressão Logística - Acurácia: {acc_lr:.2f}")
            
            print("\n=== Treinamento concluído com sucesso ===")
            return True
            
        except Exception as e:
            print("\n=== ERRO DURANTE O TREINAMENTO ===")
            print(f"Tipo do erro: {type(e).__name__}")
            print(f"Mensagem de erro: {str(e)}")
            print("\nStack trace completo:")
            print(traceback.format_exc())
            raise ValueError(f"Erro ao treinar modelos: {str(e)}")

    def remover_modelos_salvos(self, diretorio='modelos'):
        """
        Remove os arquivos dos modelos salvos
        """
        print(f"\n=== Removendo modelos salvos do diretório {diretorio} ===")
        
        if os.path.exists(diretorio):
            try:
                shutil.rmtree(diretorio)
                print(f"Diretório {diretorio} removido com sucesso")
            except Exception as e:
                print(f"Erro ao remover diretório: {str(e)}")
        else:
            print(f"Diretório {diretorio} não existe")

    def limpar_modelos(self):
        """
        Remove todos os modelos e scalers existentes
        """
        print("\n=== Limpando modelos existentes ===")
        
        # Limpar scalers
        self.scaler_consultas = StandardScaler()
        self.scaler_medicamentos = StandardScaler()
        
        # Limpar modelos
        self.modelo_consultas_rf = None
        self.modelo_consultas_lr = None
        self.modelo_medicamentos_rf = None
        self.modelo_medicamentos_lr = None
        
        # Limpar label encoders
        self.label_encoders = {}
        
        print("Todos os modelos foram removidos")

    def _otimizar_modelo_consultas(self, X, y):
        """
        Otimiza o modelo de previsão de consultas usando GridSearchCV
        """
        param_grid_rf = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        # Treinar modelo base para comparação
        modelo_base = RandomForestRegressor(random_state=42)
        modelo_base.fit(X, y)
        pred_base = modelo_base.predict(X)
        mae_base = mean_absolute_error(y, pred_base)
        mse_base = mean_squared_error(y, pred_base)
        
        # Otimizar modelo
        grid_search = GridSearchCV(RandomForestRegressor(random_state=42), 
                                 param_grid_rf, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        modelo_otm = grid_search.best_estimator_
        pred_otm = modelo_otm.predict(X)
        mae_otm = mean_absolute_error(y, pred_otm)
        mse_otm = mean_squared_error(y, pred_otm)
        
        print("\nComparação - Previsão de Consultas:")
        print("Antes da otimização:")
        print(f"MAE: {mae_base:.2f}")
        print(f"MSE: {mse_base:.2f}")
        print("\nDepois da otimização:")
        print(f"MAE: {mae_otm:.2f}")
        print(f"MSE: {mse_otm:.2f}")
        print(f"Melhores parâmetros: {grid_search.best_params_}")
        
        return modelo_otm

    def _otimizar_modelo_medicamentos(self, X, y):
        """
        Otimiza o modelo de previsão de medicamentos usando GridSearchCV
        """
        param_grid_rf = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        # Treinar modelo base para comparação
        modelo_base = RandomForestClassifier(random_state=42)
        modelo_base.fit(X, y)
        pred_base = modelo_base.predict(X)
        acc_base = accuracy_score(y, pred_base)
        
        # Otimizar modelo
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), 
                                 param_grid_rf, cv=5, scoring='accuracy')
        grid_search.fit(X, y)
        modelo_otm = grid_search.best_estimator_
        pred_otm = modelo_otm.predict(X)
        acc_otm = accuracy_score(y, pred_otm)
        
        print("\nComparação - Previsão de Necessidade de Medicamentos:")
        print("Antes da otimização:")
        print(f"Acurácia: {acc_base:.2f}")
        print("\nDepois da otimização:")
        print(f"Acurácia: {acc_otm:.2f}")
        print(f"Melhores parâmetros: {grid_search.best_params_}")
        
        return modelo_otm

    def _preparar_dados(self, df):
        """
        Prepara os dados para treinamento e predição
        """
        try:
            print("\n=== Preparando dados para treinamento ===")
            print(f"Colunas iniciais: {df.columns.tolist()}")
            
            # Criar cópia para não modificar os dados originais
            df = df.copy()
            
            # Converter tipos numéricos
            print("\nConvertendo tipos numéricos...")
            df['Idade'] = pd.to_numeric(df['Idade'])
            df['Ano_Diagnostico'] = pd.to_numeric(df['Ano_Diagnostico'])
            df['Numero_de_Consultas'] = pd.to_numeric(df['Numero_de_Consultas'])
            print("Conversão numérica concluída")
            
            # Definir mapeamentos para valores categóricos
            mapeamentos = {
                'Regiao': {
                    0: 'Norte', 1: 'Nordeste', 2: 'Centro-Oeste', 3: 'Sudeste', 4: 'Sul',
                    'Norte': 'Norte', 'Nordeste': 'Nordeste', 'Centro-Oeste': 'Centro-Oeste',
                    'Sudeste': 'Sudeste', 'Sul': 'Sul'
                },
                'Renda_Familiar': {
                    0: 'Baixa', 1: 'Média', 2: 'Alta',
                    'Baixa': 'Baixa', 'Média': 'Média', 'Alta': 'Alta'
                },
                'Acesso_a_Servicos': {
                    0: 'Não', 1: 'Sim',
                    'Não': 'Não', 'Sim': 'Sim'
                },
                'Zona': {
                    0: 'Urbana', 1: 'Rural',
                    'Urbana': 'Urbana', 'Rural': 'Rural'
                },
                'Tratamento': {
                    0: 'Terapias', 1: 'Medicamentos', 2: 'Ambos',
                    'Terapias': 'Terapias', 'Medicamentos': 'Medicamentos', 'Ambos': 'Ambos'
                }
            }
            
            # Converter valores categóricos
            print("\nConvertendo valores categóricos...")
            for coluna, mapeamento in mapeamentos.items():
                if coluna in df.columns:
                    print(f"\nProcessando coluna: {coluna}")
                    valores_originais = df[coluna].value_counts().to_dict()
                    print(f"Valores originais: {valores_originais}")
                    
                    df[coluna] = df[coluna].map(lambda x: mapeamento.get(x, x))
                    valores_mapeados = df[coluna].value_counts().to_dict()
                    print(f"Valores após mapeamento: {valores_mapeados}")
                    
                    valores_invalidos = df[coluna][~df[coluna].isin(set(mapeamento.values()))].unique()
                    if len(valores_invalidos) > 0:
                        raise ValueError(f"Valor inválido para {coluna}: {valores_invalidos[0]}. "
                                      f"Valores permitidos: {list(set(mapeamento.values()))}")
                    
                    # Criar novo encoder se não existir
                    if coluna not in self.label_encoders:
                        le = LabelEncoder()
                        le.fit(list(set(mapeamento.values())))
                        self.label_encoders[coluna] = le
                    
                    # Usar o encoder para transformar
                    df[coluna] = self.label_encoders[coluna].transform(df[coluna])
                    valores_encoded = df[coluna].value_counts().to_dict()
                    print(f"Valores após encoding: {valores_encoded}")
            
            # Verificar se todas as colunas necessárias estão presentes
            colunas_necessarias = self.colunas_basicas + ['Numero_de_Consultas']
            print(f"\nVerificando colunas necessárias: {colunas_necessarias}")
            
            colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]
            if colunas_faltantes:
                raise ValueError(f"Colunas faltantes: {', '.join(colunas_faltantes)}")
            
            print("\n=== Preparação dos dados concluída ===")
            print(f"Colunas finais: {df.columns.tolist()}")
            return df
            
        except Exception as e:
            print("\n=== ERRO DURANTE A PREPARAÇÃO DOS DADOS ===")
            print(f"Tipo do erro: {type(e).__name__}")
            print(f"Mensagem de erro: {str(e)}")
            if 'df' in locals():
                print(f"Colunas disponíveis: {df.columns.tolist()}")
            raise e

    def fazer_predicoes(self, dados_entrada):
        """
        Faz predições usando os melhores modelos para cada tarefa
        """
        try:
            print("\n=== Iniciando processo de predição ===")
            print(f"Dados de entrada recebidos: {dados_entrada}")
            
            # Verificar se os modelos foram treinados
            modelos = {
                'consultas_rf': self.modelo_consultas_rf,
                'consultas_lr': self.modelo_consultas_lr,
                'medicamentos_rf': self.modelo_medicamentos_rf,
                'medicamentos_lr': self.modelo_medicamentos_lr
            }
            modelos_faltantes = [nome for nome, modelo in modelos.items() if modelo is None]
            if modelos_faltantes:
                raise ValueError(f"Os seguintes modelos não foram treinados: {', '.join(modelos_faltantes)}")
            
            # Preparar dados iniciais usando o método _preparar_dados_predicao
            print("\nPreparando dados para predição...")
            df_pred = self._preparar_dados_predicao(dados_entrada)
            print(f"Colunas disponíveis após preparação: {df_pred.columns.tolist()}")
            
            # 1. Fazer previsões de consultas
            print("\nRealizando predição de consultas...")
            print(f"Colunas necessárias para consultas: {self.colunas_basicas}")
            colunas_faltantes = [col for col in self.colunas_basicas if col not in df_pred.columns]
            if colunas_faltantes:
                raise ValueError(f"Colunas faltantes para predição de consultas: {colunas_faltantes}")
            
            X_consultas = df_pred[self.colunas_basicas].copy()
            print("Normalizando dados para consultas...")
            X_consultas_norm = self.scaler_consultas.transform(X_consultas)
            
            pred_consultas_rf = max(1, round(self.modelo_consultas_rf.predict(X_consultas_norm)[0]))
            pred_consultas_lr = max(1, round(self.modelo_consultas_lr.predict(X_consultas_norm)[0]))
            print(f"Previsões de consultas - RF: {pred_consultas_rf}, LR: {pred_consultas_lr}")
            
            # 2. Fazer previsões para medicamentos
            print("\nRealizando predição de medicamentos...")
            print(f"Colunas necessárias para medicamentos: {self.colunas_basicas}")
            colunas_faltantes = [col for col in self.colunas_basicas if col not in df_pred.columns]
            if colunas_faltantes:
                raise ValueError(f"Colunas faltantes para predição de medicamentos: {colunas_faltantes}")
            
            X_medicamentos = df_pred[self.colunas_basicas].copy()
            print("Normalizando dados para medicamentos...")
            X_medicamentos_norm = self.scaler_medicamentos.transform(X_medicamentos)
            
            def get_prob_medicamentos(modelo, X):
                """Helper function to get medication probability"""
                if isinstance(modelo, DummyClassifier):
                    pred = modelo.predict(X)[0]
                    return 1.0 if pred == 1 else 0.0
                else:
                    try:
                        probs = modelo.predict_proba(X)[0]
                        if len(probs) == 2:
                            return float(probs[1])  # Probabilidade da classe positiva
                        return 1.0 if probs[0] == 1 else 0.0
                    except Exception as e:
                        print(f"Erro ao obter probabilidades: {str(e)}")
                        pred = modelo.predict(X)[0]
                        return 1.0 if pred == 1 else 0.0
            
            pred_medicamentos_rf = get_prob_medicamentos(self.modelo_medicamentos_rf, X_medicamentos_norm)
            pred_medicamentos_lr = get_prob_medicamentos(self.modelo_medicamentos_lr, X_medicamentos_norm)
            print(f"Previsões de medicamentos - RF: {pred_medicamentos_rf:.2f}, LR: {pred_medicamentos_lr:.2f}")
            
            resultados = {
                'consultas': {
                    'random_forest': pred_consultas_rf,
                    'regressao_linear': pred_consultas_lr
                },
                'medicamentos': {
                    'random_forest': pred_medicamentos_rf,
                    'regressao_logistica': pred_medicamentos_lr
                }
            }
            
            print("\n=== Predições concluídas com sucesso ===")
            print(f"Resultados: {resultados}")
            return resultados
            
        except Exception as e:
            print("\n=== ERRO DURANTE A PREDIÇÃO ===")
            print(f"Tipo do erro: {type(e).__name__}")
            print(f"Mensagem de erro: {str(e)}")
            print("\nStack trace completo:")
            print(traceback.format_exc())
            print("\nEstado atual dos dados:")
            print(f"Colunas necessárias para consultas: {self.colunas_basicas}")
            print(f"Colunas necessárias para medicamentos: {self.colunas_basicas}")
            if 'df_pred' in locals():
                print(f"Colunas disponíveis no DataFrame: {df_pred.columns.tolist()}")
            raise ValueError(f"Erro ao fazer predições: {str(e)}")

    def _preparar_dados_predicao(self, dados):
        """
        Prepara dados de entrada para predição
        """
        print("\n=== Preparando dados para predição ===")
        print(f"Dados recebidos: {dados}")
        
        # Criar DataFrame com os dados de entrada
        df = pd.DataFrame([dados])
        print(f"Colunas iniciais: {df.columns.tolist()}")
        
        # Converter tipos numéricos
        print("\nConvertendo tipos numéricos...")
        if 'Idade' in df.columns:
            df['Idade'] = pd.to_numeric(df['Idade'])
            print(f"Idade convertida: {df['Idade'].values[0]}")
        if 'Ano_Diagnostico' in df.columns:
            df['Ano_Diagnostico'] = pd.to_numeric(df['Ano_Diagnostico'])
            print(f"Ano_Diagnostico convertido: {df['Ano_Diagnostico'].values[0]}")
        
        # Definir mapeamentos para valores categóricos
        mapeamentos = {
            'Regiao': {
                0: 'Norte', 1: 'Nordeste', 2: 'Centro-Oeste', 3: 'Sudeste', 4: 'Sul',
                'Norte': 'Norte', 'Nordeste': 'Nordeste', 'Centro-Oeste': 'Centro-Oeste',
                'Sudeste': 'Sudeste', 'Sul': 'Sul'
            },
            'Renda_Familiar': {
                0: 'Baixa', 1: 'Média', 2: 'Alta',
                'Baixa': 'Baixa', 'Média': 'Média', 'Alta': 'Alta'
            },
            'Acesso_a_Servicos': {
                0: 'Não', 1: 'Sim',
                'Não': 'Não', 'Sim': 'Sim'
            },
            'Zona': {
                0: 'Urbana', 1: 'Rural',
                'Urbana': 'Urbana', 'Rural': 'Rural'
            },
            'Tratamento': {
                0: 'Terapias', 1: 'Medicamentos', 2: 'Ambos',
                'Terapias': 'Terapias', 'Medicamentos': 'Medicamentos', 'Ambos': 'Ambos'
            }
        }
        
        # Verificar se todas as colunas necessárias estão presentes
        colunas_necessarias = ['Idade', 'Ano_Diagnostico', 'Regiao', 'Renda_Familiar', 
                             'Acesso_a_Servicos', 'Zona', 'Tratamento']
        print(f"\nVerificando colunas necessárias: {colunas_necessarias}")
        
        colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]
        if colunas_faltantes:
            raise ValueError(f"Colunas faltantes nos dados de entrada: {', '.join(colunas_faltantes)}")
        
        # Converter valores categóricos
        print("\nConvertendo valores categóricos...")
        for coluna, mapeamento in mapeamentos.items():
            if coluna in df.columns:
                valor_original = df[coluna].values[0]
                df[coluna] = df[coluna].map(lambda x: mapeamento.get(x, x))
                valor_mapeado = df[coluna].values[0]
                print(f"{coluna}: {valor_original} -> {valor_mapeado}")
                
                valores_invalidos = df[coluna][~df[coluna].isin(set(mapeamento.values()))].unique()
                if len(valores_invalidos) > 0:
                    raise ValueError(f"Valor inválido para {coluna}: {valores_invalidos[0]}. "
                                  f"Valores permitidos: {list(set(mapeamento.values()))}")
                
                # Usar o encoder para transformar
                if coluna in self.label_encoders:
                    df[coluna] = self.label_encoders[coluna].transform(df[coluna])
                    valor_encoded = df[coluna].values[0]
                    print(f"{coluna} (encoded): {valor_mapeado} -> {valor_encoded}")
                else:
                    raise ValueError(f"Encoder não encontrado para a coluna {coluna}. Por favor, treine os modelos primeiro.")
        
        print("\n=== Preparação dos dados concluída ===")
        print(f"Colunas finais: {df.columns.tolist()}")
        return df

    def salvar_modelos(self, diretorio='modelos'):
        """
        Salva os modelos treinados
        """
        if not os.path.exists(diretorio):
            os.makedirs(diretorio)
            
        # Salvar modelos de consultas
        joblib.dump(self.modelo_consultas_rf, f'{diretorio}/modelo_consultas_rf.pkl')
        joblib.dump(self.modelo_consultas_lr, f'{diretorio}/modelo_consultas_lr.pkl')
        joblib.dump(self.scaler_consultas, f'{diretorio}/scaler_consultas.pkl')
        
        # Salvar modelos de medicamentos
        joblib.dump(self.modelo_medicamentos_rf, f'{diretorio}/modelo_medicamentos_rf.pkl')
        joblib.dump(self.modelo_medicamentos_lr, f'{diretorio}/modelo_medicamentos_lr.pkl')
        joblib.dump(self.scaler_medicamentos, f'{diretorio}/scaler_medicamentos.pkl')
        
        # Salvar outros componentes
        joblib.dump(self.label_encoders, f'{diretorio}/label_encoders.pkl')
        joblib.dump({
            'colunas_consultas': self.colunas_basicas,
            'colunas_medicamentos': self.colunas_basicas
        }, f'{diretorio}/colunas.pkl')

    def carregar_modelos(self, diretorio='modelos'):
        """
        Carrega os modelos salvos
        """
        try:
            # Verificar se o diretório existe
            if not os.path.exists(diretorio):
                raise FileNotFoundError(f"Diretório {diretorio} não encontrado. Por favor, treine os modelos primeiro.")

            # Lista de todos os arquivos necessários
            arquivos_necessarios = [
                'modelo_consultas_rf.pkl',
                'modelo_consultas_lr.pkl',
                'modelo_medicamentos_rf.pkl',
                'modelo_medicamentos_lr.pkl',
                'scaler_consultas.pkl',
                'scaler_medicamentos.pkl',
                'label_encoders.pkl',
                'colunas.pkl'
            ]
            
            # Verificar se todos os arquivos existem
            for arquivo in arquivos_necessarios:
                if not os.path.exists(os.path.join(diretorio, arquivo)):
                    raise FileNotFoundError(f"Arquivo {arquivo} não encontrado em {diretorio}. Por favor, treine os modelos primeiro.")

            # Carregar modelos de consultas
            self.modelo_consultas_rf = joblib.load(f'{diretorio}/modelo_consultas_rf.pkl')
            self.modelo_consultas_lr = joblib.load(f'{diretorio}/modelo_consultas_lr.pkl')
            self.scaler_consultas = joblib.load(f'{diretorio}/scaler_consultas.pkl')
            
            # Carregar modelos de medicamentos
            self.modelo_medicamentos_rf = joblib.load(f'{diretorio}/modelo_medicamentos_rf.pkl')
            self.modelo_medicamentos_lr = joblib.load(f'{diretorio}/modelo_medicamentos_lr.pkl')
            self.scaler_medicamentos = joblib.load(f'{diretorio}/scaler_medicamentos.pkl')
            
            # Carregar outros componentes
            self.label_encoders = joblib.load(f'{diretorio}/label_encoders.pkl')
            
            # Carregar ordem das colunas
            colunas = joblib.load(f'{diretorio}/colunas.pkl')
            self.colunas_consultas = colunas['colunas_consultas']
            self.colunas_medicamentos = colunas['colunas_medicamentos']

            return True
        except Exception as e:
            print(f"Erro ao carregar modelos: {str(e)}")
            raise e


def treinar_e_salvar_modelos():
    """
    Função para treinar e salvar os modelos
    """
    try:
        # Tentar carregar dados do diretório data/ primeiro
        try:
            df = pd.read_csv("data/base_ajustada_realista_pre_processada.csv", sep=';')
        except FileNotFoundError:
            # Se não encontrar, tentar carregar do diretório raiz
            try:
                df = pd.read_csv("base_ajustada_realista_pre_processada.csv", sep=';')
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Arquivo 'base_ajustada_realista_pre_processada.csv' não encontrado. "
                    "Verifique se o arquivo existe em:\n"
                    "- data/base_ajustada_realista_pre_processada.csv\n"
                    "- base_ajustada_realista_pre_processada.csv"
                )
        
        print("Dados carregados com sucesso. Iniciando treinamento dos modelos...")
        print(f"Colunas disponíveis: {df.columns.tolist()}")
        
        # Criar e treinar modelos
        modelo = ModeloAutismo()
        modelo.treinar_modelos(df)
        
        # Salvar modelos
        modelo.salvar_modelos()
        print("Modelos treinados e salvos com sucesso!")
        
        return True
    except Exception as e:
        print(f"Erro ao treinar modelos: {str(e)}")
        return False


if __name__ == "__main__":
    treinar_e_salvar_modelos()