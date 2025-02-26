from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error, f1_score
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
import streamlit as st

class ModeloAutismo:
    def __init__(self):
        """
        Inicializa o modelo com os parâmetros necessários
        """
        # Definir colunas para cada tipo de modelo
        self.colunas_numericas = ['Idade', 'Ano_Diagnostico']
        self.colunas_categoricas = ['Regiao', 'Zona', 'Tratamento', 'Acesso_a_Servicos', 'Renda_Familiar']
        self.colunas_basicas = self.colunas_numericas + self.colunas_categoricas
        
        # Definir colunas específicas para cada modelo
        self.colunas_consultas = self.colunas_basicas
        self.colunas_medicamentos = self.colunas_basicas
        self.colunas_demanda = self.colunas_basicas
        
        # Inicializar transformadores
        self.scaler_consultas = StandardScaler()
        self.scaler_medicamentos = StandardScaler()
        self.scaler_demanda = StandardScaler()
        
        # Inicializar label encoders para variáveis categóricas
        self.label_encoders = {
            'Regiao': LabelEncoder(),
            'Zona': LabelEncoder(),
            'Acesso_a_Servicos': LabelEncoder(),
            'Renda_Familiar': LabelEncoder()
        }
        
        # Inicializar modelos como None
        self.modelo_consultas_rf = None
        self.modelo_consultas_lr = None
        self.modelo_medicamentos_rf = None
        self.modelo_medicamentos_lr = None
        self.modelo_demanda_rf = None
        self.modelo_demanda_lr = None
        
        # Adicionar atributos para armazenar métricas
        self.metricas_otimizacao = {
            'consultas': {
                'antes': {'mae': None, 'mse': None, 'r2': None},
                'depois': {'mae': None, 'mse': None, 'r2': None},
                'params': None
            },
            'medicamentos': {
                'antes': {'accuracy': None, 'f1': None},
                'depois': {'accuracy': None, 'f1': None},
                'params': None
            },
            'demanda': {
                'antes': {'mae': None, 'mse': None, 'r2': None},
                'depois': {'mae': None, 'mse': None, 'r2': None},
                'params': None
            }
        }

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
        self.scaler_demanda = StandardScaler()
        
        # Limpar modelos
        self.modelo_consultas_rf = None
        self.modelo_consultas_lr = None
        self.modelo_medicamentos_rf = None
        self.modelo_medicamentos_lr = None
        self.modelo_demanda_rf = None
        self.modelo_demanda_lr = None
        
        # Limpar label encoders
        self.label_encoders = {
            'Regiao': LabelEncoder(),
            'Zona': LabelEncoder(),
            'Acesso_a_Servicos': LabelEncoder(),
            'Renda_Familiar': LabelEncoder()
        }
        
        print("Todos os modelos foram removidos")

    def _otimizar_modelo_consultas(self, X, y):
        """
        Otimiza o modelo de Random Forest para previsão de consultas
        Pergunta 1: Quantas consultas médicas um paciente realizará no ano?
        """
        print("\nOtimização - Modelo de Consultas:")
        
        # 1. Modelo Base (Linear Regression) para comparação
        lr = LinearRegression()
        lr.fit(X, y)
        y_pred_lr = lr.predict(X)
        mae_lr = mean_absolute_error(y, y_pred_lr)
        mse_lr = mean_squared_error(y, y_pred_lr)
        r2_lr = r2_score(y, y_pred_lr)
        
        print("\nModelo Linear (Base):")
        print(f"MAE: {mae_lr:.2f}")
        print(f"MSE: {mse_lr:.2f}")
        print(f"R²: {r2_lr:.2f}")
        
        # Armazenar métricas
        self.metricas_otimizacao['consultas'] = {
            'antes': {
                'mae': mae_lr,
                'mse': mse_lr,
                'r2': r2_lr
            },
            'depois': {
                'mae': None,
                'mse': None,
                'r2': None
            },
            'params': None
        }
        
        # 2. Random Forest com GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        print("\nIniciando busca em grade...")
        grid_search.fit(X, y)
        
        # Avaliar melhor modelo
        y_pred_rf = grid_search.predict(X)
        mae_rf = mean_absolute_error(y, y_pred_rf)
        mse_rf = mean_squared_error(y, y_pred_rf)
        r2_rf = r2_score(y, y_pred_rf)
        
        print("\nRandom Forest (Otimizado):")
        print(f"MAE: {mae_rf:.2f}")
        print(f"MSE: {mse_rf:.2f}")
        print(f"R²: {r2_rf:.2f}")
        print(f"Melhores parâmetros: {grid_search.best_params_}")
        
        # Atualizar métricas
        self.metricas_otimizacao['consultas']['depois'] = {
            'mae': mae_rf,
            'mse': mse_rf,
            'r2': r2_rf
        }
        self.metricas_otimizacao['consultas']['params'] = grid_search.best_params_
        
        # Guardar ambos os modelos
        self.modelo_consultas_lr = lr
        self.modelo_consultas_rf = grid_search.best_estimator_
        
        return grid_search.best_estimator_

    def _otimizar_modelo_medicamentos(self, X, y):
        """
        Otimiza o modelo de Random Forest para classificação de medicamentos
        Pergunta 3: Qual a probabilidade de um paciente precisar de medicamentos além da terapia?
        """
        print("\nOtimização - Modelo de Medicamentos:")
        
        # 1. Modelo Base (Logistic Regression) para comparação
        lr = LogisticRegression(random_state=42)
        lr.fit(X, y)
        y_pred_lr = lr.predict(X)
        acc_lr = accuracy_score(y, y_pred_lr)
        f1_lr = f1_score(y, y_pred_lr)
        
        print("\nRegressão Logística (Base):")
        print(f"Acurácia: {acc_lr:.2f}")
        print(f"F1-Score: {f1_lr:.2f}")
        
        # Armazenar métricas
        self.metricas_otimizacao['medicamentos'] = {
            'antes': {
                'accuracy': acc_lr,
                'f1': f1_lr
            },
            'depois': {
                'accuracy': None,
                'f1': None
            },
            'params': None
        }
        
        # 2. Random Forest com GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced', None]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        print("\nIniciando busca em grade...")
        grid_search.fit(X, y)
        
        # Avaliar melhor modelo
        y_pred_rf = grid_search.predict(X)
        acc_rf = accuracy_score(y, y_pred_rf)
        f1_rf = f1_score(y, y_pred_rf)
        
        print("\nRandom Forest (Otimizado):")
        print(f"Acurácia: {acc_rf:.2f}")
        print(f"F1-Score: {f1_rf:.2f}")
        print(f"Melhores parâmetros: {grid_search.best_params_}")
        
        # Atualizar métricas
        self.metricas_otimizacao['medicamentos']['depois'] = {
            'accuracy': acc_rf,
            'f1': f1_rf
        }
        self.metricas_otimizacao['medicamentos']['params'] = grid_search.best_params_
        
        # Guardar ambos os modelos
        self.modelo_medicamentos_lr = lr
        self.modelo_medicamentos_rf = grid_search.best_estimator_
        
        return grid_search.best_estimator_

    def _otimizar_modelo_demanda(self, X, y):
        """
        Otimiza o modelo de Random Forest para previsão de demanda
        Pergunta 2: Qual será a demanda futura por serviços de saúde na rede pública?
        """
        print("\nOtimização - Modelo de Demanda:")
        
        # 1. Modelo Base (Linear Regression) para comparação
        lr = LinearRegression()
        lr.fit(X, y)
        y_pred_lr = lr.predict(X)
        mae_lr = mean_absolute_error(y, y_pred_lr)
        mse_lr = mean_squared_error(y, y_pred_lr)
        r2_lr = r2_score(y, y_pred_lr)
        
        print("\nModelo Linear (Base):")
        print(f"MAE: {mae_lr:.2f}")
        print(f"MSE: {mse_lr:.2f}")
        print(f"R²: {r2_lr:.2f}")
        
        # Armazenar métricas
        self.metricas_otimizacao['demanda'] = {
            'antes': {
                'mae': mae_lr,
                'mse': mse_lr,
                'r2': r2_lr
            },
            'depois': {
                'mae': None,
                'mse': None,
                'r2': None
            },
            'params': None
        }
        
        # 2. Random Forest com GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        print("\nIniciando busca em grade...")
        grid_search.fit(X, y)
        
        # Avaliar melhor modelo
        y_pred_rf = grid_search.predict(X)
        mae_rf = mean_absolute_error(y, y_pred_rf)
        mse_rf = mean_squared_error(y, y_pred_rf)
        r2_rf = r2_score(y, y_pred_rf)
        
        print("\nRandom Forest (Otimizado):")
        print(f"MAE: {mae_rf:.2f}")
        print(f"MSE: {mse_rf:.2f}")
        print(f"R²: {r2_rf:.2f}")
        print(f"Melhores parâmetros: {grid_search.best_params_}")
        
        # Atualizar métricas
        self.metricas_otimizacao['demanda']['depois'] = {
            'mae': mae_rf,
            'mse': mse_rf,
            'r2': r2_rf
        }
        self.metricas_otimizacao['demanda']['params'] = grid_search.best_params_
        
        # Guardar ambos os modelos
        self.modelo_demanda_lr = lr
        self.modelo_demanda_rf = grid_search.best_estimator_
        
        return grid_search.best_estimator_

    def treinar_modelos(self, df):
        try:
            print("\n=== Iniciando treinamento dos modelos ===")
            
            # Verificar colunas necessárias
            colunas_necessarias = ['Numero_de_Consultas', 'Tratamento'] + self.colunas_basicas
            colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]
            if colunas_faltantes:
                raise ValueError(f"Colunas faltantes no DataFrame: {colunas_faltantes}")
            
            # Remover modelos antigos
            self.remover_modelos_salvos()
            self.limpar_modelos()
            
            print(f"Colunas disponíveis no DataFrame: {df.columns.tolist()}")
            print(f"Colunas básicas que serão usadas: {self.colunas_basicas}")
            
            # Preparar dados
            print("\nPreparando dados para treinamento...")
            X_consultas_norm, X_medicamentos_norm, X_demanda_norm, y_consultas, y_medicamentos, y_demanda = self._preparar_dados(df)
            
            # 1. Treinamento dos modelos para número de consultas
            print("\n1. Treinando modelos para previsão de número de consultas...")
            self.modelo_consultas_rf = self._otimizar_modelo_consultas(X_consultas_norm, y_consultas)
            
            # 2. Treinamento dos modelos para medicamentos
            print("\n2. Treinando modelos para previsão de necessidade de medicamentos...")
            self.modelo_medicamentos_rf = self._otimizar_modelo_medicamentos(X_medicamentos_norm, y_medicamentos)
            
            # 3. Treinamento dos modelos para demanda
            print("\n3. Treinando modelos para predição de demanda...")
            self.modelo_demanda_rf = self._otimizar_modelo_demanda(X_demanda_norm, y_demanda)
            
            print("\n=== Treinamento dos modelos concluído com sucesso! ===")
            
            # Salvar modelos
            self.salvar_modelos()
            
            return True
            
        except Exception as e:
            print("\n=== ERRO NO TREINAMENTO DOS MODELOS ===")
            print(f"Tipo do erro: {type(e).__name__}")
            print(f"Mensagem de erro: {str(e)}")
            raise ValueError(f"Erro ao treinar modelos: {str(e)}")

    def _preparar_dados(self, df):
        """
        Prepara os dados para treinamento e predição
        """
        print("\nPreparando dados...")
        
        # Criar cópia do DataFrame
        df_prep = df.copy()
        
        # Converter colunas numéricas
        for col in self.colunas_numericas:
            if col in df_prep.columns:
                df_prep[col] = pd.to_numeric(df_prep[col], errors='coerce')
        
        # Codificar variáveis categóricas (exceto Tratamento)
        for coluna, encoder in self.label_encoders.items():
            if coluna in df_prep.columns:
                df_prep[coluna] = encoder.fit_transform(df_prep[coluna].astype(str))
        
        # Garantir a ordem correta das colunas
        colunas_modelo = [col for col in self.colunas_basicas if col != 'Tratamento']
        X_consultas = df_prep[colunas_modelo].copy()
        X_medicamentos = df_prep[colunas_modelo].copy()
        X_demanda = df_prep[colunas_modelo].copy()
        
        # Aplicar scaling
        self.scaler_consultas.fit(X_consultas)
        self.scaler_medicamentos.fit(X_medicamentos)
        self.scaler_demanda.fit(X_demanda)
        
        X_consultas_norm = self.scaler_consultas.transform(X_consultas)
        X_medicamentos_norm = self.scaler_medicamentos.transform(X_medicamentos)
        X_demanda_norm = self.scaler_demanda.transform(X_demanda)
        
        # 1. Preparar target para consultas
        if 'Numero_de_Consultas' not in df_prep.columns:
            raise ValueError("Coluna 'Numero_de_Consultas' não encontrada no DataFrame")
        y_consultas = df_prep['Numero_de_Consultas'].values
        
        # 2. Preparar target para medicamentos
        if 'Tratamento' not in df_prep.columns:
            raise ValueError("Coluna 'Tratamento' não encontrada no DataFrame")
        
        print("\nDistribuição de tratamentos:")
        print(df_prep['Tratamento'].value_counts())
        
        y_medicamentos = df_prep['Tratamento'].apply(lambda x: 1 if x in ['Medicamentos', 'Ambos'] else 0).values
        
        print("\nDistribuição de classes para medicamentos:")
        unique, counts = np.unique(y_medicamentos, return_counts=True)
        print(dict(zip(unique, counts)))
        
        if len(np.unique(y_medicamentos)) < 2:
            raise ValueError("Dados não contêm exemplos suficientes de ambas as classes para treinar o modelo de medicamentos")
        
        # 3. Preparar target para demanda
        demanda_grupo = df_prep.groupby(['Regiao', 'Ano_Diagnostico'])['Numero_de_Consultas'].sum().reset_index()
        y_demanda = demanda_grupo['Numero_de_Consultas'].values
        X_demanda_norm = X_demanda_norm[:len(y_demanda)]
        
        print("\nDados preparados com sucesso!")
        print(f"Shapes dos dados:")
        print(f"X_consultas: {X_consultas_norm.shape}, y_consultas: {len(y_consultas)}")
        print(f"X_medicamentos: {X_medicamentos_norm.shape}, y_medicamentos: {len(y_medicamentos)}")
        print(f"X_demanda: {X_demanda_norm.shape}, y_demanda: {len(y_demanda)}")
        
        return X_consultas_norm, X_medicamentos_norm, X_demanda_norm, y_consultas, y_medicamentos, y_demanda

    def _preparar_dados_predicao(self, dados):
        """
        Prepara dados de entrada para predição
        """
        try:
            # Criar DataFrame com os dados de entrada
            df_pred = pd.DataFrame([dados])
            
            # Converter colunas numéricas
            for col in self.colunas_numericas:
                if col in df_pred.columns:
                    df_pred[col] = pd.to_numeric(df_pred[col], errors='coerce')
            
            # Codificar variáveis categóricas
            for coluna, encoder in self.label_encoders.items():
                if coluna in df_pred.columns:
                    try:
                        df_pred[coluna] = encoder.transform(df_pred[coluna].astype(str))
                    except:
                        print(f"Erro ao codificar a coluna {coluna}. Verificar se os valores são válidos.")
                        print(f"Valores permitidos para {coluna}: {list(encoder.classes_)}")
                        raise
            
            # Garantir a ordem correta das colunas
            colunas_modelo = [col for col in self.colunas_basicas if col != 'Tratamento']
            X_consultas = df_pred[colunas_modelo].copy()
            X_medicamentos = df_pred[colunas_modelo].copy()
            X_demanda = df_pred[colunas_modelo].copy()
            
            # Aplicar scaling mantendo a ordem das features
            X_consultas_norm = self.scaler_consultas.transform(X_consultas)
            X_medicamentos_norm = self.scaler_medicamentos.transform(X_medicamentos)
            X_demanda_norm = self.scaler_demanda.transform(X_demanda)
            
            return X_consultas_norm, X_medicamentos_norm, X_demanda_norm
            
        except Exception as e:
            print("\n=== ERRO NA PREPARAÇÃO DOS DADOS ===")
            print(f"Dados de entrada: {dados}")
            print(f"Colunas numéricas: {self.colunas_numericas}")
            print(f"Colunas categóricas: {self.colunas_categoricas}")
            print(f"Tipo do erro: {type(e).__name__}")
            print(f"Mensagem de erro: {str(e)}")
            raise

    def salvar_modelos(self, diretorio='modelos'):
        """
        Salva os modelos treinados, transformadores e métricas
        """
        try:
            os.makedirs(diretorio, exist_ok=True)
            
            # Salvar modelos Random Forest e lineares
            joblib.dump(self.modelo_consultas_rf, f'{diretorio}/modelo_consultas_rf.pkl')
            joblib.dump(self.modelo_consultas_lr, f'{diretorio}/modelo_consultas_lr.pkl')
            joblib.dump(self.modelo_medicamentos_rf, f'{diretorio}/modelo_medicamentos_rf.pkl')
            joblib.dump(self.modelo_medicamentos_lr, f'{diretorio}/modelo_medicamentos_lr.pkl')
            joblib.dump(self.modelo_demanda_rf, f'{diretorio}/modelo_demanda_rf.pkl')
            joblib.dump(self.modelo_demanda_lr, f'{diretorio}/modelo_demanda_lr.pkl')
            
            # Salvar transformadores
            joblib.dump(self.scaler_consultas, f'{diretorio}/scaler_consultas.pkl')
            joblib.dump(self.scaler_medicamentos, f'{diretorio}/scaler_medicamentos.pkl')
            joblib.dump(self.scaler_demanda, f'{diretorio}/scaler_demanda.pkl')
            
            # Salvar label encoders
            for coluna, encoder in self.label_encoders.items():
                joblib.dump(encoder, f'{diretorio}/encoder_{coluna}.pkl')
            
            # Salvar métricas de otimização
            joblib.dump(self.metricas_otimizacao, f'{diretorio}/metricas_otimizacao.pkl')
            
            print("Modelos e métricas salvos com sucesso!")
            return True
            
        except Exception as e:
            print("\n=== ERRO AO SALVAR MODELOS ===")
            print(f"Tipo do erro: {type(e).__name__}")
            print(f"Mensagem de erro: {str(e)}")
            return False

    def carregar_modelos(self):
        """
        Carrega os modelos salvos e suas métricas
        """
        try:
            print("\nCarregando modelos...")
            
            # Carregar modelos Random Forest e lineares
            self.modelo_consultas_rf = joblib.load('modelos/modelo_consultas_rf.pkl')
            self.modelo_consultas_lr = joblib.load('modelos/modelo_consultas_lr.pkl')
            self.modelo_medicamentos_rf = joblib.load('modelos/modelo_medicamentos_rf.pkl')
            self.modelo_medicamentos_lr = joblib.load('modelos/modelo_medicamentos_lr.pkl')
            self.modelo_demanda_rf = joblib.load('modelos/modelo_demanda_rf.pkl')
            self.modelo_demanda_lr = joblib.load('modelos/modelo_demanda_lr.pkl')
            
            # Carregar transformadores
            self.scaler_consultas = joblib.load('modelos/scaler_consultas.pkl')
            self.scaler_medicamentos = joblib.load('modelos/scaler_medicamentos.pkl')
            self.scaler_demanda = joblib.load('modelos/scaler_demanda.pkl')
            
            # Carregar label encoders
            for coluna in self.label_encoders.keys():
                self.label_encoders[coluna] = joblib.load(f'modelos/encoder_{coluna}.pkl')
            
            # Carregar métricas de otimização
            try:
                self.metricas_otimizacao = joblib.load('modelos/metricas_otimizacao.pkl')
            except:
                print("Métricas não encontradas, serão recalculadas no próximo treinamento")
            
            print("Modelos carregados com sucesso!")
            return True
            
        except Exception as e:
            print("\n=== ERRO AO CARREGAR MODELOS ===")
            print(f"Tipo do erro: {type(e).__name__}")
            print(f"Mensagem de erro: {str(e)}")
            return False

    def interface_predicao(self):
        """Interface para fazer predições via Streamlit"""
        st.title("Predições do Modelo de Autismo")
        
        # Menu lateral para escolher entre predição e variação do modelo
        menu = st.sidebar.selectbox(
            "Menu",
            ["Fazer Predições", "Variação do Modelo"]
        )
        
        if menu == "Variação do Modelo":
            self._mostrar_variacao_modelo()
        else:
            self._mostrar_interface_predicao()
    
    def _mostrar_variacao_modelo(self):
        """Mostra a comparação entre modelos antes e depois da otimização"""
        st.header("📈 Comparação dos Modelos: Antes e Depois da Otimização")
        
        # Tentar carregar os modelos e métricas se necessário
        if self.modelo_consultas_rf is None:
            if not self.carregar_modelos():
                st.error("❌ Erro ao carregar os modelos. Por favor, verifique se os modelos foram treinados corretamente.")
                return
        
        # Se não temos métricas mas temos modelos, recalcular as métricas
        if self.metricas_otimizacao['consultas']['antes']['mae'] is None:
            st.info("🔄 Recalculando métricas dos modelos...")
            try:
                from src.gerar_base_sintetica import gerar_base_sintetica
                df = gerar_base_sintetica(1000)
                
                # Preparar dados
                X_consultas_norm, X_medicamentos_norm, X_demanda_norm, y_consultas, y_medicamentos, y_demanda = self._preparar_dados(df)
                
                # Recalcular métricas para consultas
                y_pred_lr = self.modelo_consultas_lr.predict(X_consultas_norm)
                y_pred_rf = self.modelo_consultas_rf.predict(X_consultas_norm)
                
                self.metricas_otimizacao['consultas']['antes'] = {
                    'mae': mean_absolute_error(y_consultas, y_pred_lr),
                    'mse': mean_squared_error(y_consultas, y_pred_lr),
                    'r2': r2_score(y_consultas, y_pred_lr)
                }
                
                self.metricas_otimizacao['consultas']['depois'] = {
                    'mae': mean_absolute_error(y_consultas, y_pred_rf),
                    'mse': mean_squared_error(y_consultas, y_pred_rf),
                    'r2': r2_score(y_consultas, y_pred_rf)
                }
                
                # Recalcular métricas para medicamentos
                y_pred_lr = self.modelo_medicamentos_lr.predict(X_medicamentos_norm)
                y_pred_rf = self.modelo_medicamentos_rf.predict(X_medicamentos_norm)
                
                self.metricas_otimizacao['medicamentos']['antes'] = {
                    'accuracy': accuracy_score(y_medicamentos, y_pred_lr),
                    'f1': f1_score(y_medicamentos, y_pred_lr)
                }
                
                self.metricas_otimizacao['medicamentos']['depois'] = {
                    'accuracy': accuracy_score(y_medicamentos, y_pred_rf),
                    'f1': f1_score(y_medicamentos, y_pred_rf)
                }
                
                # Recalcular métricas para demanda
                y_pred_lr = self.modelo_demanda_lr.predict(X_demanda_norm)
                y_pred_rf = self.modelo_demanda_rf.predict(X_demanda_norm)
                
                self.metricas_otimizacao['demanda']['antes'] = {
                    'mae': mean_absolute_error(y_demanda, y_pred_lr),
                    'mse': mean_squared_error(y_demanda, y_pred_lr),
                    'r2': r2_score(y_demanda, y_pred_lr)
                }
                
                self.metricas_otimizacao['demanda']['depois'] = {
                    'mae': mean_absolute_error(y_demanda, y_pred_rf),
                    'mse': mean_squared_error(y_demanda, y_pred_rf),
                    'r2': r2_score(y_demanda, y_pred_rf)
                }
                
                # Salvar métricas recalculadas
                joblib.dump(self.metricas_otimizacao, 'modelos/metricas_otimizacao.pkl')
                st.success("✅ Métricas recalculadas com sucesso!")
                
            except Exception as e:
                st.error(f"❌ Erro ao recalcular métricas: {str(e)}")
                return
        
        # Mostrar métricas
        self._mostrar_metricas_consultas()
        st.markdown("---")
        self._mostrar_metricas_demanda()
        st.markdown("---")
        self._mostrar_metricas_medicamentos()
    
    def _mostrar_metricas_consultas(self):
        """Mostra métricas de consultas"""
        st.subheader("1️⃣ Previsão de Consultas Médicas")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Antes da Otimização (Linear)**")
            metricas = self.metricas_otimizacao['consultas']['antes']
            st.metric("MAE (Erro Médio Absoluto)", f"{metricas['mae']:.2f}" if metricas['mae'] is not None else "N/A")
            st.metric("MSE (Erro Quadrático Médio)", f"{metricas['mse']:.2f}" if metricas['mse'] is not None else "N/A")
            st.metric("R² (Coeficiente de Determinação)", f"{metricas['r2']:.2f}" if metricas['r2'] is not None else "N/A")
        
        with col2:
            st.markdown("**Depois da Otimização (Random Forest)**")
            metricas = self.metricas_otimizacao['consultas']['depois']
            st.metric("MAE (Erro Médio Absoluto)", f"{metricas['mae']:.2f}" if metricas['mae'] is not None else "N/A")
            st.metric("MSE (Erro Quadrático Médio)", f"{metricas['mse']:.2f}" if metricas['mse'] is not None else "N/A")
            st.metric("R² (Coeficiente de Determinação)", f"{metricas['r2']:.2f}" if metricas['r2'] is not None else "N/A")
        
        if self.metricas_otimizacao['consultas']['params']:
            with st.expander("Ver parâmetros ótimos"):
                st.json(self.metricas_otimizacao['consultas']['params'])
    
    def _mostrar_metricas_demanda(self):
        """Mostra métricas de demanda"""
        st.subheader("2️⃣ Previsão de Demanda Futura")
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**Antes da Otimização (Linear)**")
            metricas = self.metricas_otimizacao['demanda']['antes']
            st.metric("MAE (Erro Médio Absoluto)", f"{metricas['mae']:.2f}" if metricas['mae'] is not None else "N/A")
            st.metric("MSE (Erro Quadrático Médio)", f"{metricas['mse']:.2f}" if metricas['mse'] is not None else "N/A")
            st.metric("R² (Coeficiente de Determinação)", f"{metricas['r2']:.2f}" if metricas['r2'] is not None else "N/A")
        
        with col4:
            st.markdown("**Depois da Otimização (Random Forest)**")
            metricas = self.metricas_otimizacao['demanda']['depois']
            st.metric("MAE (Erro Médio Absoluto)", f"{metricas['mae']:.2f}" if metricas['mae'] is not None else "N/A")
            st.metric("MSE (Erro Quadrático Médio)", f"{metricas['mse']:.2f}" if metricas['mse'] is not None else "N/A")
            st.metric("R² (Coeficiente de Determinação)", f"{metricas['r2']:.2f}" if metricas['r2'] is not None else "N/A")
        
        if self.metricas_otimizacao['demanda']['params']:
            with st.expander("Ver parâmetros ótimos"):
                st.json(self.metricas_otimizacao['demanda']['params'])
    
    def _mostrar_metricas_medicamentos(self):
        """Mostra métricas de medicamentos"""
        st.subheader("3️⃣ Classificação de Necessidade de Medicamentos")
        col5, col6 = st.columns(2)
        
        with col5:
            st.markdown("**Antes da Otimização (Logistic)**")
            metricas = self.metricas_otimizacao['medicamentos']['antes']
            st.metric("Acurácia (Taxa de Acerto)", f"{metricas['accuracy']:.2%}" if metricas['accuracy'] is not None else "N/A")
            st.metric("F1-Score (Média Harmônica entre Precisão e Recall)", f"{metricas['f1']:.2%}" if metricas['f1'] is not None else "N/A")
        
        with col6:
            st.markdown("**Depois da Otimização (Random Forest)**")
            metricas = self.metricas_otimizacao['medicamentos']['depois']
            st.metric("Acurácia (Taxa de Acerto)", f"{metricas['accuracy']:.2%}" if metricas['accuracy'] is not None else "N/A")
            st.metric("F1-Score (Média Harmônica entre Precisão e Recall)", f"{metricas['f1']:.2%}" if metricas['f1'] is not None else "N/A")
        
        if self.metricas_otimizacao['medicamentos']['params']:
            with st.expander("Ver parâmetros ótimos"):
                st.json(self.metricas_otimizacao['medicamentos']['params'])
    
    def _mostrar_interface_predicao(self):
        """Mostra interface de predições"""
        st.subheader(" Dados do Paciente")
        
        col1, col2 = st.columns(2)
        with col1:
            idade = st.number_input("Idade:", min_value=0, max_value=100, value=10)
            regiao = st.selectbox("Região:", ['Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul'])
            renda = st.selectbox("Renda Familiar:", ['Baixa', 'Média', 'Alta'])
        
        with col2:
            ano_diagnostico = st.number_input("Ano do Diagnóstico:", min_value=2012, max_value=2024, value=2023)
            zona = st.selectbox("Zona:", ['Urbana', 'Rural'])
            acesso = st.selectbox("Acesso a Serviços:", ['Sim', 'Não'])
        
        if st.button("Fazer Predições", type="primary"):
            try:
                with st.spinner("Realizando predições..."):
                    dados_entrada = {
                        'Idade': idade,
                        'Ano_Diagnostico': ano_diagnostico,
                        'Regiao': regiao,
                        'Zona': zona,
                        'Renda_Familiar': renda,
                        'Acesso_a_Servicos': acesso
                    }
                    
                    predicoes = self.fazer_predicoes(dados_entrada)
                
                st.markdown("---")
                st.header(" Resultados das Predições")
                
                # 1. Número de Consultas
                st.subheader("1️⃣ Quantas consultas médicas o paciente realizará no ano?")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        " Random Forest (Otimizado)",
                        f"{predicoes['consultas_rf']} consultas/ano"
                    )
                with col2:
                    st.metric(
                        " Regressão Linear",
                        f"{predicoes['consultas_lr']} consultas/ano"
                    )
                
                # 2. Demanda Futura
                st.subheader("2️⃣ Qual será a demanda futura por serviços de saúde na região?")
                col3, col4 = st.columns(2)
                with col3:
                    st.metric(
                        " Random Forest (Otimizado)",
                        f"{predicoes['demanda_rf']} consultas totais"
                    )
                with col4:
                    st.metric(
                        " Regressão Linear",
                        f"{predicoes['demanda_lr']} consultas totais"
                    )
                
                # 3. Probabilidade de Medicamentos
                st.subheader("3️⃣ Qual a probabilidade do paciente precisar de medicamentos?")
                col5, col6 = st.columns(2)
                with col5:
                    st.metric(
                        " Random Forest (Otimizado)",
                        f"{predicoes['prob_medicamentos_rf']}%"
                    )
                with col6:
                    st.metric(
                        " Regressão Logística",
                        f"{predicoes['prob_medicamentos_lr']}%"
                    )
                
                # Informações adicionais
                st.markdown("---")
                st.subheader(" Informações sobre os Modelos")
                
                with st.expander("Ver detalhes dos modelos"):
                    st.markdown("""
                    **Random Forest (Otimizado)**
                    - Usa GridSearchCV para encontrar os melhores parâmetros
                    - Otimiza métricas específicas para cada tipo de predição
                    - Geralmente mais preciso que modelos lineares
                    
                    **Modelos Lineares (Base)**
                    - Regressão Linear para consultas e demanda
                    - Regressão Logística para probabilidade de medicamentos
                    - Serve como baseline para comparação
                    
                    **Métricas de Avaliação**
                    - Consultas e Demanda: MAE, MSE, R²
                    - Medicamentos: Acurácia, F1-Score
                    """)
                
            except Exception as e:
                st.error(f" Erro ao fazer predições: {str(e)}")
    
    def modelos_existem(self):
        """
        Verifica se os modelos existem
        """
        return os.path.exists('modelos/modelo_consultas_rf.pkl')

    def fazer_predicoes(self, dados_entrada):
        """
        Faz predições usando os modelos treinados e otimizados
        """
        try:
            # Verificar se os modelos existem e carregar se necessário
            if self.modelo_consultas_rf is None:
                if not self.carregar_modelos():
                    raise ValueError("Modelos não encontrados! Por favor, treine os modelos primeiro.")
            
            # Preparar dados
            X_consultas_norm, X_medicamentos_norm, X_demanda_norm = self._preparar_dados_predicao(dados_entrada)
            
            # 1. Quantas consultas médicas um paciente realizará no ano?
            consultas_rf = self.modelo_consultas_rf.predict(X_consultas_norm)[0]
            consultas_lr = self.modelo_consultas_lr.predict(X_consultas_norm)[0]
            
            # 2. Qual será a demanda futura por serviços de saúde na região?
            demanda_rf = self.modelo_demanda_rf.predict(X_demanda_norm)[0]
            demanda_lr = self.modelo_demanda_lr.predict(X_demanda_norm)[0]
            
            # 3. Qual a probabilidade de um paciente precisar de medicamentos além da terapia?
            prob_med_rf = self.modelo_medicamentos_rf.predict_proba(X_medicamentos_norm)[0][1]
            prob_med_lr = self.modelo_medicamentos_lr.predict_proba(X_medicamentos_norm)[0][1]
            
            return {
                'consultas_rf': round(consultas_rf),
                'consultas_lr': round(consultas_lr),
                'demanda_rf': round(demanda_rf, 2),
                'demanda_lr': round(demanda_lr, 2),
                'prob_medicamentos_rf': round(prob_med_rf * 100, 2),
                'prob_medicamentos_lr': round(prob_med_lr * 100, 2)
            }
            
        except Exception as e:
            print("\n=== ERRO AO FAZER PREDIÇÕES ===")
            print(f"Tipo do erro: {type(e).__name__}")
            print(f"Mensagem de erro: {str(e)}")
            raise

def main():
    """
    Função principal para executar a interface Streamlit
    """
    st.set_page_config(page_title="Modelo de Autismo", layout="wide")
    
    modelo = ModeloAutismo()
    
    # Menu lateral
    menu = st.sidebar.selectbox(
        "Selecione uma opção:",
        ["Treinamento", "Predições"]
    )
    
    if menu == "Treinamento":
        st.title("Treinamento dos Modelos de Autismo")
        
        # Botão para gerar dados sintéticos
        n_samples = st.number_input("Número de amostras para treinamento:", min_value=100, value=1000, step=100)
        
        if st.button("Gerar Dados e Treinar Modelos"):
            try:
                # Importar função de geração de dados
                from src.gerar_base_sintetica import gerar_base_sintetica
                
                # Gerar dados
                st.info("Gerando dados sintéticos...")
                df = gerar_base_sintetica(n_samples)
                st.success(f"Dados gerados com sucesso! Shape: {df.shape}")
                
                # Mostrar distribuição dos dados
                st.subheader("Distribuição dos Dados")
                st.write("Tratamentos:", df['Tratamento'].value_counts())
                st.write("Regiões:", df['Regiao'].value_counts())
                st.write("Média de consultas:", df['Numero_de_Consultas'].mean())
                
                # Treinar modelos
                st.info("Iniciando treinamento dos modelos...")
                modelo.treinar_modelos(df)
                st.success("Modelos treinados e salvos com sucesso!")
                
            except Exception as e:
                st.error(f"Erro durante o treinamento: {str(e)}")
    else:
        modelo.interface_predicao()

if __name__ == "__main__":
    # Criar diretório de modelos se não existir
    os.makedirs('modelos', exist_ok=True)
    main()