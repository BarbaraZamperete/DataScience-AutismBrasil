import streamlit as st
import os
from src.ia.modelos import ModeloAutismo

def main():
    """
    Interface principal da aplicação
    """
    st.set_page_config(
        page_title="Sistema de Análise de Autismo",
        page_icon="🧩",
        layout="wide"
    )
    
    # Título principal
    st.title("🧩 Sistema de Análise de Autismo")
    st.markdown("---")
    
    # Menu lateral
    st.sidebar.title("Menu")
    opcao = st.sidebar.selectbox(
        "Selecione uma opção:",
        ["Página Inicial", "Treinamento de Modelos", "Realizar Predições"]
    )
    
    # Criar instância do modelo
    modelo = ModeloAutismo()
    
    if opcao == "Página Inicial":
        st.header("Bem-vindo ao Sistema de Análise de Autismo")
        st.write("""
        Este sistema utiliza modelos de Machine Learning para auxiliar na análise de dados relacionados ao autismo.
        
        ### Funcionalidades:
        1. **Treinamento de Modelos**: Treine os modelos usando dados sintéticos ou reais
        2. **Realizar Predições**: Faça predições sobre:
           - Número de consultas necessárias
           - Probabilidade de necessitar medicamentos
           - Demanda de serviços por região
        
        ### Como usar:
        1. Primeiro, acesse "Treinamento de Modelos" para treinar os modelos
        2. Depois, use "Realizar Predições" para fazer análises
        """)
        
    elif opcao == "Treinamento de Modelos":
        st.header("Treinamento dos Modelos")
        
        # Opções de treinamento
        metodo_treinamento = st.radio(
            "Selecione o método de treinamento:",
            ["Dados Sintéticos", "Carregar Dados Próprios"]
        )
        
        if metodo_treinamento == "Dados Sintéticos":
            n_samples = st.number_input(
                "Número de amostras para treinamento:",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100
            )
            
            if st.button("Gerar Dados e Treinar Modelos"):
                try:
                    # Importar função de geração de dados
                    from src.gerar_base_sintetica import gerar_base_sintetica
                    
                    # Gerar dados
                    with st.spinner("Gerando dados sintéticos..."):
                        df = gerar_base_sintetica(n_samples)
                        st.success(f"Dados gerados com sucesso! Shape: {df.shape}")
                    
                    # Mostrar distribuição dos dados
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Distribuição de Tratamentos")
                        st.write(df['Tratamento'].value_counts())
                    with col2:
                        st.subheader("Distribuição por Região")
                        st.write(df['Regiao'].value_counts())
                    
                    # Treinar modelos
                    with st.spinner("Treinando modelos..."):
                        modelo.treinar_modelos(df)
                        st.success("✅ Modelos treinados e salvos com sucesso!")
                    
                except Exception as e:
                    st.error(f"❌ Erro durante o treinamento: {str(e)}")
        
        else:
            st.write("Funcionalidade em desenvolvimento...")
            
    else:  # Realizar Predições
        if not modelo.modelos_existem():
            st.warning("⚠️ Modelos não encontrados! Por favor, treine os modelos primeiro.")
            if st.button("Treinar Modelos Agora"):
                try:
                    from src.gerar_base_sintetica import gerar_base_sintetica
                    with st.spinner("Gerando dados e treinando modelos..."):
                        df = gerar_base_sintetica(1000)
                        modelo.treinar_modelos(df)
                        st.success("✅ Modelos treinados com sucesso!")
                        st.experimental_rerun()
                except Exception as e:
                    st.error(f"❌ Erro durante o treinamento: {str(e)}")
        else:
            modelo.interface_predicao()

if __name__ == "__main__":
    # Criar diretório de modelos se não existir
    os.makedirs('modelos', exist_ok=True)
    main()