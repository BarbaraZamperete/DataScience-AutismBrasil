import streamlit as st
import os
from src.ia.modelos import ModeloAutismo

def main():
    """
    Interface principal da aplicação
    """
    st.set_page_config(
        page_title="ANÁLISE DE DADOS APLICADOS A SAÚDE",
        page_icon="🧩",
        layout="wide"
    )
    
    # Título principal
    st.title("🧩 ANÁLISE DE DADOS APLICADOS A SAÚDE:")
    st.title("Sistema de Análise de Autismo no Brasil")
    st.markdown("---")
    
    # Menu lateral
    st.sidebar.title("Menu")
    opcao = st.sidebar.selectbox(
        "Escolha uma opção:",
        ["Página Inicial", "Treinamento de Modelos", "Realizar Predições"]
    )

    # Submenu para Realizar Predições
    submenu = None
    if opcao == "Realizar Predições":
        submenu = st.sidebar.selectbox(
            "Escolha o tipo de análise:",
            ["Fazer Predições", "📊 Visualizar Gráficos"]
        )
    
    # Criar instância do modelo
    modelo = ModeloAutismo()
    
    if opcao == "Página Inicial":
        st.write("""
        ### Por:  
        🙋‍♂️ Everton Reis  
        🙋‍♀️ Bárbara Zamperete  

        Este projeto é uma análise de uma base de dados sintética sobre autismo no Brasil, gerada através de um algoritmo de machine learning. O objetivo é explorar e entender as características dessa base de dados, além de treinar modelos de machine learning para prever informações relevantes.  

        ## 🎯 Objetivos do Projeto:  

        - 📊 **Analisar gráficos e estatísticas** da base de dados.  
        - 🤖 **Treinar modelos de machine learning** para prever informações sobre autismo.  
        - 🏥 **Compreender a importância da análise de dados na saúde pública.**  

        ## 📌 Por que isso é importante?  

        A análise de dados sobre autismo é crucial para entender melhor essa condição, identificar padrões e auxiliar na formulação de políticas públicas e intervenções.  

        💡 **Use o menu lateral para navegar entre as páginas e explorar as funcionalidades do projeto.**  

        Este sistema utiliza modelos de Machine Learning para auxiliar na análise de dados relacionados ao autismo.
        
        ## ⚙️ Funcionalidades:
        1. **Treinamento de Modelos**: Treine os modelos usando dados sintéticos ou reais
        2. **Realizar Predições**: Faça predições sobre:
           - Número de consultas necessárias
           - Probabilidade de necessitar medicamentos
           - Demanda de serviços por região
        
        ## 🚀 Como usar:
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
            if submenu == "Fazer Predições":
                if not modelo.modelos_existem():
                    st.warning("⚠️ Os modelos ainda não foram treinados. Por favor, acesse a seção de Treinamento primeiro.")
                else:
                    modelo.interface_predicao()
            elif submenu == "📊 Visualizar Gráficos":
                from src.analises_visualizacoes import visualizar_graficos
                visualizar_graficos()

if __name__ == "__main__":
    # Criar diretório de modelos se não existir
    os.makedirs('modelos', exist_ok=True)
    main()