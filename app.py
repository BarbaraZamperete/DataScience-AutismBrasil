import streamlit as st
from streamlit_option_menu import option_menu

from src.gerar_base_sintetica import main as gerar_base
from src.analises_visualizacoes import main as analises
from src.temp import main as teste

st.set_page_config(
    page_title="Análise de Dados de Saúde",
    layout="centered"
)

with st.sidebar:
    selected = option_menu(
        menu_title="Menu Principal",
        options=["Home", "Gerar Base Sintética", "Análises e Visualizações", "teste"],
        icons=["house", "database", "bar-chart"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "Home":

    st.header(':bar_chart: ANÁLISE DE DADOS APLICADOS A SAÚDE: AUTISMO NO BRASIL', divider='rainbow')

    st.markdown(
        '''
        <strong>Por:</strong><br/>
            <h5>
                :man-raising-hand: Everton Reis <br/>
                :woman-raising-hand: Bárbara Zamperete
            </h5>
        ''',
        unsafe_allow_html=True
    )

    st.write("""
    Este projeto é uma análise de uma base de dados sintética sobre autismo no Brasil, gerada através de um algoritmo de machine learning. 
    O objetivo é explorar e entender as características dessa base de dados, além de treinar modelos de machine learning 
    para prever informações relevantes.

    **Objetivos do Projeto:**
    - Analisar gráficos e estatísticas da base de dados.
    - Treinar modelos de machine learning para prever informações sobre autismo.
    - Compreender a importância da análise de dados na saúde pública.

    **Por que isso é importante?**
    A análise de dados sobre autismo é crucial para entender melhor essa condição, 
    identificar padrões e auxiliar na formulação de políticas públicas e intervenções.

    Use o menu lateral para navegar entre as páginas e explorar as funcionalidades do projeto.
    """)

elif selected == "Gerar Base Sintética":

    st.header(':sparkles: GERAÇÃO DE BASE SINTÉTICA DE DADOS')

    st.write("""
    Esta funcionalidade permite gerar uma base de dados sintética sobre autismo no Brasil. 
    A base é criada com amostras aleatórias, permitindo simular diferentes cenários e características 
    que podem ser úteis para análises e treinamentos de modelos de machine learning.
    """)


    st.subheader("📊 Como os Dados São Gerados:")
    st.write("""
    Você pode definir o número de amostras que deseja gerar. A geração dos dados segue algumas tendências programadas:
    - **ID**: Um identificador único para cada amostra, gerado sequencialmente.
    - **Estado**: Selecionado aleatoriamente entre os estados do Brasil, com uma distribuição uniforme.
    - **Idade**: Valores aleatórios entre 5 e 45 anos, representando a faixa etária de interesse.
    - **Tipo de Diagnóstico**: A distribuição é programada para refletir uma prevalência maior de diagnósticos leves (60%), moderados (30%) e graves (10%).
    - **Ano do Diagnóstico**: Anos aleatórios entre 2005 e 2025, simulando diagnósticos ao longo do tempo.
    - **Tratamento**: A distribuição dos tratamentos é de 40% para terapias, 40% para medicamentos e 20% para ambos.
    - **Acesso a Serviços**: 70% da amostra tem acesso a serviços, enquanto 30% não têm.
    - **Região**: A distribuição geográfica é programada para refletir uma maior concentração no Sudeste (40%) e menor no Centro-Oeste (10%).
    - **Número de Consultas**: Gerado a partir de uma distribuição de Poisson, com um valor médio de 8, garantindo que a maioria das amostras tenha pelo menos uma consulta.
    - **Apoio Familiar**: 70% da amostra tem apoio familiar, enquanto 30% não têm.
    - **Renda Familiar**: A distribuição é de 50% para baixa, 40% para média e 10% para alta renda.
    - **Tipo de Serviço**: 80% da amostra utiliza serviços públicos, enquanto 20% utiliza serviços privados.
    - **Zona**: 70% da amostra é da zona urbana e 30% da zona rural.
    """)

    container = st.container(border=True)
    
    n_samples = container.number_input("Número de amostras:", min_value=100, max_value=10000, value=1000)

    if container.button("Gerar Base Sintética"):
        df = gerar_base(n_samples)
        df.to_csv("base_ajustada_realista.csv", sep=';', index=False)
        container.success(f"Base sintética com {n_samples} amostras gerada e salva como 'base_ajustada_realista.csv'.")
        container.write("Visualização dos primeiros registros:")
        container.dataframe(df.head())

    st.write("Após a geração, você pode visualizar os dados e realizar análises adicionais.")

elif selected == "Análises e Visualizações":
    analises()
    
elif selected == "teste":
    teste()